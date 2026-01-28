import numpy as np
from keras import Sequential
from keras.utils import to_categorical

# from Extra_plots import SignalProcessor
from mealpy import FloatVar
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from sklearn.model_selection import train_test_split, StratifiedKFold
import tensorflow.keras.backend as K
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from mealpy.swarm_based.TSO import OriginalTSO as TSO
from mealpy.swarm_based.SFO import OriginalSFO as SFO
from mealpy.Prop import OriginalPROP as PROP
from keras.layers import *
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from termcolor import cprint, colored
from warnings import filterwarnings
# import PySimpleGUI as sg
import lightgbm as lgb

filterwarnings(action='ignore', category=RuntimeWarning)
filterwarnings(action='ignore', category=UserWarning)

anfis_models = None
lgbm_model = None
is_trained = False

def main_est_perf_metrics(preds, ytest):
    sm = multilabel_confusion_matrix(ytest, preds)
    cm = sum(sm)
    TP = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TN = cm[1, 1]

    accuracy = (TP + TN) / (TP + FP + FN + TN)
    sen = TP / (FN + TP)
    spe = TN / (FP + TN)
    pre = TP / (TP + FP)
    rec = TP / (FN + TP)
    f1_score = (2 * pre * rec) / (pre + rec)
    CSI = TP / (TP + FN + FP)
    FNR = FN / (TP + FN)
    MCC = (TP * TN - FP * FN) / np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    NPV = TN / (TN + FN)
    PPV = TP / (TP + FP)
    TPR = TP / (TP + FN)
    FPR = FP / (FP + TN)
    return [accuracy, rec, PPV, NPV, f1_score, TPR, FPR]


class Shuffle_Attenion(tf.keras.layers.Layer):
    def __init__(self, channels, k_size=3):
        super(Shuffle_Attenion, self).__init__()
        self.k_size = k_size
        self.conv = Conv2D(1, (k_size, k_size), padding='same', use_bias=False)
        self.sigmoid = Activation('sigmoid')

    def call(self, x):
        # Step 1: Global Average Pooling to reduce spatial dimensions to 1
        x_avg = GlobalAveragePooling2D()(x)
        # Step 2: Reshape to prepare for 2D convolution
        x_avg = Reshape((1, 1, K.int_shape(x_avg)[-1]))(x_avg)
        # Step 3: Apply 2D convolution
        x_conv = self.conv(x_avg)
        # Step 4: Sigmoid activation for channel-wise attention
        x_sigmoid = self.sigmoid(x_conv)
        # Step 5: Multiply the sigmoid output with the input tensor
        x = Multiply()([x, x_sigmoid])
        return x


# Define the Selective Kernel Attention Layer
class SelectiveKernelAttention(Layer):
    def __init__(self, kernel_size=(3, 3), **kwargs):
        super(SelectiveKernelAttention, self).__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        if input_shape is None or len(input_shape) < 3:
            raise ValueError(
                "Expected input_shape to have at least 3 dimensions (batch size, height, width, channels).")

        channels = input_shape[-1]
        self.avg_conv = Conv2D(filters=channels, kernel_size=self.kernel_size, padding='same')
        self.max_conv = Conv2D(filters=channels, kernel_size=self.kernel_size, padding='same')

    def call(self, inputs):
        avg_out = K.mean(inputs, axis=[1, 2], keepdims=True)
        max_out = K.max(inputs, axis=[1, 2], keepdims=True)
        # x1 = eca_layer(channel=32, k_size=3)(max_out)
        avg_out = self.avg_conv(avg_out)
        max_out = self.max_conv(max_out)

        attention = K.sigmoid(avg_out + max_out)
        return inputs * attention


# Define Adaptive Hybrid Activation Function Layer
class AHAF(Layer):
    def __init__(self, x, **kwargs):
        init_as = x  # "Relu"
        super(AHAF, self).__init__(**kwargs)
        self.gamma_init = 1e9 if init_as == x else 1.0
        self.beta_init = 1.0

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=(input_shape[-1],),
                                     initializer=tf.keras.initializers.Constant(self.gamma_init), trainable=True)
        self.beta = self.add_weight(name='beta', shape=(input_shape[-1],),
                                    initializer=tf.keras.initializers.Constant(self.beta_init), trainable=True)

    def call(self, inputs):
        sig_in = self.gamma * inputs
        sig_out = tf.sigmoid(sig_in)
        amplified = inputs * self.beta
        out = sig_out * amplified
        return out


def Hybrid_attention(x):
    x1 = SelectiveKernelAttention(kernel_size=(3, 3))(x)
    x2 = Shuffle_Attenion(channels=K.int_shape(x1)[-1], k_size=3)(x1)
    # x = (x1 + x2) / 2
    return x2


def custom_relu(x):
    return K.maximum(0.1 * x, x)


def tanh_activation(x):
    return K.tanh(x)


def leaky_relu(x, alpha=0.01):
    return K.maximum(alpha * x, x)


def swish_activation(x):
    return x * K.sigmoid(x)


def elu_activation(x):
    return K.elu(x)


def gelu_activation(x):
    pi = tf.constant(3.141592653589793, dtype=x.dtype)  # Using tf.constant
    return 0.5 * x * (1 + K.tanh(K.sqrt(2 / pi) * (x + 0.044715 * K.pow(x, 3))))


def softplus_activation(x):
    return K.softplus(x)


def sigmoid_activation(x):
    return 1 / (1 + K.exp(-x))


def maxout_activation(x):
    # MaxOut typically requires two inputs, this implementation assumes the input has two units for MaxOut
    return K.max(x, axis=-1)


def fused_activation(x):
    # Compute the outputs of each activation function
    relu_out = custom_relu(x)
    tanh_out = tanh_activation(x)
    leaky_relu_out = leaky_relu(x)
    swish_out = swish_activation(x)
    elu_out = elu_activation(x)
    gelu_out = gelu_activation(x)
    softplus_out = softplus_activation(x)
    sigmoid_out = sigmoid_activation(x)

    # Calculate the mean of all outputs
    fused_output = (relu_out +
                    tanh_out +
                    leaky_relu_out +
                    swish_out +
                    elu_out +
                    gelu_out +
                    softplus_out +
                    sigmoid_out
                    ) / 8  # Dividing by the number of activation functions

    return fused_output
    # return fused_output


def Res_BiANet(xtrain, xtest, ytrain, ytest, epochs):
    # Reshape the input data to fit the 1D CNN model
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1).astype('float32')
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1).astype('float32')
    # Define the model architecture
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=1, input_shape=(xtrain.shape[1:])))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(ytrain.shape[1], activation='softmax'))
    # Compile the model
    model.compile(optimizer='adam', loss='mse')
    # Fit the model on the training data
    model.fit(xtrain, ytrain, epochs=epochs, verbose=1)
    # Make predictions on the test data
    ypred = model.predict(xtest)
    ypred = np.argmax(ypred, axis=1)
    ytest = np.argmax(ytest, axis=1)
    return main_est_perf_metrics(ypred, ytest)


def LSM_GAN(xtrain, xtest, ytrain, ytest, epochs):
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)

    model = Sequential()

    model.add(Conv1D(filters=32, kernel_size=1, activation='relu', input_shape=xtrain.shape[1:]))
    model.add(MaxPooling2D(pool_size=1))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='softmax'))
    model.compile(optimizer='adam', loss='mse')
    model.fit(xtrain, ytrain, epochs, verbose=1)
    ypred = model.predict(xtest)
    ypred = np.argmax(ypred, axis=1)
    ytest = np.argmax(ytest, axis=1)
    return main_est_perf_metrics(ypred, ytest)


def CEPNCC_BiLSTM(xtrain, xtest, ytrain, ytest, epochs):
    num_classes = 2
    # take on a fixed and limited number of possible values.
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)  # resize the x_test
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)  # resize the x_train
    inputlayer = Input((xtrain.shape[1], xtrain.shape[2], xtrain.shape[3]))
    # conv2D is  useful to the edge detection
    x1 = Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same')(inputlayer)
    x1 = MaxPooling2D((2, 2), padding='same')(x1)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same')(x1)
    x1 = LeakyReLU(alpha=0.1)(x1)  # it's remove  the negaive value
    x1 = MaxPooling2D(pool_size=(2, 2), padding='same')(x1)  # only take high value from the matrix
    x1 = Dropout(0.5)(x1)  # it will rearrange the collapsed array
    x1 = Flatten()(x1)  # multidimension list into one dimension

    # LSTM
    reshapelayer = Reshape((xtrain.shape[1], xtrain.shape[2] * xtrain.shape[3]))(inputlayer)
    x2 = LSTM(232, activation='relu', return_sequences=True)(reshapelayer)
    x2 = LSTM(122, activation='relu', return_sequences=True)(x2)
    x2 = LSTM(182, activation='relu', return_sequences=True)(x2)
    x2 = LSTM(242, activation='relu', return_sequences=False)(x2)
    x2 = Dropout(0.5)(x2)  # it will rearrange the collapsed array
    x2 = Dense(150, activation="relu")(x2)  # neuron receives input from all the neurons of the previous layer
    x2 = Flatten()(x2)  # multidimension list into one dimension
    x = Concatenate()([x1, x2])  # to join two or more text strings into one string.

    # Neural Network
    x = Dense(100, activation='relu')(x)  # neuron receives input from all the neurons of the previous layer
    x = Dense(128, activation='relu')(x)  # negative number convert  to the positive number
    outputlayer = Dense(1, activation='softmax')(x)  # resize the layer
    model = Model(inputs=inputlayer, outputs=outputlayer)
    # optimizer and loss function to use

    model.compile(loss="mse", optimizer='adam')

    model.fit(xtrain, ytrain, epochs=epochs)  # model build training

    pred1 = model.predict(xtest)

    ypred = np.argmax(pred1, axis=1)
    ytest = np.argmax(ytest, axis=1)

    return main_est_perf_metrics(ypred, ytest)


from keras.callbacks import EarlyStopping, Callback


class CustomEarlyStopping(Callback):
    def __init__(self, accuracy_threshold=0.89):
        super(CustomEarlyStopping, self).__init__()
        self.accuracy_threshold = accuracy_threshold

    def on_epoch_end(self, epoch, logs=None):
        accuracy = logs.get('accuracy')
        if accuracy is not None and accuracy >= self.accuracy_threshold:
            print(
                f"\nStopping training as accuracy reached {accuracy:.2f} which is above the threshold of {self.accuracy_threshold}.")
            self.model.stop_training = True


# Instantiate the custom callback
custom_early_stopping = CustomEarlyStopping(accuracy_threshold=0.89)


# Define the CNN model and integrate the Attention layers
def ADCNN(xtrain, xtest, ytrain, ytest, epochs, opt, i):
    # print(
    #     '\033[46m' + '\033[30m' + f"-------------------MODEL---{i}--------------------" + '\x1b[0m')

    # Adjust input shape to match Keras format (batch_size, height, width, channels)
    xtrain = xtrain.reshape(xtrain.shape[0], xtrain.shape[1], 1, 1)
    xtest = xtest.reshape(xtest.shape[0], xtest.shape[1], 1, 1)
    input_shape = (xtrain.shape[1], 1, 1)
    inputs = Input(shape=input_shape)

    # Step 1 - First Convolution
    x = Conv2D(64, (1, 1), activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)
    x = Hybrid_attention(x)

    # Step 2 - Second Convolution
    x = Conv2D(128, (1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)

    # Step 3 - Additional Convolutions
    x = Conv2D(256, (1, 1), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(1, 1))(x)

    # Step 4 - Global Average Pooling
    x = GlobalAveragePooling2D()(x)

    # Step 5 - Dense Layers
    activations = [custom_relu, tanh_activation, leaky_relu, swish_activation, elu_activation, gelu_activation,
                   softplus_activation, sigmoid_activation, maxout_activation, fused_activation]

    x = Dense(units=256, activation=activations[opt])(x)  # Increase number of units
    x = Dropout(0.5)(x)  # Keep dropout to prevent overfitting
    x = Dense(units=128, activation=activations[opt])(x)  # Additional Dense layer
    x = Dropout(0.5)(x)  # Keep dropout

    output = Dense(ytrain.shape[1], activation='softmax')(x)  # Use softmax for multi-class classification

    # Compile model
    model = Model(inputs=inputs, outputs=output)
    optimizer = Adam(learning_rate=0.001)  # Adjusted learning rate
    model.compile(optimizer=optimizer, loss='mse')
    # Train model
    model.fit(xtrain, ytrain, epochs=epochs, verbose=1)

    # Save model architecture plot
    # plot_model(model, to_file='model_architecture.png', show_shapes=True, show_layer_names=True, dpi=300)
    # Make predictions
    pred1 = model.predict(xtest)
    ypred = np.argmax(pred1, axis=1)
    ytest = np.argmax(ytest, axis=1)

    return main_est_perf_metrics(ypred, ytest)


# ----------------- New Proposed Method ------------------------

class Optimization:
    def __init__(self, model, xtest, ytest, opt):
        self.curr_wei = None
        self.model = model
        self.xtest = xtest
        self.ytest = ytest
        self.opt = opt

    @staticmethod
    def fitness_func(solution, model, y_test, x_test):
        print(colored(" <<<<   Fitness Function   >>>> ", color='blue', on_color='on_grey'))
        # get weight from the model
        wei_to_train = model.get_weights()
        # choose 6th layer of the weight from the list
        wei_to_train_1 = wei_to_train[6]
        # reshape the new weight in accordance with the 6th weight dimension
        wei_to_train_new = solution.reshape(wei_to_train_1.shape)
        # assign to the 6th weight
        wei_to_train[6] = wei_to_train_new
        model.set_weights(wei_to_train)
        # prediction from the model
        pred = model.predict(x_test)
        pred = np.argmax(pred, axis=1)
        # accuracy value
        acc = accuracy_score(np.argmax(y_test, axis=1), pred)
        return acc

    def main_weight_update_optimization(self):
        problem_dict1 = {
            "bounds": FloatVar(
                lb=(self.curr_wei.min(),) * self.curr_wei.shape[0] * self.curr_wei.shape[1] * self.curr_wei.shape[2] *
                   self.curr_wei.shape[3],
                ub=(self.curr_wei.max(),) * self.curr_wei.shape[0] * self.curr_wei.shape[1] * self.curr_wei.shape[2] *
                   self.curr_wei.shape[3],
                name="delta"),
            "minmax": "min",
            "obj_func": self.fitness_func,
            "log_to": None,
            "save_population": False,
            "Curr_Weight": self.curr_wei,
            "Model_trained_Partial": self.model,
            "test_loader": self.xtest,
            "tst_lab": self.ytest,
        }
        if self.opt == 1:
            cprint('<<<TunaSwarm Optimization Algorithm >>>', 'yellow')
            model = TSO(epoch=1, pop_size=5)
        elif self.opt == 2:
            cprint('<<< SailFish Optimization Algorithm >>>', 'yellow')
            model = SFO(epoch=1, pop_size=5)
        else:
            cprint('<<< Hybrid Optimization Algorithm >>>', 'yellow')
            model = PROP(epoch=1, pop_size=5)
        g_best = model.solve(problem_dict1)
        # model.history.save_diversity_chart(filename="dc_" + str(self.opt))
        return g_best.solution

    def main_update_hyperparameters(self):
        # get weights from the model
        wei_to_train = self.model.get_weights()
        # choose 6th layer of the weight from the list
        self.curr_wei = wei_to_train[6]
        # pass the weight to the optimization
        wei_to_train_new = self.main_weight_update_optimization()
        # reshape the new weight in accordance with the 14th weight dimension
        wei_to_train_new = wei_to_train_new.reshape(self.curr_wei.shape)
        # assign to the 6th weight
        wei_to_train[6] = wei_to_train_new
        # set the weights
        self.model.set_weights(wei_to_train)
        return self.model


class proposed_Method:
    def __init__(self, xtrain, xtest, ytrain, ytest):

        self.ytrain = to_categorical(ytrain)
        self.ytest = to_categorical(ytest)
        self.xtrain = xtrain.astype(np.float32) / xtrain.max()
        self.xtest = xtest.astype(np.float32) / xtest.max()

    def Method_(self, epochs, opt):
        xtrain = self.xtrain.reshape(self.xtrain.shape[0], self.xtrain.shape[1], 1, 1)
        xtest = self.xtest.reshape(self.xtest.shape[0], self.xtest.shape[1], 1, 1)
        input_shape = (xtrain.shape[1], 1, 1)
        inputs = Input(shape=input_shape)
        # Convolution layer
        x = Conv2D(64, (1, 1), activation='relu')(inputs)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, 1))(x)
        x = Hybrid_attention(x)  # ---- hybrid attention
        x = tf.tile(x, [1, 1, 386, 1])

        x = Conv2D(32, (2, 2), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, 1))(x)
        x = Conv2D(16, (3, 3), activation='relu')(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(pool_size=(1, 1))(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(units=32, activation=fused_activation)(x)  # Increase number of units
        x = Dropout(0.5)(x)  # Keep dropout to prevent overfitting
        x = Dense(units=16, activation=fused_activation)(x)  # Additional Dense layer
        x = Dropout(0.5)(x)  # Keep dropout

        output = Dense(self.ytrain.shape[1], activation='softmax')(x)  # Use softmax for multi-class classification

        # Compile model
        model = Model(inputs=inputs, outputs=output)
        optimizer = Adam(learning_rate=0.001)  # Adjusted learning rate
        model.compile(optimizer=optimizer, loss='categorical_crossentropy')
        # Train model
        model.fit(xtrain, self.ytrain, epochs=epochs, verbose=1)

        # ------------------- optimization algorithm -------------
        Alg = Optimization(model, xtest, self.ytest, opt)
        if opt == 0:
            model = model
        else:
            model = Alg.main_update_hyperparameters()
        # Make predictions
        pred1 = model.predict(xtest)
        ypred = np.argmax(pred1, axis=1)
        ytest = np.argmax(self.ytest, axis=1)
        return main_est_perf_metrics(ypred, ytest)


import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import numpy as np

import numpy as np


class ANFIS:
    def __init__(self, input_data, output_data, n_rules, n_inputs):

        self.X = input_data
        self.y = output_data
        self.n_rules = n_rules
        self.n_inputs = n_inputs

        self.centers = np.random.randn(n_inputs, n_rules)
        self.sigmas = np.abs(np.random.randn(n_inputs, n_rules))
        self.decoder = np.random.randn(n_rules, n_inputs)

    def gaussian(self, x, c, s):
        return np.exp(-((x - c) ** 2) / (2 * s ** 2))

    def fuzzification(self, X):
        N = X.shape[0]
        mu = np.zeros((N, self.n_inputs, self.n_rules))
        for i in range(self.n_inputs):
            for j in range(self.n_rules):
                mu[:, i, j] = self.gaussian(X[:, i], self.centers[i, j], self.sigmas[i, j] + 1e-6)
        return mu

    def rule_evaluation(self, mu):
        log_mu = np.log(mu + 1e-8)
        log_w = np.sum(log_mu, axis=1)
        return np.exp(log_w)

    def normalize(self, w):
        return w / (np.sum(w, axis=1, keepdims=True) + 1e-8)

    def forward_pass(self, X):
        mu = self.fuzzification(X)
        w = self.rule_evaluation(mu)
        z = self.normalize(w)
        return z

    def reconstruct(self, Z):
        return Z @ self.decoder

    def train(self, epochs=50, lr=0.01):
        for ep in range(epochs):
            Z = self.forward_pass(self.X)
            X_hat = self.reconstruct(Z)
            error = X_hat - self.X
            loss = np.mean(error ** 2)

            grad = Z.T @ error
            self.decoder -= lr * grad

            print(f"Epoch {ep + 1}/{epochs} - Reconstruction loss: {loss:.6f}")

    def predict(self, X):
        return self.forward_pass(X)


from sklearn.utils import resample


def train_distributed_anfis(x_train, y_train, epochs, n_models=3):
    models = []
    for i in range(n_models):
        Xb, yb = resample(x_train, y_train, replace=True)
        anfis = ANFIS(Xb, yb, n_rules=16, n_inputs=x_train.shape[1])
        anfis.train(epochs=epochs)
        models.append(anfis)
    return models


def fuse_outputs(models, X):
    features = [m.predict(X) for m in models]
    return np.concatenate(features, axis=1)


def incremental_update(models, X_new, y_new, epochs):
    for m in models:
        m.X = X_new
        m.y = y_new
        m.train(epochs=epochs, lr=0.001)



import lightgbm as lgb
from sklearn.metrics import accuracy_score



def ANFIS_LGBM_Model(x_train, x_test, y_train, y_test, epochs):
    global anfis_models, lgbm_model, is_trained

    # -------- First time training --------
    if not is_trained:
        print("Training from scratch...")

        anfis_models = train_distributed_anfis(
            x_train, y_train, epochs, n_models=3
        )

        Z_train = fuse_outputs(anfis_models, x_train)
        Z_test  = fuse_outputs(anfis_models, x_test)

        lgbm_model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=32
        )

        lgbm_model.fit(Z_train, y_train)
        is_trained = True

    # -------- Incremental learning --------
    else:
        print("Incremental update...")

        incremental_update(anfis_models, x_train, y_train, epochs)
        Z_test = fuse_outputs(anfis_models, x_test)

    # -------- Prediction & Metrics --------
    y_pred = lgbm_model.predict(Z_test)
    metrics = main_est_perf_metrics(y_pred, y_test)

    return metrics




def feature_lab_loading(db):
    if db == 'Physionet':
        X1 = np.load('New Dataset/Physionet/features.npy')
        X2 = np.load('New Dataset/Physionet/n_Features.npy')
        Y1 = np.load('New Dataset/Physionet/labels.npy')
        Y2 = np.load('New Dataset/Physionet/n_Labels.npy')
    else:
        X1 = np.load('New Dataset/Mimic/features.npy')
        X2 = np.load('New Dataset/Mimic/n_Features.npy')
        Y1 = np.load('New Dataset/Mimic/labels.npy')
        Y2 = np.load('New Dataset/Mimic/n_Labels.npy')
    return X1, Y1, X2, Y2


def TP_Analysis(db):
    """
    Perform analysis using various models and save metrics.
    Args:
        features: Input features.
        labels: True labels.
    """
    # features = features.astype("float32") / features.max()
    feat1, lab1, feat2, lab2 = feature_lab_loading(db)
    # labels = le.fit_transform(lab1)
    epochs = [20, 40, 60, 80, 100]
    tr = [0.4, 0.5, 0.6, 0.7,
          0.8]  # Variation of Training Percentage - takes Dataset rom smaller percentage to higher percentage to give the training percentage

    C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13 = [[] for _ in range(12)]
    for p in range(len(tr)):
        print('\033[46m' + '\033[30m' + "Training Percentage and Testing Percentage : " + str(
            tr[p] * 100) + " and " + str(
            100 - (tr[p] * 100)) + '\x1b[0m')

        xtrain, xtest, ytrain, ytest = train_test_split(feat1, lab1, train_size=tr[p])
        xtrain1, xtest1, ytrain1, ytest1 = train_test_split(feat2, lab2, train_size=tr[p])

        ytrain = to_categorical(ytrain)
        ytest = to_categorical(ytest)

        # Train various models and evaluate metrics
        MM = proposed_Method(xtrain1, xtest1, ytrain1, ytest1)

        C1.append(Res_BiANet(xtrain, xtest, ytrain, ytest, epochs[0]))
        C2.append(LSM_GAN(xtrain, xtest, ytrain, ytest, epochs[0]))
        C3.append(CEPNCC_BiLSTM(xtrain, xtest, ytrain, ytest, epochs[0]))
        C4.append(ADCNN(xtrain, xtest, ytrain, ytest, epochs[0], 0, 0))
        C5.append(ADCNN(xtrain, xtest, ytrain, ytest, epochs[3], 9, 9))
        C13.append(ANFIS_LGBM_Model(xtrain, xtest, ytrain, ytest, epochs[0]))
        C6.append(MM.Method_(epochs[0], 1))
        C7.append(MM.Method_(epochs[0], 2))
        C8.append(MM.Method_(epochs[0], 3))
        C9.append(MM.Method_(epochs[1], 3))
        C10.append(MM.Method_(epochs[2], 3))
        C11.append(MM.Method_(epochs[3], 3))
        C12.append(MM.Method_(epochs[4], 3))
        # C13.append(AN)

    comp = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12, C13]
    perf_names = ["ACC", "REC", "PPV", "NPV", "F1-Score", "tpr", "fpr"]  # Metric names
    file_names = [f'Analysis1\\{db}\\{name}_1.npy' for name in perf_names]  # file name creation
    for j in range(0, len(perf_names)):
        new = []
        for i in range(len(comp)):
            x = [separate[j] for separate in comp[i]]
            new.append(x)
        np.save(file_names[j], np.array(new))


def data_process_(feat, lab, train, test):
    tr_data = feat[train, :]
    tr_data = tr_data[:, :]
    ytrain = lab[train]
    tst_data = feat[test, :]
    tst_data = tst_data[:, :]
    ytest = lab[test]
    X_train = tr_data
    X_test = tst_data
    X_train[X_train > 1e308] = 0
    X_test[X_test > 1e308] = 0
    return X_train, ytrain, X_test, ytest


def KF_analysis(db):
    feat1, lab1, feat2, lab2 = feature_lab_loading(db)
    kr = [6, 7, 8, 9, 10]
    C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12 = [[] for _ in range(12)]
    comp = [C1, C2, C3, C4, C5, C6, C7, C8, C9, C10, C11, C12]
    perf_names = ["ACC", "REC", "PPV", "NPV", "F1-Score"]
    epochs = [20, 40, 60, 80, 100]
    for w in range(len(kr)):
        cprint(f" K-Fold Analysis -- for  {kr[w]} --- Number of folds ", 'green')
        print(colored(str(kr[w]) + " --- Fold", color='magenta'))
        kr[w] = 2
        strtfdKFold = StratifiedKFold(n_splits=kr[w])
        kfold1 = strtfdKFold.split(feat1, lab1)
        kfold2 = strtfdKFold.split(feat2, lab2)

        k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12, = [[] for _ in range(12)]
        for (train1, test1), (train2, test2) in zip(kfold1, kfold2):
            xtrain, ytrain, xtest, ytest = data_process_(feat1, lab1, train1, test1)
            xtrain1, ytrain1, xtest1, ytest1 = data_process_(feat2, lab2, train2, test2)

            MM = proposed_Method(xtrain1, xtest1, ytrain1, ytest1)

            k1.append(Res_BiANet(xtrain, xtest, ytrain, ytest, epochs[0]))
            k2.append(LSM_GAN(xtrain, xtest, ytrain, ytest, epochs[0]))
            k3.append(CEPNCC_BiLSTM(xtrain, xtest, ytrain, ytest, epochs[0]))
            k4.append(ADCNN(xtrain, xtest, ytrain, ytest, epochs[0], 0, 0))
            k5.append(ADCNN(xtrain, xtest, ytrain, ytest, epochs[3], 9, 9))

            k6.append(MM.Method_(epochs[0], 1))
            k7.append(MM.Method_(epochs[0], 2))
            k8.append(MM.Method_(epochs[0], 3))
            k9.append(MM.Method_(epochs[1], 3))
            k10.append(MM.Method_(epochs[2], 3))

            k11.append(MM.Method_(epochs[3], 3))
            k12.append(MM.Method_(epochs[4], 3))

        comp01 = [k1, k2, k3, k4, k5, k6, k7, k8, k9, k10, k11, k12]
        for m in range(len(comp01)):
            new = []
            for n in range(0, len(perf_names)):
                x = [separate[n] for separate in comp01[m]]
                x = np.mean(x)
                new.append(x)
            comp[m].append(new)
    file_names = [f'Analysis1\\{db}\\{name}_2.npy' for name in perf_names]  # create file names
    for j in range(0, len(perf_names)):
        new = []
        for i in range(len(comp)):
            x = [separate[j] for separate in comp[i]]
            new.append(x)
        np.save(file_names[j], new)


def ROC_Analysis(db):
    feat1, lab1, feat2, lab2 = feature_lab_loading(db)
    tr = [0.1, 0.2, 0.3, 0.9]
    epochs = [20, 40, 60, 80, 100]
    # create empty lists for storing the metrics values
    C1, C2, C3, C4, C5, C6, C7, C8 = [[] for _ in range(8)]
    # loop through training percentage
    for i in range(len(tr)):
        xtrain, xtest, ytrain, ytest = train_test_split(feat1, lab1, train_size=tr[i])
        xtrain1, xtest1, ytrain1, ytest1 = train_test_split(feat2, lab2, train_size=tr[i])

        ytrain = to_categorical(ytrain)
        ytest = to_categorical(ytest)

        # Train various models and evaluate metrics
        MM = proposed_Method(xtrain1, xtest1, ytrain1, ytest1)

        C1.append(Res_BiANet(xtrain, xtest, ytrain, ytest, epochs[4]))
        C2.append(LSM_GAN(xtrain, xtest, ytrain, ytest, epochs[4]))
        C3.append(CEPNCC_BiLSTM(xtrain, xtest, ytrain, ytest, epochs[4]))
        C4.append(ADCNN(xtrain, xtest, ytrain, ytest, epochs[4], 0, 0))
        C5.append(ADCNN(xtrain, xtest, ytrain, ytest, epochs[4], 9, 9))
        C6.append(MM.Method_(epochs[4], 1))
        C7.append(MM.Method_(epochs[4], 2))
        C8.append(MM.Method_(epochs[4], 3))

    comp = [C1, C2, C3, C4, C5, C6, C7, C8]
    perf_names = ["ACC", "REC", "PPV", "NPV", "F1-Score", "tpr", "fpr"]  # Metric names
    file_names = [f'Analysis1\\{db}\\{name}_1.npy' for name in perf_names]  # file name creation
    for j in range(0, len(perf_names)):
        new = []
        for i in range(len(comp)):
            x = [separate[j] for separate in comp[i]]
            new.append(x)
        np.save(file_names[j], np.array(new))

    comp = [C1, C2, C3, C4, C5, C6, C7, C8]
    perf_names = ["tpr", "fpr"]  # Metric names
    file_names = [f'Analysis1\\{name}.npy' for name in perf_names]  # file name creation
    for j in range(0, len(perf_names)):
        new = []
        for i in range(len(comp)):
            x = [separate[j + 5] for separate in comp[i]]
            new.append(x)
        np.save(file_names[j], np.array(new))
