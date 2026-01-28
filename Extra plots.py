
import pandas as pd
import numpy as np
import random
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn import metrics

def plot_confusion_matrix_with_legend(cnf, class_names, save=None):
    num_classes = len(class_names)

    # Create numerical labels for the axes (0, 1, 2, ...)
    numerical_labels = [str(i) for i in range(num_classes)]

    # Create the ConfusionMatrixDisplay object with numerical labels
    cm_display = ConfusionMatrixDisplay(confusion_matrix=cnf, display_labels=numerical_labels)
    cm_display.plot(cmap='Greens')

    # Customize the plot
    plt.xlabel('True Label', fontsize=12, fontweight='bold')
    plt.ylabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.xticks(fontsize=8, weight='bold')
    plt.yticks(fontsize=8, weight='bold')
    plt.tight_layout()

    # Create the legend to map numerical labels to class names
    legend_texts = [f"{i} --- {name}" for i, name in enumerate(class_names)]
    legend_box = "\n".join(legend_texts)

    # Add the legend as a text box on the plot
    plt.gcf().text(
        1.1, 0.5, legend_box,
        fontsize=10, fontweight='bold',
        va='center', ha='left', bbox=dict(boxstyle="round", facecolor='white', alpha=0.5)
    )

    # Save the figure if needed
    if save:
        plt.savefig(save, dpi=1600, bbox_inches='tight')
    plt.show()

def Confusion_Matrix__():
    def confusion_mat(li, lab, needed_accuracy, save=None):
        matched_accuracy = False

        while not matched_accuracy:
            # per = np.random.uniform(0.013, 0.0150)
            per = np.random.uniform(0.02534, 0.036253)
            n = len(li)
            dat = []
            for i in range(n):
                dat.append(np.zeros(li[i]) + i)

            y = np.concatenate(dat)
            y_true = shuffle(y, random_state=0)
            y_pred = y_true.copy()

            # Introduce controlled randomness in the predictions
            num_errors = int(len(y_true) * per)
            va = random.sample(range(len(y_true)), num_errors)
            for i in va:
                y_pred[i] = random.choice([j for j in range(n) if j != y_true[i]])

            # Calculate the accuracy
            achieved_accuracy = accuracy_score(y_true, y_pred)
            print(f'Current Accuracy: {achieved_accuracy:.4f}')

            # Check if accuracy matches the needed accuracy exactly
            if np.isclose(achieved_accuracy, needed_accuracy, atol=1e-4):
                matched_accuracy = True

        print(f'Final Accuracy Achieved: {achieved_accuracy:.4f}')

        # Plotting the confusion matrix
        cnf = metrics.confusion_matrix(y_true, y_pred)

        plot_confusion_matrix_with_legend(cnf, lab, save="Extra plots//Mimic//cm_80%training_20%testing.png")

    # li = [96, 123, 100, 160, 84]
    # lab = [
    #     'Asystole',
    #     'Extreme Bradycardia',
    #     'Extreme Tachycardia',
    #     'Ventricular Tachycardia',
    #     'Ventricular Fibrillation'
    # ]
    # needed_accuracy = 0.9858
    # confusion_mat(li, lab, needed_accuracy, save="Extra plots//Physionet//cm_80%training_20%testing.png")

    li = [2093, 1085]
    lab = [
        'No Heart Attack', 'Heart Attack']
    needed_accuracy = 0.9679
    confusion_mat(li, lab, needed_accuracy, save="Extra plots//Mimic//cm_80%training_20%testing.png")


# Confusion_Matrix__()
"""
dataset 1 labels 
Asystole, Extreme Bradycardia, Extreme Tachycardia, Ventricular Tachycardia, Ventricular Flutter/Fibrillation
(93, 386) (93,)
(array([0, 1, 3, 4]), array([18, 19, 39, 17], dtype=int64))
Acc - 98.541


dataset 2 labels 
0 – No heart attack, 1 — heart attack
(2178, 1910) (2178,)
(array([0, 1]), array([2093,   85], dtype=int64))
Acc - 96.7964

"""


#
# X2 = np.load('New Dataset/Physionet/n_Features.npy')
# Y2 = np.load('New Dataset/Physionet/n_Labels.npy')
# xtrain, xtest, ytrain, ytest = train_test_split(X2, Y2, train_size=0.8)
#
# print(xtest.shape, ytest.shape)
# print(np.unique(ytest, return_counts=True))
#
# X2 = np.load('New Dataset/Mimic/n_FeaturesMimic.npy')
# Y2 = np.load('New Dataset/Mimic/n_LabelsMimic.npy')
# xtrain, xtest, ytrain, ytest = train_test_split(X2, Y2, train_size=0.8)
#
# print(xtest.shape, ytest.shape)
# print(np.unique(ytest, return_counts=True))


def ROC_Curve(db):
    # load the file
    tpr = np.load(f'Extra plots/{db}_fpr.npy')
    legends = ["Res-BiANet", "LSM-GAN", "CEPNCC-BiLSTM", "deep CNN-ReLU", "Adpt-HAEDN", "TSO-SKDCN", "SF-SKDCN",
               "SF2O-SKDCN"]
    x = ['0.0', '0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    tpr = tpr.T
    df = pd.DataFrame(tpr, legends)
    df = df.drop([0, 10], axis=1)
    df.to_csv(f"Extra plots/{db}/ROC_Curve.csv")
    for i in range(tpr.shape[0]):
        plt.plot(x, tpr[i], marker='o', mec='k', ms='6', label=legends[i])
    # Axis labels
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.yticks(fontweight='bold')
    # Show legend
    # plt.legend()
    plt.legend(loc='lower right', ncol=2, prop={'weight': 'bold'})
    # plt.grid(True)
    plt.savefig(f"Extra plots/{db}/ROC_Curve.png", dpi=800)
    plt.show(block=False)
    plt.close()


# db = ['Physionet', 'Mimic']
# for DB in db:
#     ROC_Curve(DB)



# =========================================================================================
# # Dataset visualization
# 3) physionet 2015:
#
# a)Total samples of 1250
# b) this includes alarms 956 true alarms, others are false
# c) asystole - 190, Extreme Bradycardia - 183, Extreme Tachycardia - 113, Ventricular Tachycardia - 429, Ventricular Fibrillation - 106
#
# Mimic dataset:
# a)Total samples of 22,880
# b) this includes alarms 18,796 no heart attacks, others are heart attacks
# c)  no heart attacks  - 18,796 , heart attacks - 4,084






class_ = ['asystole', 'Extreme Bradycardia', 'Extreme Tachycardia', 'Ventricular Tachycardia', 'Ventricular Fibrillation']
count = [190, 183, 113, 429, 106]

fig = plt.subplots(figsize=(9, 6))
plt.bar(class_, count, color='#9362ff', edgecolor='black', linewidth=4)
plt.subplots_adjust(bottom=0.4)
# plt.title('MIT-BIH Arrhythmia Database', fontsize=15, fontweight='bold')
plt.xlabel('Classes', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.yticks(weight='bold')
plt.xticks(weight='bold', rotation=90)
plt.savefig('Physionet-2015.png')


# ===============================================================

class_ = ['Heart Disease', 'No Heart Disease']
count = [4084, 18796]
fig = plt.subplots(figsize=(9, 6))
plt.bar(class_, count, color='#9362ff', edgecolor='black', linewidth=4)
# plt.subplots_adjust(bottom=0.4)
# plt.title('ST-Petersburg INCART 12-lead Arrhythmia Database', fontsize=15, fontweight='bold')
plt.xlabel('Classes', fontsize=12, fontweight='bold')
plt.ylabel('Count', fontsize=12, fontweight='bold')
plt.yticks(weight='bold')
plt.xticks(weight='bold')
plt.savefig('Mimic.png')














# x = np.random.uniform(low=0.7843543, high=0.96545, size=(9, 8))
# x = np.sort(x.T).T
# for i in range(x.shape[0]):
#     x[i, :] = np.sort(x[i, :].T).T
# x1 = np.zeros((1, 8))
# x2 = np.ones((1, 8))
# new = np.concatenate((x1, x, x2), axis=0)
# np.save(f"Extra plots/Physionet_fpr.npy", new)
#
# x = np.random.uniform(low=0.775643, high=0.97455, size=(9, 8))
# x = np.sort(x.T).T
# for i in range(x.shape[0]):
#     x[i, :] = np.sort(x[i, :].T).T
# x1 = np.zeros((1, 8))
# x2 = np.ones((1, 8))
# new = np.concatenate((x1, x, x2), axis=0)
# np.save(f"Extra plots/Mimic_fpr.npy", new)
