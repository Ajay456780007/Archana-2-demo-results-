import random

import pandas as pd
import numpy as np
from termcolor import cprint
import matplotlib.pyplot as plt

import matplotlib

from warnings import filterwarnings

filterwarnings(action='ignore', category=UserWarning)
matplotlib.use('Agg')


def load_parameter(db):
    """
    param:
    output ---- list of arrays
    """
    if db == "DS1":
        db = "Physionet"
    else:
        db = "Mimic"

    A = np.load(f"Analysis/{db}/ACC_1.npy", allow_pickle=True)
    B = np.load(f"Analysis/{db}/PPV_1.npy", allow_pickle=True)
    C = np.load(f"Analysis/{db}/NPV_1.npy", allow_pickle=True)
    D = np.load(f"Analysis/{db}/REC_1.npy", allow_pickle=True)
    E = np.load(f"Analysis/{db}/ACC_2.npy", allow_pickle=True)
    F = np.load(f"Analysis/{db}/PPV_2.npy", allow_pickle=True)
    G = np.load(f"Analysis/{db}/NPV_2.npy", allow_pickle=True)
    H = np.load(f"Analysis/{db}/REC_2.npy", allow_pickle=True)

    I = np.load(f"Analysis/{db}/F1-Score_1.npy", allow_pickle=True)
    J = np.load(f"Analysis/{db}/F1-Score_2.npy", allow_pickle=True)

    A_ = A[[0, 1, 2, 3, 4, 5, 6, 11], :]
    B_ = B[[0, 1, 2, 3, 4, 5, 6, 11], :]
    C_ = C[[0, 1, 2, 3, 4, 5, 6, 11], :]
    D_ = D[[0, 1, 2, 3, 4, 5, 6, 11], :]
    E_ = E[[0, 1, 2, 3, 4, 5, 6, 11], :]
    F_ = F[[0, 1, 2, 3, 4, 5, 6, 11], :]
    G_ = G[[0, 1, 2, 3, 4, 5, 6, 11], :]
    H_ = H[[0, 1, 2, 3, 4, 5, 6, 11], :]
    I_ = I[[0, 1, 2, 3, 4, 5, 6, 11], :]
    J_ = J[[0, 1, 2, 3, 4, 5, 6, 11], :]

    A1_ = A[[7, 8, 9, 10, 11], :]
    B1_ = B[[7, 8, 9, 10, 11], :]
    C1_ = C[[7, 8, 9, 10, 11], :]
    D1_ = D[[7, 8, 9, 10, 11], :]
    E1_ = E[[7, 8, 9, 10, 11], :]
    F1_ = F[[7, 8, 9, 10, 11], :]
    G1_ = G[[7, 8, 9, 10, 11], :]
    H1_ = H[[7, 8, 9, 10, 11], :]
    I1_ = I[[7, 8, 9, 10, 11], :]
    J1_ = J[[7, 8, 9, 10, 11], :]
    return [A_, B_, C_, D_, E_, F_, G_, H_, I_, J_, A1_, B1_, C1_, D1_, E1_, F1_, G1_, H1_, I1_, J1_]



def COMP_PLOT(perf, legends, x, y, type, db):
    perf = perf * 100
    if type == "tp":
        filename1 = f'Results\\{db}\\TP\\Comp_' + y + '.csv'
        filename2 = f'Results\\{db}\\TP\\Comp_' + y + '.png'
        filename3 = f'Results\\{db}\\TP\\Line\\Comp_' + y + '.png'
        h = ['40', '50', '60', '70', '80']
    else:
        filename1 = f'Results\\{db}\\KF\\Comp_' + y + '.csv'
        filename2 = f'Results\\{db}\\kF\\Comp_' + y + '.png'
        filename3 = f'Results\\{db}\\KF\\Line\\Comp_' + y + '.png'
        h = ['6', '7', '8', '9', '10']

    perf = perf.T
    data = {i: perf[i] for i in range(perf.shape[0])}
    df = pd.DataFrame(data)
    df.index = legends
    df.columns = h
    df.to_csv(filename1)
    cprint(f'{y} tabular data is stored to the file path {filename1}', color='green')

    perf = perf.T
    index = np.arange(5)
    bar_width = 0.10
    opacity = 0.8
    # plt.figure(val)
    plt.figure(figsize=(7, 6))

    #  ------------ Create a bar plot ------------

    colors = ['#008c9e', '#00508c', '#3f0086', '#8700aa', '#8c2400', '#c56000', '#565656', '#18754a',
              '#ff7c00', '#8cd721', '#360070']
    for i in range(0, perf.shape[0]):
        if i == 0:
            plt.bar(index, perf[i, :], bar_width, alpha=opacity, edgecolor="black", color=colors[i], linewidth=1.5,
                    label=legends[i])
        else:
            plt.bar(index + i * bar_width, perf[i, :], bar_width, alpha=opacity, edgecolor="black", linewidth=1.5,
                    color=colors[i],
                    label=legends[i])
    plt.xlabel(x, fontsize=18, fontweight='bold')
    plt.ylabel(y, fontsize=18, fontweight='bold')
    plt.xticks(index + 0.1, tuple(h), weight='bold')
    plt.yticks(weight='bold')
    plt.legend(loc='lower right', ncol=2, prop={'weight': 'bold', "size": 12}, framealpha=0.5)
    plt.savefig(filename2, dpi=800)
    cprint(f'{y} Graph is stored to the file path {filename2}', color='magenta')
    plt.pause(1)
    plt.close()

    # ------------- Line Plot --------------
    plt.figure(figsize=(7, 6))
    for i in range(perf.shape[0]):
        plt.plot(perf[i, :], marker='o', ms='10', mec='k', linewidth=2, linestyle='-', label=legends[i],
                 color=colors[i])
    plt.xlabel(x, fontsize=18, fontweight='bold')
    plt.ylabel(y, fontsize=18, fontweight='bold')
    plt.xticks(index + 0.1, tuple(h), weight='bold')
    plt.yticks(weight='bold')
    plt.legend(loc='lower right', prop={'weight': 'bold', "size": 12}, framealpha=0.5)
    plt.savefig(filename3, dpi=800)
    cprint(f'{y} Graph is stored to the file path {filename3}', color='magenta')
    plt.pause(1)
    plt.close()


def PERF_PLOT(perf, legends, x, y, type, db):
    perf = perf * 100
    if type == "tp":
        filename1 = f'Results\\{db}\\TP\\Perf_' + y + '.csv'
        filename2 = f'Results\\{db}\\TP\\Perf_' + y + '.png'
        filename3 = f'Results\\{db}\\TP\\Bar\\Perf_' + y + '.png'
        h = ['40', '50', '60', '70', '80']
    else:
        filename1 = f'Results\\{db}\\KF\\Perf_' + y + '.csv'
        filename2 = f'Results\\{db}\\KF\\Perf_' + y + '.png'
        filename3 = f'Results\\{db}\\KF\\Bar\\Perf_' + y + '.png'
        h = ['6', '7', '8', '9', '10']

    perf = perf.T
    data = {i: perf[i] for i in range(perf.shape[0])}
    df = pd.DataFrame(data)
    df.index = legends
    df.columns = h
    df.to_csv(filename1)
    cprint(f'{y} tabular data is stored to the file path {filename1}', color='green')

    perf = perf.T
    index = np.arange(5)  # -- n_groups = 5
    colors = ['#008c9e', '#00508c', '#3f0086', '#8700aa', '#8c2400', '#c56000', '#565656', '#18754a',
              '#ff7c00', '#8cd721', '#360070']
    # --------- Create a line chart -------------
    plt.figure(figsize=(7, 6))
    for i in range(perf.shape[0]):
        plt.plot(perf[i, :], marker='H', ms='12', mec='k', linewidth=3, linestyle='-', label=legends[i],
                 color=colors[i])
    plt.xlabel(x, fontsize=18, fontweight='bold')
    plt.ylabel(y, fontsize=18, fontweight='bold')
    plt.xticks(index + 0.1, tuple(h), weight='bold')
    plt.yticks(weight='bold')
    plt.legend(loc='lower right', prop={'weight': 'bold', "size": 12}, framealpha=0.5)
    plt.savefig(filename2, dpi=800)
    cprint(f'{y} Graph is stored to the file path {filename2}', color='magenta')
    plt.pause(1)
    plt.close()

    # ----------- Create a Bar plot --------------
    bar_width = 0.14
    opacity = 0.8
    plt.figure(figsize=(7, 6))
    for i in range(0, perf.shape[0]):
        if i == 0:
            plt.bar(index, perf[i, :], bar_width, alpha=opacity, edgecolor="black", color=colors[i], linewidth=1.5,
                    label=legends[i])
        else:
            plt.bar(index + i * bar_width, perf[i, :], bar_width, alpha=opacity, edgecolor="black", linewidth=1.5,
                    color=colors[i],
                    label=legends[i])
    plt.xlabel(x, fontsize=18, fontweight='bold')
    plt.ylabel(y, fontsize=18, fontweight='bold')
    plt.xticks(index + 0.1, tuple(h), weight='bold')
    plt.yticks(weight='bold')
    plt.legend(loc='lower right', ncol=1, prop={'weight': 'bold', "size": 12}, framealpha=0.5)
    plt.savefig(filename3, dpi=800)
    cprint(f'{y} Graph is stored to the file path {filename3}', color='magenta')
    plt.pause(1)
    plt.close()


def PLOT_(ds):
    [A_, B_, C_, D_, E_, F_, G_, H_, I_, J_, A1_, B1_, C1_, D1_, E1_, F1_, G1_, H1_, I1_, J1_] = load_parameter(ds)

    # name -- Proposed Sail Fish Foraging Optimization-based Selective Kernel attention-enabled Deep CNN
    legends = ["Res-BiANet", "LSM-GAN", "CEPNCC-BiLSTM", "deep CNN-ReLU", "Adpt-HAEDN", "TSO-SKDCN", "SF-SKDCN",
               "SF2O-SKDCN"]

    legends1 = ["SF2O-SKDCN at epochs=20", "SF2O-SKDCN at epochs=40", "SF2O-SKDCN at epochs=60",
                "SF2O-SKDCN at epochs=80", "SF2O-SKDCN at epochs=100"]

    # Training Percentage ----- TP
    # comparative and performance plot

    xlab = "Training(%)"
    COMP_PLOT(A_, legends, xlab, "Accuracy(%)", "tp", ds)
    COMP_PLOT(B_, legends, xlab, "PPV(%)", "tp", ds)
    COMP_PLOT(C_, legends, xlab, "NPV(%)", "tp", ds)
    COMP_PLOT(D_, legends, xlab, "Recall(%)", "tp", ds)
    COMP_PLOT(I_, legends, xlab, "F1-Score(%)", "tp", ds)

    PERF_PLOT(A1_, legends1, xlab, "Accuracy(%)", "tp", ds)
    PERF_PLOT(B1_, legends1, xlab, "PPV(%)", "tp", ds)
    PERF_PLOT(C1_, legends1, xlab, "NPV(%)", "tp", ds)
    PERF_PLOT(D1_, legends1, xlab, "Recall(%)", "tp", ds)
    PERF_PLOT(I1_, legends1, xlab, "F1-Score(%)", "tp", ds)

    # #  K-Fold --- cross fold validation
    # # comparative and performance plot
    xlab = "K-Fold"
    COMP_PLOT(E_, legends, xlab, "Accuracy(%)", "kf", ds)
    COMP_PLOT(F_, legends, xlab, "PPV(%)", "kf", ds)
    COMP_PLOT(G_, legends, xlab, "NPV(%)", "kf", ds)
    COMP_PLOT(H_, legends, xlab, "Recall(%)", "kf", ds)
    COMP_PLOT(J_, legends, xlab, "F1-Score(%)", "kf", ds)

    PERF_PLOT(E1_, legends1, xlab, "Accuracy(%)", "kf", ds)
    PERF_PLOT(F1_, legends1, xlab, "PPV(%)", "kf", ds)
    PERF_PLOT(G1_, legends1, xlab, "NPV(%)", "kf", ds)
    PERF_PLOT(H1_, legends1, xlab, "Recall(%)", "kf", ds)
    PERF_PLOT(J1_, legends1, xlab, "F1-Score(%)", "kf", ds)


def Saved_to_csv(db):
    Analysis = ['TP', 'KF']
    new_df = pd.DataFrame()
    for i in range(len(Analysis)):
        x1 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Comp_Accuracy(%).csv', encoding='ISO-8859-1')
        x2 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Comp_PPV(%).csv', encoding='ISO-8859-1')
        x3 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Comp_NPV(%).csv', encoding='ISO-8859-1')
        x4 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Comp_Recall(%).csv', encoding='ISO-8859-1')
        temp_df = pd.concat([x1, x2, x3, x4], axis=1, ignore_index=True)
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=True)

        x1 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Perf_Accuracy(%).csv', encoding='ISO-8859-1')
        x2 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Perf_PPV(%).csv', encoding='ISO-8859-1')
        x3 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Perf_NPV(%).csv', encoding='ISO-8859-1')
        x4 = pd.read_csv(f'Results\\{db}\\{Analysis[i]}\\Perf_Recall(%).csv', encoding='ISO-8859-1')
        temp_df = pd.concat([x1, x2, x3, x4], axis=1, ignore_index=True)
        new_df = pd.concat([new_df, temp_df], axis=0, ignore_index=True)

    # Save the combined DataFrame to a CSV file
    new_df.to_csv(f'Comparative_Combined{db}.csv', index=False)


def convergence(db):
    mm = np.load(f'NPY/convergence_{db}.npy', allow_pickle=True)
    legends = ["TSO-SKDCN", "SF-SKDCN", "SF2O-SKDCN"]
    df = pd.DataFrame(mm)
    df.index = legends
    df.to_csv(f'Results/{db}/Convergence.csv')
    cprint(f' tabular data is stored to the Results folder ', color='green')

    # Define colors for the lines
    colors = ['#008c9e', '#00508c', '#3f0086', '#8700aa', '#8c2400', '#c56000', '#565656']
    max_idx = np.argmax(np.max(mm, axis=1))
    sorted_indices = np.argsort(np.max(mm, axis=1))[::-1]
    for i in sorted_indices:
        plt.plot(mm[i], label=legends[i], linestyle='-', color=colors[i], markersize=8)
    plt.yticks(fontweight='bold')
    plt.xticks(fontweight='bold')
    plt.xlabel('Epochs', fontsize=18, fontweight='bold')
    plt.ylabel('Loss', fontsize=18, fontweight='bold')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.legend(legends, loc='upper right', ncol=2, prop={'weight': 'bold'})
    # Save the plot as an image
    plt.savefig(f'Results/{db}/Convergence.png', dpi=800)
    # Show the plot
    plt.show()
    # Close the plot
    plt.close()


def Statistical_Feature(d):
    all = []
    for i in range(d.shape[0]):
        xx = d[i]
        mean = np.mean(xx)
        median = np.median(xx)
        std_dev = np.std(xx)
        variance = np.var(xx)
        all.append(np.hstack((mean, median, std_dev, variance)))
    return np.array(all)


def load_stat(db):
    def csv_process(x):
        x = np.array(x)
        x = x[:, 1:]
        return x

    Analysis = ['TP', 'KF']
    for i in Analysis:
        A = pd.read_csv(f"Results/{db}/{i}/Comp_Accuracy(%).csv")
        B = pd.read_csv(f"Results/{db}/{i}/Comp_PPV(%).csv")
        C = pd.read_csv(f"Results/{db}/{i}/Comp_NPV(%).csv")
        D = pd.read_csv(f"Results/{db}/{i}/Comp_Recall(%).csv")

        A, B, C, D = csv_process(A), csv_process(B), csv_process(C), csv_process(D)

        index = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        met = [A, B, C, D]
        for j in range(len(met)):
            x1 = Statistical_Feature(met[j])
            df = pd.DataFrame(x1)
            df.index = ["Res-BiANet", "LSM-GAN", "CEPNCC-BiLSTM", "deep CNN-ReLU", "Adpt-HAEDN", "TSO-SKDCN",
                        "SF-SKDCN", "SF2O-SKDCN"]
            df.columns = ['Mean', 'Medium', 'Standard Deviation', 'Variance']
            df.to_csv(f'Results/{db}/Statistical/{i}_statistical_{index[j]}.csv')


def Main_plot():
    db = ['DS1', 'DS2']
    for DB in db:
        PLOT_(DB)
        convergence(DB)
        load_stat(DB)

