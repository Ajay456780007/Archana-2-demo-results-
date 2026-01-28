# --- Import  Necessary Python Modules ---
from warnings import filterwarnings
# import PySimpleGUI as sg
from Sub_Functions.open_popup import open_popup
from Analysis import TP_Analysis, KF_analysis
from PLOT import Main_plot, PLOT_
from Sub_Functions.Plot import ALL_GRAPH_PLOT
from Features import SignalProcessor

filterwarnings(action='ignore', category=RuntimeWarning)
filterwarnings(action='ignore', category=UserWarning)

if __name__ == '__main__':
    # Prompt with a popup dialog
    choose = open_popup("Do you want complete execution ?")
    DB = ['Mimic']
    if choose == "Yes":
        for DATA in DB:
            processor = SignalProcessor()
            processor.preprocess_signals(DATA)

            TP_Analysis(DATA)
            KF_analysis(DATA)
            PLOT_(DATA)

    else:
        for DATA in DB:
            # if you choose No graphs will be directly plotted
            Plot = ALL_GRAPH_PLOT()

            Plot.GRAPH_RESULT(DATA)
