import numpy as np
import pandas as pd
import os

path1 = "Results/DS2/TP"
path = "Analysis3/Comparative_Analysis/Mimic/"
os.makedirs(path, exist_ok=True)

csv_files = os.listdir(path1)

for csv in csv_files:
    if csv.endswith(".csv") and csv.startswith("Comp_"):

        df = pd.read_csv(os.path.join(path1, csv))

        # Remove model names column
        df_numeric = df.iloc[:, 1:]

        # Convert to numpy (only values)
        data = df_numeric.to_numpy(dtype=np.float32)

        name = csv.split("%")[0]

        if name == "Comp_Accuracy(":
            np.save(os.path.join(path, "ACC_1.npy"), data)

        elif name == "Comp_F1-Score(":
            np.save(os.path.join(path, "F1score_1.npy"), data)

        elif name == "Comp_NPV(":
            np.save(os.path.join(path, "NPV_1.npy"), data)

        elif name == "Comp_PPV(":
            np.save(os.path.join(path, "PPV_1.npy"), data)

        elif name == "Comp_Recall(":
            np.save(os.path.join(path, "REC_1.npy"), data)
