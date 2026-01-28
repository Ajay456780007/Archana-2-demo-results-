import os
import numpy as np

def sync_last_row_from_comparative_to_performance(DB):
    comp_base = f"Analysis3/KF_Comp/{DB}"
    perf_base = f"Analysis3/KF_Perf/{DB}"

    files = ["ACC_1.npy", "F1score_1.npy", "NPV_1.npy", "PPV_1.npy", "REC_1.npy"]

    for fname in files:
        comp_path = os.path.join(comp_base, fname)
        perf_path = os.path.join(perf_base, fname)

        # Load both files
        comp_data = np.load(comp_path)
        perf_data = np.load(perf_path)

        if comp_data.shape[1] != perf_data.shape[1]:
            raise ValueError(f"Column mismatch in {fname}")

        # Take last row from Comparative Analysis
        last_row = comp_data[-1].copy()

        # Replace last row in Performance Analysis
        perf_data[-1] = last_row

        # Save back
        np.save(perf_path, perf_data)

        print(f"Synchronized last row for: {fname}")

# Run
sync_last_row_from_comparative_to_performance("Mimic")
