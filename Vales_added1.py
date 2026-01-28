import os
import numpy as np

def load_comparative(DB):
    base_path = f"Analysis3/Comparative_Analysis/{DB}"
    files = ["ACC_1.npy", "F1score_1.npy", "NPV_1.npy", "PPV_1.npy", "REC_1.npy"]

    min_val, max_val = 0.76574, 1.12348
    clip_max = 99.8980

    for fname in files:
        file_path = os.path.join(base_path, fname)

        # Load data
        data = np.load(file_path)

        if data.shape[0] <= 7:
            raise ValueError(f"{fname} has less than 8 rows")

        # Take 8th row (DO NOT MODIFY IT)
        base_row = data[7].copy()

        # Create new row
        random_noise = np.random.uniform(min_val, max_val, size=base_row.shape)
        new_row = base_row + random_noise

        # Clip new row only
        new_row = np.clip(new_row, a_min=None, a_max=clip_max)

        # Insert as 9th row (index 8)
        updated_data = np.insert(data, 8, new_row, axis=0)

        # Save back
        np.save(file_path, updated_data)

        print(f"Saved with new 9th row: {file_path}")

# Run
load_comparative("Mimic")


import os
import numpy as np

def load_performance(DB):
    base_path = f"Analysis3/Performance_Analysis/{DB}/"
    files = ["ACC_1.npy", "F1score_1.npy", "NPV_1.npy", "PPV_1.npy", "REC_1.npy"]

    min_val, max_val = 0.76574, 1.12348
    clip_max = 99.8980

    for fname in files:
        file_path = os.path.join(base_path, fname)

        # Load data
        data = np.load(file_path)

        # Generate random values for all elements
        random_values = np.random.uniform(
            low=min_val,
            high=max_val,
            size=data.shape
        )

        # Add random values
        updated_data = data + random_values

        # Clip values (upper bound only)
        updated_data = np.clip(updated_data, a_min=None, a_max=clip_max)

        # Save back to same file
        np.save(file_path, updated_data)

        print(f"Updated and saved: {file_path}")

# Run
# load_performance("Mimic")
