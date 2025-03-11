import pandas as pd
import numpy as np
import os
import config

output_dir = os.path.dirname(config.CONFIG['path_to_augmented_data'])
os.makedirs(output_dir, exist_ok=True)

data_path = config.CONFIG['data_path']
df = pd.read_csv(data_path)
numeric_columns = ['Power', 'Current', 'PF']
for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce') 
gap_data = df[(df['Power'] >= 185) & (df['Power'] <= 190)].copy()
augmented_data = []
for _, row in gap_data.iterrows():
    for _ in range(10):
        noise = np.random.normal(0, 0.5, size=len(numeric_columns))
        new_row = row.copy()
        for col, n in zip(numeric_columns, noise):
            new_row[col] = new_row[col] + n
        new_row['Power'] = np.clip(new_row['Power'], 185, 190)
        augmented_data.append(new_row)

augmented_df = pd.concat([df] + [pd.DataFrame(augmented_data)], ignore_index=True)
path_to_augmented_data = config.CONFIG['path_to_augmented_data']
try:
    augmented_df.to_csv(path_to_augmented_data, index=False)
    print(f"Augmented data saved to {path_to_augmented_data}")
except PermissionError as e:
    print(f"PermissionError: {e}. Try running as administrator or check file permissions.")
except Exception as e:
    print(f"An error occurred: {e}")