import pandas as pd
import os
from tqdm import tqdm

# Load the CSV
csv_path = "dataset/minikinetics50/minikinetics50_train_all.csv"

df = pd.read_csv(csv_path)


# Shift rows before index 25742 one column to the left
misaligned = df.iloc[25742:].copy()
aligned = pd.DataFrame(columns=df.columns)

for i, row in tqdm(misaligned.iterrows()):
    shifted = row[1:].tolist() + [None]  # shift values left by one
    aligned.loc[i] = shifted

df.update(aligned)
df.to_csv(csv_path, index=False)


# # Fix misaligned rows starting from index 25742
# misaligned = df.iloc[25742:].copy()
# aligned = pd.DataFrame(columns=df.columns)

# for i, row in misaligned.iterrows():
#     shifted = [None] + row[:-1].tolist()  # shift values right by one
#     aligned.loc[i] = shifted

# df.update(aligned)
# # # Add a folder to each path in the 'action_swap_path' column
# # df['mask_path'] = df['mask_path'].apply(
# #     lambda path: path.replace("/dataset/segmented_minikinetics50/train_masks/", "/dataset/minikinetics50/binarymasks_minikinetics50_train")
# # )

# # df['segmented_path'] = df['segmented_path'].apply(
# #     lambda path: path.replace("/dataset/segmented_minikinetics50/train/", "/dataset/minikinetics50/segmented_minikinetics50_train")
# # )

# Save back to CSV
# df.to_csv(csv_path, index=False)