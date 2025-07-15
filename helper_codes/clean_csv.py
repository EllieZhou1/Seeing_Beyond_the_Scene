from datetime import datetime
import os
import pandas as pd

df = pd.read_csv("/n/fs/visualai-scr/Data/Kinetics_cvf/raw/validate.csv")

print("Initial length of dataset is ", len(df))
df['full_path'] = df.apply(
            lambda row: os.path.join(
                "/n/fs/visualai-scr/Data/Kinetics_cvf/frames/",
                "val",
                row['label'],
                f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}"
            ),
            axis=1
        )
df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)
print("Length of dataset after removing non-existing paths is ", len(df))

df['num_files'] = df['full_path'].apply(lambda p: sum(1 for entry in os.scandir(p) if entry.is_file()))
df = df[df['num_files'] > 0].reset_index(drop=True)

df.to_csv('/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clean_validate.csv', index=False)

print("Length of dataset after removing empty directories is ", len(df))
