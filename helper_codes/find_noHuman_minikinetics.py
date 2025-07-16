# This will compare dataset/mini50_clean_train.csv and dataset/segmented_minikinetics50/segmented_minikinetics50_train.csv
# 1. Find the rows which are present in mini50_clean_train.csv, but not in the other 
#   a) aka the videos which were not detected to have humans
# 2. Add these data fields: segmented_path,mask_path
# 3. Create segmentation with just all zeroes and mask with all zeroes
#      zero_frame = np.zeros_like(frame, dtype=np.uint8)
#      segmented_frames.append(zero_frame)

import pandas as pd
import csv
import torch
import numpy as np
import os

base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"
df_all = pd.read_csv(os.path.join(base_dir, "dataset/mini50_clean_train.csv"))
df_hasPerson = pd.read_csv(os.path.join(base_dir, "dataset/segmented_minikinetics50/segmented_minikinetics50_train.csv"))

