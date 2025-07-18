#Make binary masks from the minikinetics50 segmented dataset

#Steps
#1. Load the csv into a df
#2. For each video, create a binary mask for the vids
#3. Save the binary masks in a folder --> train_masks, val_masks

import pandas as pd
import torch
import glob
import os
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

og_csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_all.csv"
df = pd.read_csv(og_csv_path)
folder_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/binarymask_action_swap"
os.makedirs(folder_path, exist_ok=True)

# Helper function for multiprocessing
def process_row(row):
    segmented_path = row['segmented_path']
    action_swap_path = row['action_swap_path']
    action_swap_label = os.path.basename(action_swap_path)
    output_paths = []

    for file_path in glob.glob(os.path.join(segmented_path, "*.jpg")):
        img = Image.open(file_path).convert("RGB")
        np_img = np.array(img)
        binary_mask = (np_img.sum(axis=2) > 20).astype(np.uint8)
        mask_img = Image.fromarray(binary_mask * 255)

        save_path = os.path.join(folder_path, action_swap_label, os.path.basename(file_path))
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        mask_img.save(save_path)

    return os.path.join(folder_path, action_swap_label)

with ProcessPoolExecutor(max_workers=8) as executor:
    mask_paths = list(tqdm(executor.map(process_row, [row for _, row in df.iterrows()]), total=len(df)))

df["mask_path"] = mask_paths
df.to_csv(og_csv_path, index=False)