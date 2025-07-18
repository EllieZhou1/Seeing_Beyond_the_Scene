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
from PIL import Image
import shutil
from tqdm import tqdm

#has person: ,label,youtube_id,time_start,time_end,split,full_path,num_files,segmented_path,mask_path
#all: label,youtube_id,time_start,time_end,split,full_path,num_files

base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"
df_all = pd.read_csv(os.path.join(base_dir, "dataset/mini50_clean_train.csv"))
df_hasPerson = pd.read_csv(os.path.join(base_dir, "dataset/minikinetics50/segmented_minikinetics50_train.csv"))
out_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/minikinetics50_train_all.csv"

human_paths = set(df_hasPerson['full_path']) #A set of all paths which contains person in video
df_noPerson = df_all[~df_all['full_path'].isin(human_paths)]

df_hasPerson = df_hasPerson.drop(columns=["Unnamed: 0"], errors='ignore')

df_hasPerson['mask_path'] = df_hasPerson['mask_path'].apply(
    lambda path: path.replace("/dataset/segmented_minikinetics50/train_masks/", "/dataset/minikinetics50/binarymasks_minikinetics50_train/")
)
df_hasPerson['segmented_path'] = df_hasPerson['segmented_path'].apply(
    lambda path: path.replace("/dataset/segmented_minikinetics50/train/", "/dataset/minikinetics50/segmented_minikinetics50_train/")
)

df_hasPerson.to_csv(out_csv, index=False)

print(f"Found a total of {len(df_hasPerson)} to contain human")
print(f"Found a total of {len(df_noPerson)} to NOT contain human")
print(f"Found a total of {len(df_all)} in TOTAL in MiniKinetics50 Validation Set")


new_rows = []
for index, row in tqdm(df_noPerson.iterrows(), desc='saving stuff'):

    label = row['label']
    youtube_id = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']
    full_path = row['full_path']
    num_files = row['num_files']

    vid_name = f"{youtube_id}_{time_start:06d}_{time_end:06d}"

    
    seg_path = os.path.join(base_dir, "dataset/minikinetics50/segmented_minikinetics50_train", label, vid_name) 
    mask_path = os.path.join(base_dir, "dataset/minikinetics50/binarymasks_minikinetics50_train", label, vid_name) 

    if os.path.exists(seg_path):
        shutil.rmtree(seg_path)
    if os.path.exists(mask_path):
        shutil.rmtree(mask_path)

    # print("Path to save segmented video:", seg_path)
    # print("Path to binary mask:", mask_path)
    os.makedirs(seg_path, exist_ok=False) #its not supposed to exist yet...
    os.makedirs(mask_path, exist_ok=False)

    with Image.open(f"{full_path}/000001.jpg") as img:
        width, height = img.size
    
    np_array = np.zeros((height, width, 3), dtype=np.uint8) #the "segmented img" which is JUST zeroes

    #Save the segmented images
    for i in range (0, 32):
        pil_img = Image.fromarray(np_array)
        pil_img.save(f"{seg_path}/{i:06d}.jpg")
        pil_img.save(f"{mask_path}/{i:06d}.jpg")
    
    row['segmented_path'] = seg_path
    row['mask_path'] = mask_path
    new_rows.append(row)

pd.DataFrame(new_rows).to_csv(out_csv, mode='a', header=False, index=False)
