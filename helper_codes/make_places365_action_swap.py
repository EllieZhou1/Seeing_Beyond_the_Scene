import os
import csv
import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import concurrent.futures

base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"

#TODO: Change to be validation
minikinetics_csv_path = os.path.join(base_dir, "dataset/minikinetics50/new_minikinetics50_validation_all.csv")
places365_csv_path= os.path.join(base_dir, "dataset/places365.csv")

summary_csv_path = os.path.join(base_dir, "dataset/places365/new_places365_actionswap_validation_summary.csv")
out_csv_path = os.path.join(base_dir, "dataset/places365/new_places365_actionswap_validation.csv")
out_img_path = os.path.join(base_dir, "dataset/places365/new_places365_actionswap_validation")

#Go through each row in minikinetics_csv_path
#Randomly sample a random element in places365
# Write the minikinetics50 & the selected places365 image & label into summary_csv_path, which will keep track
    #of what was paired with what

#Now for creating the actual image and csv that I am going to be using the dataset (has to be same format as the training one)
    # Create the action swap image & write it to out_img_path
    # Write it to out_csv_path with the following fields:
    # label,youtube_id,time_start,time_end,split,full_path,num_files,segmented_path,mask_path

df_mk = pd.read_csv(minikinetics_csv_path)
df_places365= pd.read_csv(places365_csv_path)

summary = []
actionswap_rows = []

def process_row(row):
    import random  # Needed inside function if using multiprocessing
    random_sample = df_places365.sample(n=1) #randomly sample a row in places365
    places365_label = random_sample['label'].iloc[0]
    places365_path = random_sample['full_path'].iloc[0]

    mk_label = row['label']
    mk_youtube_id = row['youtube_id']
    mk_time_start = row['time_start']
    mk_time_end = row['time_end']
    mk_full_path = row['full_path']
    mk_num_files = row['num_files']
    mk_segmented_path = row['segmented_path']
    mk_mask_path = row['mask_path']

    dictionary = {'places365_label':places365_label, 'places365_full_path':places365_path, 'mk_label':mk_label, 'mk_full_path':mk_full_path}

    pil_places365 = Image.open(places365_path).convert("RGB")
    label_path = os.path.join(out_img_path, mk_label, f"{mk_youtube_id}_{mk_time_start:06d}_{mk_time_end:06d}")
    os.makedirs(label_path, exist_ok=True)

    for index in range (1, 33):
        seg_path = os.path.join(mk_segmented_path, f"{index:06d}.jpg")
        mask_path = os.path.join(mk_mask_path, f"{index:06d}.jpg")

        np_seg = np.array(Image.open(seg_path).convert("RGB"))
        np_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        np_mask /= 255.0
        np_mask = np.stack([np_mask]*3, axis=-1)

        target_size = (np_seg.shape[1], np_seg.shape[0])  # (width, height)

        places365_resized = pil_places365.resize(target_size, Image.Resampling.LANCZOS)
        places365_resized = np.array(places365_resized)

        action_swap_img = np_seg + (1.0 - np_mask) * places365_resized
        action_swap_img = np.clip(action_swap_img, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(action_swap_img, mode="RGB")

        img_path = os.path.join(label_path, f"{index:06d}.jpg")
        pil_img.save(img_path)

    new_row = {
        'label':mk_label, 
        'youtube_id':mk_youtube_id, 
        'time_start':mk_time_start, 
        'time_end':mk_time_end, 
        'full_path':label_path,
        'num_files':mk_num_files,
        'segmented_path':mk_segmented_path,
        'mask_path':mk_mask_path
    }

    return dictionary, new_row

with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
    results = list(tqdm(executor.map(process_row, [row for _, row in df_mk.iterrows()]), total=len(df_mk)))

for dictionary, new_row in results:
    summary.append(dictionary)
    actionswap_rows.append(new_row)


df_summary = pd.DataFrame(summary)
df_summary.to_csv(summary_csv_path, index=False)

df_actionswap = pd.DataFrame(actionswap_rows)
df_actionswap.to_csv(out_csv_path, index=False)
