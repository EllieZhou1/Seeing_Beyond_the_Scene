#Eval SIGLIP2 on HAT Action Swap

import pickle
import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm.auto import tqdm
import csv
import pandas
import wandb
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoProcessor, AutoModel
import matplotlib.pyplot as plt
import math
import wandb
import mmcv
from typing import List, Tuple
import json
from scipy import ndimage
import glob



ORIGINAL_DIR = "/n/fs/visualai-scr/Data/Kinetics_cvf/frames_highres/val"
SEG_DIR = "/n/fs/visualai-scr/Data/HAT4/seg"
INPAINT_DIR = "/n/fs/visualai-scr/Data/HAT4/inpaint"
THRESH = 128  # segmentation threshold (0..255)

USE_PKL = True

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="CLIP on HAT Action Swap")
    parser.add_argument("--mix", type=int, help="Mix (1, 2, or 3)")
    return parser.parse_args()

## my code ###
def center_of_mass_or_center(mask: np.ndarray) -> Tuple[float, float]:
    """Return (row, col) center of mass; fall back to geometrical center if empty."""
    if mask.sum() > 0:
        c = ndimage.measurements.center_of_mass(mask)
        return float(c[0]), float(c[1])
    h, w = mask.shape[:2]
    return h / 2.0, w / 2.0

## my code ###
def paste_foreground_on_background(fg_img: Image.Image, fg_mask: np.ndarray, bg_img: Image.Image, move_rc: Tuple[int, int]) -> Image.Image:
    """Paste fg_img onto bg_img using binary mask and (row, col) shift."""
    fg_img = Image.fromarray(fg_img).convert("RGB")
    bg_img = Image.fromarray(bg_img).convert("RGB")
    r_shift, c_shift = move_rc
    # Ensure mask is 0..255 uint8
    mask = (fg_mask > THRESH).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).convert("L")
    bg_img = bg_img.copy()
    bg_img.paste(fg_img, (int(c_shift), int(r_shift)), mask_pil)
    return bg_img

def get_image(human_path, bg_path):
    """
    Args:
        human_path (str): Path to the human video (eg: smoking/<ID_start_end>)
        background_path (str): Path to the background video (eg: smoking/<ID_start_end>)
    Returns:
        Action Swap Image (A PIL Image)
    """

    fg_rgb_path = os.path.join(ORIGINAL_DIR, human_path)
    fg_seg_path = os.path.join(SEG_DIR, human_path)
    bg_rgb_path = os.path.join(INPAINT_DIR, bg_path)
    bg_seg_path = os.path.join(SEG_DIR, bg_path)

    human_file_count = len(glob.glob(f"{fg_rgb_path}/*"))
    background_file_count = len(glob.glob(f"{bg_rgb_path}/*"))

    fg_rgb_path = os.path.join(fg_rgb_path, f"{int(human_file_count/2):06d}.jpg")
    fg_seg_path = os.path.join(fg_seg_path, f"{int(human_file_count/2):06d}.png")
    bg_rgb_path = os.path.join(bg_rgb_path, f"{int(background_file_count/2):06d}.jpg")
    bg_seg_path = os.path.join(bg_seg_path, f"{int(background_file_count/2):06d}.png")

    # print("Foreground RGB Path:", fg_rgb_path)
    # print("Foreground Segmentation Path:", fg_seg_path)
    # print("Background RGB Path:", bg_rgb_path)
    # print("Background Segmentation Path:", bg_seg_path)
    
    # Load images and masks
    fg_img = np.array(Image.open(fg_rgb_path).convert("RGB"))
    fg_mask =np.array(Image.open(fg_seg_path).convert("L"))  # single-channel mask 0..255
    bg_img = np.array(Image.open(bg_rgb_path).convert("RGB"))
    bg_mask = np.array(Image.open(bg_seg_path).convert("L"))

    h, w = fg_img.shape[0], fg_img.shape[1]

    if h < w: #Scale the height of background to match the short side (height of fg)
        new_h = h
        new_w = int(round(h / bg_img.shape[0] * bg_img.shape[1]))
        bg_img = mmcv.imresize(bg_img, (new_w, new_h))
        bg_mask = mmcv.imresize(bg_mask, (new_w, new_h))
    else: #Scale the width of background to match the short side (width of fg)
        new_w = w
        new_h = int(round(w / bg_img.shape[1] * bg_img.shape[0]))
        bg_img = mmcv.imresize(bg_img, (new_w, new_h))
        bg_mask = mmcv.imresize(bg_mask, (new_w, new_h))

    # Compute centers
    fg_center_r, fg_center_c = center_of_mass_or_center(np.array(fg_mask))
    bg_center_r, bg_center_c = center_of_mass_or_center(np.array(bg_mask))
    move = (int(round(bg_center_r - fg_center_r)), int(round(bg_center_c - fg_center_c)))

    # Paste
    composed = paste_foreground_on_background(fg_img, fg_mask, bg_img, move)
    # composed.save("image.jpg")  # Save for debugging
    return composed


def main():
    args = parse_args()
    # run = wandb.init(
    #     project="SlowFast_Kinetics",
    #     name=f"SIGLIP2 on Mix {args.mix}",
    #     # mode='online',
    #     mode='disabled',
    #     settings=wandb.Settings(_service_wait=300)
    # )

    print("Args", args)
    mix = args.mix

    # run.log({
    #     "mix": mix
    # })

    # from collections import defaultdict
    # default_entry = lambda: {"pred_human": 0, "pred_background": 0, "total": 0}
    # per_human_class = defaultdict(default_entry)
    # per_background_class = defaultdict(default_entry)

    json_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/original_hat_actionswap/actionswap_mcq_rand{mix}.json"

    rows = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip blanks
                rows.append(json.loads(line))

    # df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
    # all_labels = df["name"].tolist()
    # print(all_labels)

    # Load CLIP model

    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

    print("Loaded SIGLIP2 model")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    mapping = None

    human_total = 0
    background_total = 0
    total = 0

    # json_file = []

    # #Load the pickle file (dict object, where key = human video path, value = [bg_vid path, _, _])
    # if USE_PKL:
    #     with open(PKL_PATH, "rb") as f:
    #         mapping = pickle.load(f)
    # else:
    #     mapping = {
    #         "hitting baseball/idueIYDAbZc_000149_000159": ["eating ice cream/0fCDlKYkRxc_000081_000091", -1, -1]
    #     }

    for item in tqdm(rows):
        human_path = item["human_path"]
        background_path = item["background_path"]
        choices = item["choices"]

        image = get_image(human_path, background_path)

        inputs = processor(text=choices, images=[image], return_tensors="pt",
                            padding="max_length", max_length=64).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
        probs = torch.sigmoid(logits_per_image)  # convert to probabilities

        predicted_label = choices[probs.argmax()]
        human_label = human_path.split("/")[0]  # Extract the label from the key
        background_label = background_path.split("/")[0]  # Extract the label from the background path

        if predicted_label == human_label:
            human_total += 1
            # per_human_class[human_label]["pred_human"] += 1
            # per_background_class[background_label]["pred_human"] += 1
        if predicted_label == background_label:
            background_total += 1
            # per_human_class[human_label]["pred_background"] += 1
            # per_background_class[background_label]["pred_background"] += 1

        # per_human_class[human_label]["total"] += 1
        # per_background_class[background_label]["total"] += 1

        total += 1

    print("Human total:", human_total)
    print("Background total:", background_total)
    print("Total:", total)
    print("Human Accuracy:", human_total/total)
    print("Background Error:", background_total/total)

    # run.log({
    #     "human total": human_total,
    #     "background total": background_total,
    #     "total": total,
    #     "Human Accuracy": human_total / total,
    #     "Background Error": background_total / total
    # })

    # with open (f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/siglip/mix_{mix}_actionswap_per_human_class.json", "w") as f:
    #     json.dump(per_human_class, f, indent=4)

    # with open (f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/siglip/mix_{mix}_actionswap_per_background_class.json", "w") as f:
    #     json.dump(per_background_class, f, indent=4)

if __name__ == "__main__":
    main()




        




    
