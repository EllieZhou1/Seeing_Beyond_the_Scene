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
from transformers import AutoModel, AutoTokenizer, AutoConfig
import matplotlib.pyplot as plt
import math
import wandb
import mmcv
from typing import List, Tuple
import json
from scipy import ndimage
import glob
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ShortSideScale,
) 


ORIGINAL_DIR = "/n/fs/visualai-scr/Data/Kinetics_cvf/frames_highres/val"
SEG_DIR = "/n/fs/visualai-scr/Data/HAT4/seg"
INPAINT_DIR = "/n/fs/visualai-scr/Data/HAT4/inpaint"
THRESH = 128  # segmentation threshold (0..255)

USE_PKL = True


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run action swap experiments")
    parser.add_argument("--mix", type=int, help="The HAT mix to test on (1, 2, or 3)")
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

def get_image(human_path, bg_path, human_idx, background_idx):
    """
    Args:
        human_path (str): Path to the human video (eg: smoking/<ID_start_end>/000001.jpg)
        background_path (str): Path to the background video (eg: smoking/<ID_start_end>/000001.jpg)
    Returns:
        Action Swap Image (A PIL Image)
    """
    fg_rgb_path = os.path.join(ORIGINAL_DIR, human_path, f"{human_idx:06d}.jpg")
    fg_seg_path = os.path.join(SEG_DIR, human_path, f"{human_idx:06d}.png")
    bg_rgb_path = os.path.join(INPAINT_DIR, bg_path, f"{background_idx:06d}.jpg")
    bg_seg_path = os.path.join(SEG_DIR, bg_path, f"{background_idx:06d}.png")
    
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

#Define input transforms
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames_slow = 8

transform = Compose(
        [
            Lambda(lambda x: x/255.0), #Scale values to [0, 1]
            NormalizeVideo(mean, std), #Normalize each channel
            ShortSideScale( #Scale the short side of the video to be "side_size", preserve the aspect ratio
                size=side_size
            ),
            CenterCropVideo(crop_size), #Take the center crop of [256x256]
        ]
)

df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
all_labels = [label for label in df["name"]]

# Compute indices for 8 and 32 evenly spaced frames
def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

# #Given the path to the frames directory and a list of indicies, load the video frames
# #Returns a tensor of the video frames
# def load_video_frames(frames_path, indices):
#     frames = []
#     for i in indices:
#         image_path = os.path.join(frames_path, f"{i:06d}.jpg")  # Assuming frames are named as 000001.jpg, 000002.jpg, etc.
#         img = Image.open(image_path).convert('RGB')  # Load as RGB
#         img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
#         frames.append(img_tensor)

#     video_tensor = torch.stack(frames, dim=1)  # [3, num_frames, H, W]
#     video_tensor = transform(video_tensor)  # Apply transformations
#     return video_tensor


def main():
    args = parse_args()
    mix = args.mix
    PKL_PATH = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/original_hat_actionswap/actionswap_rand_{mix}.pickle"

    df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
    all_labels = df["name"].tolist()
    print(all_labels)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)       
    print("finished Loading model")
    model.to(device)
    model.eval()

    mapping = None

    human_total = 0
    background_total = 0
    total = 0

    #Load the pickle file (dict object, where key = human video path, value = [bg_vid path, _, _])
    if USE_PKL:
        with open(PKL_PATH, "rb") as f:
            mapping = pickle.load(f)
    else:
        mapping = {
            "hitting baseball/idueIYDAbZc_000149_000159": ["eating ice cream/0fCDlKYkRxc_000081_000091", -1, -1]
        }


    json_file = []
    json_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/pretrained_slow_only/hat_actionswap_mix{mix}_results.json"

    for key, value in tqdm(mapping.items()):
        human_path = key
        background_path = value[0]


        full_human_path = os.path.join(ORIGINAL_DIR, human_path)
        full_background_path = os.path.join(INPAINT_DIR, background_path)

        num_files_A = len(glob.glob(os.path.join(full_human_path, "*")))
        num_files_B = len(glob.glob(os.path.join(full_background_path, "*")))


        human_label = human_path.split("/")[0]
        background_label = background_path.split("/")[0]

        human_indices = sample_indices(8, num_files_A)
        background_indices = sample_indices(8, num_files_B)

        frames = []

        for i in range(0, 8): #Go through each human, bg frame pair
            human_idx = human_indices[i]
            background_idx = background_indices[i]
            image = get_image(human_path, background_path, human_idx, background_idx) #PIL Image of the HT Action Swap Image
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # [C, H, W]
            frames.append(image_tensor)

        video_tensor = torch.stack(frames, dim=1).float()  # [3, num_frames, H, W]
        video_tensor = transform(video_tensor)  # Apply transformations
        video_tensor = video_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(video_tensor)
            print("Outputs", outputs.shape)
            output_list = outputs.tolist()[0]
            predicted = torch.argmax(outputs).item()
            predicted_label = all_labels[predicted] #Get the label that got predicted
            
            # print("human label", human_label, "background label", background_label, "predicted label", predicted_label)
            if predicted_label == human_label:
                human_total += 1
            elif predicted_label == background_label:
                background_total += 1
            total += 1
        
        json_file.append({
            "human_path":human_path,
            "background_path":background_path,
            "predictions":output_list
        })
        
    print("Human total", human_total)
    print("Background total", background_total)
    print("Total", total)
    print("Human acc", human_total/total)
    print("Background error", background_total/total)

    with open(json_path, "w") as f:
        json.dump(json_file, f, indent=4)



if __name__ == "__main__":
    main()