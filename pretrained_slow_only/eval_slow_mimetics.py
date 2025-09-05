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

    mimetics_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/new_mimetics_all.csv"
    df = pd.read_csv(mimetics_csv)

    mapping = None

    human_total = 0
    total = 0

    json_file = []
    json_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/pretrained_slow_only/mimetics_results.json"

    for index, row in tqdm(df.iterrows()):
        full_path = row["full_path"]
        num_files = row["num_files"]
        label = row["label"]

        indices = sample_indices(8, num_files)

        frames = []

        for i in indices: #Go through each human, bg frame pair
            image = Image.open(os.path.join(full_path, f"{i:06d}.jpg")).convert("RGB") #Load the mimetics frame
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
            if predicted_label == label:
                human_total += 1

            total += 1
        
        json_file.append({
            "full_path":full_path,
            "label":label,
            "predictions":output_list
        })
        
    print("Human total", human_total)
    print("Total", total)
    print("Human acc", human_total/total)

    with open(json_path, "w") as f:
        json.dump(json_file, f, indent=4)

if __name__ == "__main__":
    main()