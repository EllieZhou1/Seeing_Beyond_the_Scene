import torch
from datetime import datetime
import os
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
import numpy as np
from PIL import Image

import torch.optim as optim
import torch.nn as nn
import glob
import wandb
import argparse

from typing import Dict
import json
import urllib

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
num_frames_fast = 32
num_frames_slow = 8
device = "cuda" if torch.cuda.is_available() else "cpu"

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


json_filename = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/class50_to_label.json"

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id --> label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

#Create a classname --> id mapping
kinetics_classname_to_id = {v: k for k, v in kinetics_id_to_classname.items()}

# ========== DATA PIPELINE ==========

class KineticsDataset2(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_videos=None):
        self.max_videos = max_videos
        self.df = pd.read_csv(csv_path)
        #TODO: reduce df to the max_videos if specified


    def __len__(self):
        if self.max_videos is None:
            return len(self.df)
        else:
            return min(len(self.df), self.max_videos)

    
    # Compute indices for 8 and 32 evenly spaced frames
    def sample_indices(self, n, total_frames):
        return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]
    
    #Given the path to the frames directory and a list of indicies, load the video frames
    #Returns a tensor of the video frames
    def load_video_frames(self, frames_path, indices):
        frames = []
        for i in indices:
            image_path = os.path.join(frames_path, f"{i:06d}.jpg")  # Assuming frames are named as 000001.jpg, 000002.jpg, etc.
            img = Image.open(image_path).convert('RGB')  # Load as RGB
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
            #img_tensor = img_tensor.to(device)
            frames.append(img_tensor)
    
        video_tensor = torch.stack(frames, dim=1)  # [3, num_frames, H, W]
        video_tensor = transform(video_tensor)  # Apply transformations
        return video_tensor
            

    def __getitem__(self, idx):
        #if(idx % 10 == 0):
        # print("Get item ", idx)

        row = self.df.iloc[idx]
        label = row['label']

        total_frames = row ['num_files']

        idx_fast = self.sample_indices(num_frames_fast, total_frames)
        idx_slow = self.sample_indices(num_frames_slow, total_frames)

        #Shape should be [3, 32, 256, 256] for the fast tensor
        #Shape should be [3, 8, 256, 256] for the slow tensor
        fast_tensor = self.load_video_frames(row['full_path'], idx_fast)
        slow_tensor = self.load_video_frames(row['full_path'], idx_slow)

        inputs=[slow_tensor, fast_tensor]

        result = {
            "inputs": inputs, #a list of 2 tensors
            "label": kinetics_classname_to_id[label],
        }
        return result
    

class HAT_bg_Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_videos=None):
        self.max_videos = max_videos
        self.df = pd.read_csv(csv_path)
        #TODO: reduce df to the max_videos if specified


    def __len__(self):
        if self.max_videos is None:
            return len(self.df)
        else:
            return min(len(self.df), self.max_videos)

    
    # Compute indices for 8 and 32 evenly spaced frames
    def sample_indices(self, n, total_frames):
        return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]
    
    #Given the path to the frames directory and a list of indicies, load the video frames
    #Returns a tensor of the video frames
    def load_video_frames(self, frames_path, indices):
        frames = []
        for i in indices:
            image_path = os.path.join(frames_path, f"{i:06d}.jpg")  # Assuming frames are named as 000001.jpg, 000002.jpg, etc.
            img = Image.open(image_path).convert('RGB')  # Load as RGB
            img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
            #img_tensor = img_tensor.to(device)
            frames.append(img_tensor)
    
        video_tensor = torch.stack(frames, dim=1)  # [3, num_frames, H, W]
        video_tensor = transform(video_tensor)  # Apply transformations
        return video_tensor
            

    def __getitem__(self, idx):
        #if(idx % 10 == 0):
        print("Get item ", idx)

        row = self.df.iloc[idx]
        label = row['label']

        total_frames = row ['num_files']

        idx_fast = self.sample_indices(num_frames_fast, total_frames)
        idx_slow = self.sample_indices(num_frames_slow, total_frames)

        #Shape should be [3, 32, 256, 256] for the fast tensor
        #Shape should be [3, 8, 256, 256] for the slow tensor
        fast_tensor = self.load_video_frames(row['full_path'], idx_fast)
        slow_tensor = self.load_video_frames(row['full_path'], idx_slow)

        inputs=[slow_tensor, fast_tensor]

        result = {
            "inputs": inputs, #a list of 2 tensors
            "label": kinetics_classname_to_id[label],
        }
        return result
    
