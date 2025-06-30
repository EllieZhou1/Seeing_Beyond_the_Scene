import torch
from datetime import datetime
import os
import pandas as pd
from pytorchvideo.data.encoded_video import EncodedVideo
import csv
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
from decord import VideoReader, cpu
from decord.bridge import set_bridge
set_bridge('torch')


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


json_filename = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/kinetics_classnames.json"

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id --> label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

#Create a classname --> id mapping
kinetics_classname_to_id = {v: k for k, v in kinetics_id_to_classname.items()}

# ========== DATA PIPELINE ==========
class MimeticsDataset(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_videos=None):
        self.max_videos = max_videos
        self.df = pd.read_csv(csv_path)
        #TODO: reduce df to the max_videos if specified


    def __len__(self):
        if self.max_videos is None:
            return len(self.df)
        else:
            return min(len(self.df), self.max_videos)
    
    #Given the path to the frames directory and a list of indicies, load the video frames
    #Returns a tensor of the video frames
    def load_video_frames(self, video_path, num_frames):

        vr = VideoReader(video_path, ctx=cpu(0))
        duration = len(vr)
        target_frame_idx = np.linspace(0, duration-1, num_frames).round().astype(int)

        frames = vr.get_batch(target_frame_idx)
        frames = frames.permute(3,0,1,2)
    
        video_tensor = transform(frames)  # Apply transformations
        return video_tensor
            

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        label = row['label']

        #Shape should be [3, 32, 256, 256] for the fast tensor
        #Shape should be [3, 8, 256, 256] for the slow tensor
        video_path = row['full_path']
        fast_tensor = self.load_video_frames(video_path, 32)
        slow_tensor = fast_tensor[:, ::4]

        inputs=[slow_tensor, fast_tensor]

        result = {
            "inputs": inputs, #a list of 2 tensors
            "label": kinetics_classname_to_id[label],
        }
        return result
    
