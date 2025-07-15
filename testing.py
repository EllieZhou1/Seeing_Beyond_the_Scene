#this is a copy of slowfast.ipynb


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
from tqdm.auto import tqdm
import wandb

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
    
    ApplyTransformToKey,
    ShortSideScale,
    UniformTemporalSubsample,
    UniformCropVideo
) 
import concurrent.futures

#Define input transforms
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256


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

def sample_indices(n, num_frames):
    #if num_frames < n:
        #raise ValueError(f"Requested {n} frames, but only {num_frames} available.")
    #num_frames = min(num_frames, n)  # Ensure num_frames is at least n
    return [int(round(i * (num_frames - 1) / (n - 1) + 1)) for i in range(n)]

def load_video_frames(frames_path, indices):
    frames = []
    for i in indices:
        image_path = os.path.join(frames_path, f"{i:06d}.jpg")  # Assuming frames are named as 000001.jpg, 000002.jpg, etc.
        img = Image.open(image_path).convert('RGB')  # Load as RGB
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # [C, H, W]
        frames.append(img_tensor)
    
    video_tensor = torch.stack(frames, dim=1)  # [3, num_frames, H, W]
    video_tensor = transform(video_tensor)  # Apply transformations
    return video_tensor

# tensor = load_video_frames('/n/fs/visualai-scr/Data/Kinetics_cvf/frames/train/testifying/---QUuC4vJs_000084_000094', 
#                            [1, 11, 20, 30, 40, 49, 59, 69, 78, 88, 97, 107, 117, 126, 136, 146, 155, 165, 175, 
#                             184, 194, 204, 213, 223, 232, 242, 252, 261, 271, 281, 290, 300])

# transformed_tensor = transform(tensor)
# print(transformed_tensor.shape)
print(sample_indices(32, 12))

print(len(sample_indices(32, 12)))

#df = pd.read_csv("/n/fs/visualai-scr/Data/Kinetics_cvf/raw/train.csv")


# print("Initial length of dataset is ", len(df))
# df['full_path'] = df.apply(
#             lambda row: os.path.join(
#                 "/n/fs/visualai-scr/Data/Kinetics_cvf/frames/train.csv",
#                 row['split'],
#                 row['label'],
#                 f"{row['youtube_id']}_{int(row['time_start']):06d}_{int(row['time_end']):06d}"
#             ),
#             axis=1
#         )
# df = df[df['full_path'].apply(os.path.exists)].reset_index(drop=True)
# print("Length of dataset after removing non-existing paths is ", len(df))

# df['num_files'] = df['full_path'].apply(lambda p: sum(1 for entry in os.scandir(p) if entry.is_file()))
# df = df[df['num_files'] > 0].reset_index(drop=True)

# df.to_csv('/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clean_train.csv', index=False)

# print("Length of dataset after removing empty directories is ", len(df))

import sklearn
from sklearn.metrics import classification_report
report = classification_report(y_true=[1, 2, 3], y_pred=[1, 2, 3], labels=[1,2,3,4,5], target_names=['class0', 'class1', 'class2', 'class3', 'class4'])
## print(report)

