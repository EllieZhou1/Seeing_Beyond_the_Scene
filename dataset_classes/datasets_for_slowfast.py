#REMOVING BACKGROUND
#Use


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
import gc

from typing import Dict
import json
import urllib
from matplotlib import pyplot as plt
import os
import sys
import shutil
import tempfile

os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2")
sys.path.insert(0, "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2")
# print("current working direcotry", os.getcwd())
from sam2.build_sam import build_sam2_video_predictor
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")
# print("current working direcotry", os.getcwd())


from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import (
    CenterCropVideo,
    NormalizeVideo,
)
from pytorchvideo.data.encoded_video import EncodedVideo
from pytorchvideo.transforms import (
    ShortSideScale,
) 

import multiprocessing as mp
mp.set_start_method('spawn', force=True)

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
class KineticsDataset_RemoveBG(torch.utils.data.Dataset):
    def __init__(self, csv_path, max_videos=None):
        self.max_videos = max_videos
        self.df = pd.read_csv(csv_path)
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

        sam2_checkpoint = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2/checkpoints/sam2.1_hiera_large.pt"

        #model_cfg, expects a config path, not an absolute path
        os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2/sam2")
        model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
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


        #Make a temporary directory with only the sampled frames
        temp_dir = tempfile.mkdtemp()
        for i in indices:
            source = os.path.join(frames_path, f"{i:06d}.jpg")
            dest = os.path.join(temp_dir, f"{i:06d}.jpg")
            os.symlink(source, dest)

            img = Image.open(source).convert('RGB')  # Load as RGB
            img_np = np.array(img) #convert to np image
            frames.append(img_np)
    
        video_np = np.stack(frames, axis=0)
        print("VIDEO NP shape", video_np.shape)

        img0_np = video_np[0]
        results = self.yolo(img0_np) #get bbox for the first frame
        bboxes = results.xyxy[0]
        person_bboxes = bboxes[bboxes[:, 5] == 0] # get bboxes for class == person only
        person_bboxes = person_bboxes[:, :4].cpu().numpy()
        
        inference_state = self.predictor.init_state(video_path=temp_dir)

        #If a person was detected, then segment out the person
        if person_bboxes.shape[0] >= 1:
            box = np.array(person_bboxes[0], dtype=np.float32)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box=box,
            )

            imgs = []

            #Go through each video frame, propogate the segmentation, calculate the binary mask
            for out_frame_idx, _, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                binarymask = (out_mask_logits.squeeze(0).squeeze(0)) > 0.0 #binary mask, where 1 is person, 0 isnot
                binarymask = binarymask.unsqueeze(2).to(device)

                img = torch.from_numpy(video_np[out_frame_idx]).to(device)
                seg_img = binarymask * img
                imgs.append(seg_img)

                seg_img_np = seg_img.cpu().numpy()
                if seg_img_np.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                    seg_img_np = np.transpose(seg_img_np, (1, 2, 0))

                seg_img_np = np.clip(seg_img_np, 0, 255).astype(np.uint8)
                print("Seg image shape:", seg_img_np.shape, seg_img_np.min(), seg_img_np.max())
                Image.fromarray(seg_img_np).save(f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/segmented_frames/seg_img_{out_frame_idx}.png")

            video_tensor = torch.stack(imgs, dim=0)

            #Delete the temp directory
            shutil.rmtree(temp_dir)  # Clean up after
        
        else:
            print(f"Could not find a person in this video: {frames_path}")
            video_tensor = torch.from_numpy(video_np)
        

        video_tensor = video_tensor.permute(3, 0, 1, 2)
        print("Video_tensor shape: ", video_tensor.shape)

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

    def show_mask(mask, ax, obj_id=None, random_color=False):
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)


    def show_points(coords, labels, ax, marker_size=200):
        pos_points = coords[labels==1]
        neg_points = coords[labels==0]
        ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
        ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


    def show_box(box, ax):
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


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
        fast_tensor = self.load_video_frames(video_path, 64)
        slow_tensor = fast_tensor[:, ::8]

        inputs=[slow_tensor, fast_tensor]

        result = {
            "inputs": inputs, #a list of 2 tensors
            "label": kinetics_classname_to_id[label],
        }
        return result
