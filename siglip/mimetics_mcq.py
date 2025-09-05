import pickle
import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModel
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



ORIGINAL_DIR = "/n/fs/visualai-scr/Data/Kinetics_cvf/frames_highres/val"
SEG_DIR = "/n/fs/visualai-scr/Data/HAT4/seg"
INPAINT_DIR = "/n/fs/visualai-scr/Data/HAT4/inpaint"
THRESH = 128  # segmentation threshold (0..255)

USE_PKL = True


def main():
    mimetics_mcq = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq_all_fields.csv"
    mcq_df = pd.read_csv(mimetics_mcq)


    df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
    all_labels = df["name"].tolist()
    print(all_labels)

    # Load CLIP model
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")

    print("Loaded SIGLIP2 model")
    # ---- GPU setup ----
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = "cuda"
    print(f"Using device: {device}")
    model.to(device)
    model.eval()

    mapping = None

    human_total = 0
    total = 0

    for index, row in tqdm(mcq_df.iterrows()):
        full_path = os.path.join(row["full_path"], f"{int(row['num_files'] / 2):06d}.jpg")
        label = row["label"]

        image = Image.open(full_path).convert("RGB")

        inputs = processor(text=all_labels, images=[image], return_tensors="pt",
                            padding="max_length", max_length=64).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
        probs = torch.sigmoid(logits_per_image)  # convert to probabilities

        predicted_label = all_labels[probs.argmax()]
        
        if predicted_label == label:
            human_total += 1

        total += 1


    print("Human total:", human_total)
    print("Total:", total)
    print("Human Accuracy:", human_total/total)

if __name__ == "__main__":
    main()



