#Testing on anything! using pre-trained SlowFast
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
import argparse

from typing import Dict
import json
import urllib

import torch
import yaml

from dataset import KineticsDataset2
from dataset_mimetics import MimeticsDataset

print("Starting evaluation script")
#Create an argument parser to allow for a dynamic batch size
parser = argparse.ArgumentParser(description="Training script with tunable batch size")
parser.add_argument(
    "--config",
    type=str,
    help="configfile"
)
args = parser.parse_args()

with open(f"config/{args.config}", "r") as f:
    CONFIG = yaml.safe_load(f)

print("Loaded yaml file")

print("Using config file: ", f"config/{args.config}")
print("Config loaded: ", CONFIG)

CONFIG['device'] = "cuda" if torch.cuda.is_available() else "cpu"

# 1. load the model 
my_model = torch.hub.load('facebookresearch/pytorchvideo', 'slowfast_r50', pretrained=True)
# 2. Move to device
my_model = my_model.to(CONFIG['device'])
# 3. Wrap with DataParallel if multiple GPUs
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    my_model = nn.DataParallel(my_model)


#Initiate the Wandb
run = wandb.init(
    project="Slowfast_Evaluation_on_Mimetics",
)

run.define_metric("test loss (epoch avg)")
run.define_metric("test accuracy (epoch avg)")


wandb.watch(my_model, log='all', log_freq = 100)

print(torch.version.cuda)  # Should print a CUDA version, not None
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print("Using device:", CONFIG['device'])
print("Started making dataset")

#create a dataset instance for validation set
validation_dataset = MimeticsDataset(
     csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['val_csv']),
     max_videos=None
)

validation_len = len(validation_dataset)

print("Made dataset. Length of validation dataset is ", validation_len)

my_validation_dataloader = torch.utils.data.DataLoader(validation_dataset, CONFIG['batch_size'], 
                                                       num_workers=CONFIG['num_dataloader_workers'], pin_memory=True, shuffle=False)

print("Made dataloaders")

my_loss_fn = nn.CrossEntropyLoss(reduction='mean')         

def evaluation(model, loss_fn, dataloader):
    model.eval()
    test_loss = []
    correct = []

    for i, batch in enumerate(dataloader):
            batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                                         
            inputs = [x.to(CONFIG['device']) for x in batch["inputs"]]
            labels = batch["label"].to(CONFIG['device'])

            print("         Starting batch ", i, " with batch size ", batch_size)
            outputs = model(inputs)
            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            
            test_loss.append(loss.item())

            predicted = torch.argmax(outputs, dim=1)
            correct += ((predicted == labels).detach().cpu().numpy().tolist()) #add the number of correct predictions in this batch
            
    avg_loss = np.array(test_loss).mean()
    avg_accuracy = np.array(correct).mean()

    print("Average Loss=", avg_loss, "Average Accuracy=", avg_accuracy)

    run.log({
        "test loss (epoch avg)": avg_loss,
        "test accuracy (epoch avg)": avg_accuracy,
    })
            

if __name__ == "__main__":
    evaluation(my_model, my_loss_fn, my_validation_dataloader)


