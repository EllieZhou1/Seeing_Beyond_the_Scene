import torch
from datetime import datetime
import os
from pytorchvideo.data.encoded_video import EncodedVideo
import csv
import numpy as np
from PIL import Image
import copy

import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
import time

from dataset_classes.dataset_dual_orig_seg import DatasetConcat
from dataset_classes.datasets_for_slow import DatasetSlow
from dataset_classes.dataset_orig_binmask import DatasetOrigBinmask
from sklearn.metrics import classification_report

from custom_models import *

def parse_args():
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

    print("Using config file: ", f"config/{args.config}")
    print("Config loaded: ", CONFIG)

    CONFIG['device'] = "cuda" if torch.cuda.is_available() else "cpu"
    return CONFIG
                

def test_epoch(model, loss_fn, dataloader):
    model.eval()
    test_loss = []
    correct = []

    all_outputs = []
    all_labels = []
    top5_acc_total = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='test epoch'):
            if CONFIG['model_type'] == 'slow_r50':
                inputs = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"]])
                outputs = model(inputs)
            elif CONFIG['model_type'] == 'sum_concat' or CONFIG['model_type'] == 'stack_concat':           
                inputs_orig = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][0]])
                inputs_seg = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][1]])
                outputs = model(inputs_orig, inputs_seg)
            elif CONFIG['model_type'] == 'weighted_focus1' or CONFIG['model_type'] == 'weighted_focus2':
                inputs_orig = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][0]])
                inputs_mask = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][1]])
                alpha, beta, outputs = model(inputs_orig, inputs_mask)
            else:
                raise ValueError(f"Unknown model type: {CONFIG['model_type']}")
                                      
            labels = batch["label"].to(CONFIG['device'])

            # Top-5 Accuracy
            top5_preds = torch.topk(outputs, k=5, dim=1).indices
            top5_correct = top5_preds.eq(labels.view(-1, 1)).any(dim=1).float()
            top5_acc_total += top5_correct.detach().cpu().numpy().tolist()

            # Collect outputs and labels for mAP
            all_outputs.append(outputs.detach().cpu())
            all_labels.append(labels.detach().cpu())

            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            test_loss.append(loss.item())

            predicted = torch.argmax(outputs, dim=1)
            print("predicted", predicted)
            print("labels", labels)
            correct += ((predicted == labels).detach().cpu().numpy().tolist()) #add the number of correct predictions in this batch

    top5_acc = np.mean(top5_acc_total)
    print(f"Test Loss: {np.array(test_loss).mean()}, Top 1 Accuracy: {np.array(correct).mean()}, Top 5 Accuracy: {top5_acc}")

    # Compute mAP
    from sklearn.metrics import average_precision_score
    from sklearn.preprocessing import label_binarize

    all_outputs = torch.cat(all_outputs, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_labels_bin = label_binarize(all_labels, classes=np.arange(all_outputs.shape[1]))
    map_score = average_precision_score(all_labels_bin, all_outputs, average="macro")
    print(f"mean Average Precision (mAP): {map_score}")

    if CONFIG['model_type'] == 'weighted_focus1' or CONFIG['model_type'] == 'weighted_focus2':
        print(f"Alpha: {alpha.mean().item()}, Beta: {beta.mean().item()}")

    torch.cuda.empty_cache()
    return {
        "top1_acc": np.array(correct).mean(),
        "top5_acc": top5_acc,
        "mAP": map_score
    }


#Construct the given model
def build_model(model_type: str):
    if model_type == "slow_r50":
        print("Starting creating slowr50 model")
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        model.blocks[-1].proj = nn.Linear(2048, 50)
        return model
    elif model_type == "sum_concat":
        print("Starting creating sum_concat model")
        return SumConcat()
    elif model_type == "stack_concat":
        print("Starting creating stack_concat model")
        return StackConcat()     
    elif model_type == "weighted_focus1":
        print("Starting creating weighted focus 1 model")
        return WeightedFocusNet1()             
    elif model_type == 'weighted_focus2':
        print("Starting creating weighted focus 2 model")
        return WeightedFocusNet2()
    print("Finished building the model")

def build_dataset():
    if CONFIG['test']['dataset_type'] == 'segmented_mimetics':
        dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_mimetics']),
            col='segmented_path',
            max_videos=None
        )
    elif CONFIG['test']['dataset_type'] == 'dual_mimetics_and_segmented_mimetics':
        dataset = DatasetConcat(
            seg_csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_mimetics']),
            max_videos=None
        )
    elif CONFIG['test']['dataset_type'] == 'dual_mimetics_and_binarymask_mimetics':
        dataset = DatasetOrigBinmask(
            seg_csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_mimetics']),
            max_videos=None
        )
    elif CONFIG['test']['dataset_type'] == 'HAT_action_swap':
        dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['HAT_action_swap']),
            col='action_swap_path',
            max_videos=None
        )
    elif CONFIG['test']['dataset_type'] == 'segmented_HAT_action_swap':
        dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['HAT_action_swap']),
            col='segmented_path',
            max_videos=None
        )
    elif CONFIG['test']['dataset_type'] == 'dual_HAT_and_segmented_HAT':
        dataset = DatasetConcat(
            seg_csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['HAT_action_swap']),
            col_orig='action_swap_path',
            col_seg='segmented_path',
            max_videos=None
        )
    elif CONFIG['test']['dataset_type'] == 'dual_HAT_and_binarymask_HAT':
        dataset = DatasetOrigBinmask(
            seg_csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['HAT_action_swap']),
            col_orig='action_swap_path',
            col_mask='mask_path',
            max_videos=None
        )
    else:
        raise ValueError(f"Unknown dataset type for train: {CONFIG['test']['dataset_type']}")

    return dataset

def test_model():
    # 1. load the model 
    my_model = build_model(CONFIG['model_type'])
    my_model = my_model.to(CONFIG['device'])

    print(f"Loading in epoch {CONFIG['last_epoch_saved']} weights")
    last_epoch_saved = int(CONFIG['last_epoch_saved'])
    checkpoint = torch.load(os.path.join(CONFIG['weights_dir'], f"weights_{last_epoch_saved:06d}.pth"), map_location=CONFIG['device'])
    
    state_dict = checkpoint['model']

    if list(state_dict.keys())[0].startswith('module.'):
        # Remove 'module.' prefix
        new_state_dict = {k[7:]: v for k, v in state_dict.items()}
        my_model.load_state_dict(new_state_dict)
    else:
        my_model.load_state_dict(state_dict)

    # Create a dataset instance for training set
    my_dataset = build_dataset()

    len_dataset = len(my_dataset)
    print("Made dataset. Length of dataset is ", len_dataset)

    my_dataloader = torch.utils.data.DataLoader(my_dataset, CONFIG['batch_size'], 
                    num_workers=20, pin_memory=False, shuffle=False, drop_last=False)

    print("Made dataloaders")

    my_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    my_model.eval()
    test_epoch(my_model, my_loss_fn, my_dataloader)
    
            
if __name__ == "__main__":
    CONFIG = parse_args()
    print(CONFIG)

    target_names = ['playing guitar', 'bowling', 'playing saxophone', 'brushing teeth', 
                    'playing basketball', 'tying tie', 'skiing slalom', 'brushing hair', 
                    'punching person (boxing)', 'playing accordion', 'archery', 
                    'catching or throwing frisbee', 'drinking', 'reading book', 
                    'eating ice cream', 'flying kite', 'sweeping floor', 
                    'walking the dog', 'skipping rope', 'clean and jerk', 
                    'eating cake', 'catching or throwing baseball', 
                    'skiing (not slalom or crosscountry)', 'juggling soccer ball', 
                    'deadlifting', 'driving car', 'cleaning windows', 'shooting basketball', 
                    'canoeing or kayaking', 'surfing water', 'playing volleyball', 'opening bottle', 
                    'playing piano', 'writing', 'dribbling basketball', 'reading newspaper', 'playing violin', 
                    'juggling balls', 'playing trumpet', 'smoking', 'shooting goal (soccer)', 'hitting baseball', 
                    'sword fighting', 'climbing ladder', 'playing bass guitar', 'playing tennis', 'climbing a rope', 
                    'golf driving', 'hurdling', 'dunking basketball']

    test_model()
