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
import pandas as pd

from dataset_classes.dataset_dual_orig_seg import *
from dataset_classes.datasets_for_slow import *
from dataset_classes.dataset_orig_binmask import *
from sklearn.metrics import classification_report

from custom_models import *

mimetics_mcq_dataset = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq.csv"
actionswap_mcq_dataset = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq.csv"


json_filename = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/class50_to_label.json"

with open(json_filename, "r") as f:
    kinetics_classnames = json.load(f)

# Create an id --> label name mapping
kinetics_id_to_classname = {}
for k, v in kinetics_classnames.items():
    kinetics_id_to_classname[v] = str(k).replace('"', "")

#Create a classname --> id mapping
kinetics_classname_to_id = {v: k for k, v in kinetics_id_to_classname.items()}

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


def test_epoch(model, dataloader, isHATActionSwap, dataset_name):
    model.eval()

    correct_human = []
    correct_bg = []

    count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='test epoch'):
            if CONFIG['model_type'] == 'slow_r50' or CONFIG['model_type'] == 'slow_humanseg':
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

            choices = batch["choices"] # Shape = Tensor of [Batch Size][5] = 20x5

            for i in range (outputs.shape[0]): #Go through each element in the batch
                choices_indices = choices[i] 
                output_probs = outputs[i][choices_indices] #Output probabilities for the 5 choices
                best_choice_pos = torch.argmax(output_probs).item() #position of the highest probability (from 0-4)
                chosen_index = choices_indices[best_choice_pos].item() #index from 0-49
                count += 1
                if isHATActionSwap:
                    label_human = batch["label"][i].item()
                    label_background = batch["label_background"][i].item()
                    correct_human.append(chosen_index == label_human)
                    correct_bg.append(chosen_index == label_background)
                else:
                    label_human = batch["label"][i].item()
                    correct_human.append(chosen_index == label_human)

        print("COUNT = ", count)
        print("Human Accuracy:", np.array(correct_human).mean())
        if isHATActionSwap:
            print("Background Accuracy", np.array(correct_bg).mean())



#Construct the given model
def build_model(model_type: str):
    if model_type == "slow_r50":
        print("Starting creating slowr50 model")
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        model.blocks[-1].proj = nn.Linear(2048, 50)
        return model
    elif model_type == 'slow_humanseg':
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
    elif model_type == "weighted_focus2":
        print("Starting creating weighted focus 2 model")        
        return WeightedFocusNet2() 
    print("Finished building the model")

def build_dataset():
    #CONSTRUCTING TEST DATASET
    if CONFIG['model_type'] == 'slow_r50':
        test_dataset_mimetics = DatasetSlow_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['mimetics_test']),
            col='full_path',
            isHATActionSwap=False,
            max_videos=None
        )
        test_dataset_hat_actionswap = DatasetSlow_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['hat_actionswap_test']),
            col='action_swap_path',
            isHATActionSwap=True,
            max_videos=None
        )
    elif CONFIG['model_type'] == 'slow_humanseg':
        test_dataset_mimetics = DatasetSlow_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['mimetics_test']),
            col='segmented_path',
            isHATActionSwap=False,
            max_videos=None
        )
        test_dataset_hat_actionswap = DatasetSlow_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['hat_actionswap_test']),
            col='segmented_path',
            isHATActionSwap=True,
            max_videos=None
        )
    elif CONFIG['model_type'] == 'sum_concat' or CONFIG['model_type'] == 'stack_concat':
        test_dataset_mimetics = DatasetConcat_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['mimetics_test']),
            col_orig='full_path',
            col_seg='segmented_path',
            isHATActionSwap=False,
            max_videos=None
        )
        test_dataset_hat_actionswap = DatasetConcat_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['hat_actionswap_test']),
            col_orig='action_swap_path',
            col_seg='segmented_path',
            isHATActionSwap=True,
            max_videos=None
        )
    elif CONFIG['model_type'] == 'weighted_focus1' or CONFIG['model_type'] == 'weighted_focus2':
        test_dataset_mimetics = DatasetOrigBinmask_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['mimetics_test']),
            col_orig='full_path',
            col_mask='mask_path',
            isHATActionSwap=False,
            max_videos=None
        )
        test_dataset_hat_actionswap = DatasetOrigBinmask_MCQ(
            csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['hat_actionswap_test']),
            col_orig='action_swap_path',
            col_mask='mask_path',
            isHATActionSwap=True,
            max_videos=None
        )
    return test_dataset_mimetics, test_dataset_hat_actionswap

def load_model_and_datasets():
    # 1. load the model 
    my_model = build_model(CONFIG['model_type'])

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        my_model = nn.DataParallel(my_model)
    
    my_model = my_model.to(CONFIG['device'])

    #If continuing training, then load in the model's weights
    #TODO: Change the "last epoch" to the last epoch you want to load
    last_epoc_saved = CONFIG['last_epoch_saved']
    if last_epoc_saved == "None" or last_epoc_saved is None:
         last_epoc_saved = 0
    else:
        print(f"Loading in epoch {CONFIG['last_epoch_saved']} weights")
        last_epoc_saved = int(CONFIG['last_epoch_saved'])
        checkpoint = torch.load(os.path.join("saved_weights", CONFIG['weights_dir'], f"weights_{last_epoc_saved:06d}.pth"), map_location=CONFIG['device'])
        model_checkpoint = checkpoint['model']
        new_state_dict = {}
        for k, v in model_checkpoint.items():
            new_k = k.replace('module.', '') if k.startswith('module.') else k
            new_state_dict[new_k] = v

        my_model.load_state_dict(new_state_dict)
        # my_model.load_state_dict(model_checkpoint)

    # Create a dataset instance for training set
    test_dataset_mimetics, test_dataset_hat_actionswap = build_dataset()

    test_mimetics_len = len(test_dataset_mimetics)
    test_hat_actionswap_len = len(test_dataset_hat_actionswap)

    # print("Made dataset. Length of minikinetics50 test dataset is ", test_mk50_len)
    print("Made dataset. Length of mimetics test dataset is ", test_mimetics_len)
    print("Made dataset. Length of HAT action swap test dataset is ", test_hat_actionswap_len)

    my_test_mimetics_dataloader = torch.utils.data.DataLoader(test_dataset_mimetics, CONFIG['batch_size'], 
                                                        num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=False, drop_last=False)

    my_test_hat_actionswap_dataloader = torch.utils.data.DataLoader(test_dataset_hat_actionswap, CONFIG['batch_size'], 
                                                        num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=False, drop_last=False)

    return my_model, my_test_mimetics_dataloader, my_test_hat_actionswap_dataloader
    
            
if __name__ == "__main__":
    CONFIG = parse_args()
    print(CONFIG)

    train_step = 0

    #Initiate the Wandb
    run = wandb.init(
        project="Slowfast_Kinetics",
        name=CONFIG['wandb_name'],
        config=CONFIG,
        #mode='online',
        mode='disabled', #TODO: change this
        settings=wandb.Settings(_service_wait=300)
    )

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

    my_model, my_test_mimetics_dataloader, my_test_hat_actionswap_dataloader = load_model_and_datasets()

    
    print("Starting Evaluation Now:")

    print("TESTING MIMETICS")
    test_acc2 = test_epoch(my_model, my_test_mimetics_dataloader, isHATActionSwap=False, dataset_name="mimetics final")


    print("TESTING HAT ACTION SWAP")
    test_acc3 = test_epoch(my_model, my_test_hat_actionswap_dataloader, isHATActionSwap=True, dataset_name="hat action swap final")
