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

from slowfast_kinetics.dataset_classes.dataset_dual_orig_seg import DatasetConcat
from slowfast_kinetics.dataset_classes.datasets_for_slow import DatasetSlow

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


def train_epoch(model, epoch, optimizer, loss_fn, dataloader):

    model.train()
    global train_step
    total_train_loss = [] #Contains a list of the train loss for each batch in the epoch
    total_correct = [] #contains a list of the # of correctly classified samples for each batch in the epoch

    starttime = time.time()
    for batch in tqdm(dataloader, desc=f"Train epoch {epoch}"):
            # print(f" Starting batch {batch}")
            # batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                 
                        
            inputs_orig = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][0]])
            inputs_seg = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][0]])

            labels = batch["label"].to(CONFIG['device'])

            optimizer.zero_grad()
            outputs = model(inputs_orig, inputs_seg)
            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()

            predicted = torch.argmax(outputs, dim=1)
            correct = (predicted == labels).detach().cpu().numpy() #add the number of correct predictions in this batch
            accuracy = correct.mean()
            current_lr = optimizer.param_groups[0]['lr']

            #Log performance every 50 batch
            run.log({
                    "train loss": loss, 
                    "train accuracy": accuracy,
                    "global_step": train_step,
                    "step_time": time.time() - starttime,
                    "learning rate": current_lr
            })
            starttime = time.time()
            train_step += 1

            total_train_loss.append(train_loss)
            total_correct += correct.tolist()
            torch.cuda.empty_cache()

    current_lr = optimizer.param_groups[0]['lr']   

    print(f"Train Loss (epoch avg): {np.array(total_train_loss).mean()}, Train Accuracy (epoch avg): {np.array(total_correct).mean()}")

    run.log({
        "train loss (epoch avg)": np.array(total_train_loss).mean(),
        "train accuracy (epoch avg)": np.array(total_correct).mean(),
        "global_step": train_step,
        "epoch": epoch,
        "learning rate": current_lr
    })
                

def test_epoch(model, epoch, loss_fn, dataloader):

    model.eval()
    global train_step
    test_loss = []
    correct = []

    with torch.no_grad():
        #for batch in tqdm(dataloader, desc=f"Testing Epoch {epoch}", disable=(epoch != 0)):   
        for batch in tqdm(dataloader, desc='test epoch'):

                inputs_orig = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][0]])
                inputs_seg = torch.stack([x.to(CONFIG['device']) for x in batch["inputs"][1]])                             
                labels = batch["label"].to(CONFIG['device'])

                outputs = model(inputs_orig, inputs_seg)
                loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
                
                test_loss.append(loss.item())

                predicted = torch.argmax(outputs, dim=1)
                correct += ((predicted == labels).detach().cpu().numpy().tolist()) #add the number of correct predictions in this batch
            

    run.log({
        "test loss (epoch avg)": np.array(test_loss).mean(),
        "test accuracy (epoch avg)": np.array(correct).mean(),
        "global_step": train_step,
        "epoch": epoch
    })

    print(f"Test Loss (epoch avg): {np.array(test_loss).mean()}, Test Accuracy (epoch avg): {np.array(correct).mean()}")
    return np.array(correct).mean()


#Construct the given model
def build_model(model_type: str):
    if model_type == "slow_r50":
        print("Starting creating slowr50 model")
        model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        model.blocks[-1].proj = nn.Linear(2048, 50)
        return model
    elif model_type == "sum_concat":
        print("Starting creating sum_concat model")
        slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
        orig_stem = copy.deepcopy(slow_model.blocks[0])
        orig_layer1 = copy.deepcopy(slow_model.blocks[1])
        orig_layer2 = copy.deepcopy(slow_model.blocks[2])

        seg_stem = copy.deepcopy(slow_model.blocks[0])
        seg_layer1 = copy.deepcopy(slow_model.blocks[1])
        seg_layer2 = copy.deepcopy(slow_model.blocks[2])

        layer3 = slow_model.blocks[3]
        layer4 = slow_model.blocks[4]
        layer5 = slow_model.blocks[5]
        layer5.proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)

        return SumConcat(orig_stem, orig_layer1, orig_layer2,
                            seg_stem, seg_layer1, seg_layer2, 
                            layer3, layer4, layer5)
    
    elif model_type == "stack_concat":
        print("Starting creating stack_concat model")
        slow_model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)

        orig_stem = copy.deepcopy(slow_model.blocks[0])
        orig_layer1 = copy.deepcopy(slow_model.blocks[1])
        orig_layer2 = copy.deepcopy(slow_model.blocks[2])

        seg_stem = copy.deepcopy(slow_model.blocks[0])
        seg_layer1 = copy.deepcopy(slow_model.blocks[1])
        seg_layer2 = copy.deepcopy(slow_model.blocks[2])

        layer3 = slow_model.blocks[3]

        #change input channels of these from 512 to 1024 bc we stacked
        layer3.res_blocks[0].branch1_conv = nn.Conv3d(1024, 1024, kernel_size=(1, 1, 1), stride=(1, 2, 2), bias=False)
        layer3.res_blocks[0].branch2.conv_a = nn.Conv3d(1024, 256, kernel_size=(3, 1, 1), 
                                                                        stride=(1, 1, 1), padding=(1, 0, 0), bias=False)

        layer4 = slow_model.blocks[4]
        layer5 = slow_model.blocks[5]
        layer5.proj = torch.nn.Linear(in_features=2048, out_features=50, bias=True)

        return StackConcat(orig_stem, orig_layer1, orig_layer2,
                            seg_stem, seg_layer1, seg_layer2, 
                            layer3, layer4, layer5)     
             
    print("Finished building the model")

def build_dataset():
    #TRAIN DATASET
    if CONFIG['dataset_type_train'] == 'segmented_minikinetics50_train':
        train_dataset = DatasetSlow(
            csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_minikinetics50_train']),
            max_videos=None
        )

    elif CONFIG['dataset_type_train'] == 'original_minikinetics50_train':
        train_dataset = DatasetSlow(
            csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['original_minikinetics50_train']),
            max_videos=None
        )
        validation_dataset = DatasetSlow(
            csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['original_minikinetics50_val']),
            max_videos=None
        )

    elif CONFIG['dataset_type_train'] == 'dual_original_and_segmented_minikinetics50_train':
        train_dataset = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['dual_original_and_segmented_minikinetics50_train']),
            max_videos=None
        )
        validation_dataset = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['dual_original_and_segmented_minikinetics50_val']),
            max_videos=None
        )
    else:
        raise ValueError(f"Unknown dataset type for train: {CONFIG['dataset_type_train']}")
    
    #Validation dataset
    if CONFIG['dataset_type_val'] == 'segmented_minikinetics50_val':
        validation_dataset = DatasetSlow(
            csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_minikinetics50_val']),
            max_videos=None
        )
    elif CONFIG['dataset_type_val'] == 'original_minikinetics50_val':
        validation_dataset = DatasetSlow(
            csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['original_minikinetics50_val']),
            max_videos=None
        )
    elif CONFIG['dataset_type_val'] == 'dual_original_and_segmented_minikinetics50_val':
        validation_dataset = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['dual_original_and_segmented_minikinetics50_val']),
            max_videos=None
        )
    else:
        raise ValueError(f"Unknown dataset type for validation: {CONFIG['dataset_type_val']}")
    
    return train_dataset, validation_dataset

def train_model():
    # 1. load the model 

    my_model = build_model(CONFIG['model_type'])
    my_model = my_model.to(CONFIG['device'])

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        my_model = nn.DataParallel(my_model)

    #If continuing training, then load in the model's weights
    #TODO: Change the "last epoch" to the last epoch you want to load

    last_epoc_saved = CONFIG['last_epoch_saved']
    if last_epoc_saved == "None" or last_epoc_saved is None:
         last_epoc_saved = 0
    else:
        print(f"Loading in epoch {CONFIG['last_epoch_saved']} weights")
        last_epoc_saved = int(CONFIG['last_epoch_saved'])
        checkpoint = torch.load(f"saved_weights/slowfast_minikinetics50/slow_baseline/weights_{last_epoc_saved:06d}.pth", map_location=CONFIG['device'])
        my_model.load_state_dict(checkpoint['model'])
                  
    # Create a dataset instance for training set
    train_dataset, validation_dataset = build_dataset()
    
    train_len = len(train_dataset)
    validation_len = len(validation_dataset)

    print("Made dataset. Length of training dataset is ", train_len)
    print("Made dataset. Length of validation dataset is ", validation_len)


    my_train_dataloader = torch.utils.data.DataLoader(train_dataset, CONFIG['batch_size'], 
                                                    num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=True, drop_last=True)

    my_validation_dataloader = torch.utils.data.DataLoader(validation_dataset, CONFIG['batch_size'], 
                                                        num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=False, drop_last=False)



    print("Made dataloaders")

    my_optimizer = optim.SGD(my_model.parameters(), lr=CONFIG['learning_rate'])
    my_scheduler = ReduceLROnPlateau(my_optimizer, mode='max', patience=40, threshold=1e-2)
    my_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    print("initialized optimizer, scheduler, and loss function")

    test_acc = test_epoch(my_model, -1, my_loss_fn, my_validation_dataloader)
    print("Test Accuracy before training=", test_acc)

    for epoch in tqdm(range(last_epoc_saved + 1, last_epoc_saved + 1 + CONFIG['num_epochs']), desc='Training Epochs'):
        print("Starting epoch", epoch)
        my_model.train()
        train_epoch(my_model, epoch, my_optimizer, my_loss_fn, my_train_dataloader)

        test_acc = test_epoch(my_model, epoch, my_loss_fn, my_validation_dataloader)
        my_scheduler.step(test_acc)

        print("Finish epoch ", epoch, "training. Starting validation")

        
        file_path = os.path.join(CONFIG['metadata_dir'], "saved_weights", CONFIG['weights_dir'], f"weights_{epoch:06d}.pth")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({"model":my_model.state_dict(),
                    "optimizer": my_optimizer.state_dict()}, file_path)

    my_model.eval()
    test_epoch(my_model, epoch, my_loss_fn, my_validation_dataloader)
    
            
if __name__ == "__main__":
    CONFIG = parse_args()
    print(CONFIG)

    train_step = 0

    #Initiate the Wandb
    run = wandb.init(
        project="Slowfast_Kinetics",
        name=CONFIG['wandb_name'],
        config=CONFIG,
        mode=CONFIG['wandb_mode'] if 'wandb_mode' in CONFIG else None,
    )

    run.define_metric("learning rate")
    run.define_metric("train loss", step_metric="train_step")
    run.define_metric("train accuracy", step_metric="train_step")

    run.define_metric("train loss (epoch avg)", step_metric="epoch")
    run.define_metric("train accuracy (epoch avg)", step_metric="epoch")
    run.define_metric("test loss (epoch avg)", step_metric="epoch")
    run.define_metric("test accuracy (epoch avg)", step_metric="epoch")

    train_model()


