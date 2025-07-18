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


def train_epoch(model, epoch, optimizer, loss_fn, dataloader):
    model.train()
    global train_step
    total_train_loss = [] #Contains a list of the train loss for each batch in the epoch
    total_correct = [] #contains a list of the # of correctly classified samples for each batch in the epoch

    # total_predicted = [] #contains a list of the predicted labels for each sample in the epoch
    # total_labels = [] #contains a list of the true labels for each sample in the epoch

    starttime = time.time()
    for batch in tqdm(dataloader, desc=f"Train epoch {epoch}"):
            # print(f" Starting batch {batch}")
            # batch_size = batch["inputs"][0].shape[0]  # Number of samples in the batch                 

            optimizer.zero_grad()

            #Depending on the model type, we will have different inputs
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


            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            loss.backward()
            optimizer.step()
            
            train_loss = loss.item()

            predicted = torch.argmax(outputs, dim=1)

            correct = (predicted == labels).detach().cpu().numpy() #add the number of correct predictions in this batch

            accuracy = correct.mean()
            current_lr = optimizer.param_groups[0]['lr']

            #Log performance every batch
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
            # total_predicted += predicted.detach().cpu().numpy().tolist()
            # total_labels += labels.detach().cpu().numpy().tolist()

            torch.cuda.empty_cache()

    current_lr = optimizer.param_groups[0]['lr']   

    print(f"Train Loss (epoch avg): {np.array(total_train_loss).mean()}, Train Accuracy (epoch avg): {np.array(total_correct).mean()}")
    # report = classification_report(y_true=total_labels, y_pred=total_predicted, labels=list(range(50)), target_names=target_names, zero_division=0)
    # print("Report:\n", report)

    if CONFIG['model_type'] == 'weighted_focus1' or CONFIG['model_type'] == 'weighted_focus2':
        print(f"Alpha: {alpha.mean().item()}, Beta: {beta.mean().item()}")

    run.log({
        "train loss (epoch avg)": np.array(total_train_loss).mean(),
        "train accuracy (epoch avg)": np.array(total_correct).mean(),
        "global_step": train_step,
        "epoch": epoch,
        "learning rate": current_lr
    })
                

def test_epoch(model, epoch, loss_fn, dataloader, which_set):
    model.eval()
    global train_step
    test_loss = []
    correct = []

    # total_predicted = [] #contains a list of the predicted labels for each sample in the epoch
    # total_labels = [] #contains a list of the true labels for each sample in the epoch

    with torch.no_grad():
        #for batch in tqdm(dataloader, desc=f"Testing Epoch {epoch}", disable=(epoch != 0)):   
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

                #print("Output shape:", outputs.shape)
            else:
                raise ValueError(f"Unknown model type: {CONFIG['model_type']}")
                                      
            labels = batch["label"].to(CONFIG['device'])

            loss = loss_fn(outputs, labels) #the avg of the losses for the samples in the batch
            test_loss.append(loss.item())

            predicted = torch.argmax(outputs, dim=1)
            correct += ((predicted == labels).detach().cpu().numpy().tolist()) #add the number of correct predictions in this batch
            
            # total_predicted += predicted.detach().cpu().numpy().tolist()
            # total_labels += labels.detach().cpu().numpy().tolist()
            
    if which_set == 'places365_action_swap':
        run.log({
            "Action Swap Test Loss (epoch avg)": np.array(test_loss).mean(),
            "Action Swap Test Accuracy (epoch avg)": np.array(correct).mean(),
            "global_step": train_step,
            "epoch": epoch
        })
        print("Action Swap Test Loss (epoch avg): ", np.array(test_loss).mean())
        print("Action Swap Test Accuracy (epoch avg): ", np.array(correct).mean())

    elif which_set == 'minikinetics50':
        run.log({
            "MK50 test loss (epoch avg)": np.array(test_loss).mean(),
            "MK50 test accuracy (epoch avg)": np.array(correct).mean(),
            "global_step": train_step,
            "epoch": epoch
        })
        print("MK50 Test Loss (epoch avg): ", np.array(test_loss).mean())
        print("MK50 Test Accuracy (epoch avg): ", np.array(correct).mean())

    # report = classification_report(y_true=total_labels, y_pred=total_predicted, labels=list(range(50)), target_names=target_names, zero_division=0)
    # print("Report:\n", report)

    if CONFIG['model_type'] == 'weighted_focus1' or CONFIG['model_type'] == 'weighted_focus2':
        print(f"Alpha: {alpha.mean().item()}, Beta: {beta.mean().item()}")

    torch.cuda.empty_cache()

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

    elif CONFIG['dataset_type_train'] == 'dual_original_and_segmented_minikinetics50_train':
        train_dataset = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_minikinetics50_train']),
            max_videos=None
        )
    elif CONFIG['dataset_type_train'] == 'original_and_binmask_minikinetics50_train':
        train_dataset = DatasetOrigBinmask (
            seg_csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_minikinetics50_train']),
            max_videos=None
        )

    ###### FOR TRAINING ON THE PLACES365 ACTION SWAP MIXED WITH MINIKINETICS ############
    elif CONFIG['dataset_type_train'] == 'original_mix_train': #Mix (Places365Bg+MKforeground) & Minikinetics
        train_dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], "dataset/places365/mix_train.csv"),
            col='full_path',
            max_videos=None
        )
    elif CONFIG['dataset_type_train'] == 'segmented_mix_train':
        train_dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], "dataset/places365/mix_train.csv"),
            col='segmented_path',
            max_videos=None
        )
    elif CONFIG['dataset_type_train'] == 'dual_original_and_segmented_mix_train':
        train_dataset = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], "dataset/places365/mix_train.csv"),
            max_videos=None
        )
    elif CONFIG['dataset_type_train'] == 'dual_original_and_binmask_mix_train':
        train_dataset = DatasetOrigBinmask(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], "dataset/places365/mix_train.csv"),
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
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_minikinetics50_val']),
            max_videos=None
        )
    elif CONFIG['dataset_type_val'] == 'original_and_binmask_minikinetics50_val':
        validation_dataset = DatasetOrigBinmask (
            seg_csv_path = os.path.join(CONFIG['metadata_dir'], CONFIG['segmented_minikinetics50_val']),
            max_videos=None
        )

    ###### FOR VALIDATION ON THE PLACES365 ACTION SWAP MIXED WITH MINIKINETICS ############
    elif CONFIG['dataset_type_val'] == 'original_mix_val': #Mix (Places365Bg+MKforeground) & Minikinetics
        validation_dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], "dataset/places365/places365_actionswap_val.csv"),
            col='full_path',
            max_videos=None
        )

        validation_dataset2 = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], "dataset/minikinetics50/minikinetics50_validation_all.csv"),
            col='full_path',
            max_videos=None
        )
    elif CONFIG['dataset_type_val'] == 'segmented_mix_val':
        validation_dataset = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], "dataset/places365/places365_actionswap_val.csv"),
            col='segmented_path',
            max_videos=None
        )
        validation_dataset2 = DatasetSlow(
            csv_path = os.path.join(CONFIG['metadata_dir'], "dataset/minikinetics50/minikinetics50_validation_all.csv"),
            col='segmented_path',
            max_videos=None
        )
    elif CONFIG['dataset_type_val'] == 'dual_original_and_segmented_mix_val':
        validation_dataset = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], "dataset/places365/places365_actionswap_val.csv"),
            max_videos=None
        )

        validation_dataset2 = DatasetConcat(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], "dataset/minikinetics50/minikinetics50_validation_all.csv"),
            max_videos=None
        )
    elif CONFIG['dataset_type_val'] == 'dual_original_and_binmask_mix_val':
        validation_dataset = DatasetOrigBinmask(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], "dataset/places365/places365_actionswap_val.csv"),
            max_videos=None
        )

        validation_dataset2 = DatasetOrigBinmask(
            seg_csv_path=os.path.join(CONFIG['metadata_dir'], "dataset/minikinetics50/minikinetics50_validation_all.csv"),
            max_videos=None
        )
    else:
        raise ValueError(f"Unknown dataset type for validation: {CONFIG['dataset_type_val']}")
    
    return train_dataset, validation_dataset, validation_dataset2

def train_model():
    # 1. load the model 
    my_model = build_model(CONFIG['model_type'])

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        my_model = nn.DataParallel(my_model)
    
    my_model = my_model.to(CONFIG['device'])
    my_optimizer = optim.Adam(my_model.parameters(), lr=CONFIG['learning_rate'])
    my_scheduler = ReduceLROnPlateau(my_optimizer, mode='max', patience=40, threshold=1e-2)

    #If continuing training, then load in the model's weights
    #TODO: Change the "last epoch" to the last epoch you want to load

    last_epoc_saved = CONFIG['last_epoch_saved']
    if last_epoc_saved == "None" or last_epoc_saved is None:
         last_epoc_saved = 0
    else:
        print(f"Loading in epoch {CONFIG['last_epoch_saved']} weights")
        last_epoc_saved = int(CONFIG['last_epoch_saved'])
        checkpoint = torch.load(os.path.join("saved_weights", CONFIG['weights_dir'], f"weights_{last_epoc_saved:06d}.pth"), map_location=CONFIG['device'])
        my_model.load_state_dict(checkpoint['model'])
        my_optimizer.load_state_dict(checkpoint['optimizer'])


    # print("\nTrainable Parameters:")
    # for name, param in my_model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.shape}")

    # Create a dataset instance for training set
    train_dataset, validation_dataset_places365_actionswap, validation_dataset_mk = build_dataset()

    train_len = len(train_dataset)
    validation_len_swap = len(validation_dataset_places365_actionswap)
    validation_len_mk = len(validation_dataset_mk)

    print("Made dataset. Length of training dataset is ", train_len)
    print("Made dataset. Length of validation dataset places365 action swap is ", validation_len_swap)
    print("Made dataset. Length of validation dataset minikinetics ", validation_len_mk)



    my_train_dataloader = torch.utils.data.DataLoader(train_dataset, CONFIG['batch_size'], 
                                                    num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=True, drop_last=True)

    my_validation_swap_dataloader = torch.utils.data.DataLoader(validation_dataset_places365_actionswap, CONFIG['batch_size'], 
                                                        num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=False, drop_last=False)
    
    my_validation_mk_dataloader = torch.utils.data.DataLoader(validation_dataset_mk, CONFIG['batch_size'], 
                                                        num_workers=CONFIG['num_dataloader_workers'], pin_memory=False, shuffle=False, drop_last=False)

    print("Made dataloaders")

    my_loss_fn = nn.CrossEntropyLoss(reduction='mean')

    print("initialized optimizer, scheduler, and loss function")

    # TODO: uncomment this
    test_acc = test_epoch(my_model, -1, my_loss_fn, my_validation_swap_dataloader, which_set='places365_action_swap')
    print("Places365 Action Swap Test Accuracy before training=", test_acc)

    test_acc = test_epoch(my_model, -1, my_loss_fn, my_validation_mk_dataloader, which_set='minikinetics50')
    print("MiniKinetics50 Test Accuracy before training=", test_acc)


    for epoch in tqdm(range(last_epoc_saved + 1, last_epoc_saved + 1 + CONFIG['num_epochs']), desc='Training Epochs'):
        print("Starting epoch", epoch)
        my_model.train()
        train_epoch(my_model, epoch, my_optimizer, my_loss_fn, my_train_dataloader)

        swap_test_acc = test_epoch(my_model, epoch, my_loss_fn, my_validation_swap_dataloader, which_set='places365_action_swap')
        mk_test_acc = test_epoch(my_model, epoch, my_loss_fn, my_validation_mk_dataloader, which_set='minikinetics50')

        my_scheduler.step((swap_test_acc + mk_test_acc)/2)

        print("Finish epoch ", epoch, "training. Starting validation")

        
        file_path = os.path.join(CONFIG['metadata_dir'], "saved_weights", CONFIG['weights_dir'], f"weights_{epoch:06d}.pth")
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        torch.save({"model":my_model.state_dict(),
                    "optimizer": my_optimizer.state_dict(), "scheduler": my_scheduler.state_dict()}, file_path)

    my_model.eval()
    test_acc = test_epoch(my_model, epoch, my_loss_fn, my_validation_swap_dataloader, which_set='places365_action_swap')
    test_acc = test_epoch(my_model, epoch, my_loss_fn, my_validation_mk_dataloader, which_set='minikinetics50')
    
            
if __name__ == "__main__":
    CONFIG = parse_args()
    print(CONFIG)

    train_step = 0

    #Initiate the Wandb
    run = wandb.init(
        project="Slowfast_Kinetics",
        name=CONFIG['wandb_name'],
        config=CONFIG,
        mode='online',
        settings=wandb.Settings(_service_wait=300)
        # mode='disabled'
    )

    run.define_metric("learning rate")
    run.define_metric("train loss", step_metric="train_step")
    run.define_metric("train accuracy", step_metric="train_step")

    run.define_metric("train loss (epoch avg)", step_metric="epoch")
    run.define_metric("train accuracy (epoch avg)", step_metric="epoch")
    # run.define_metric("test loss (epoch avg)", step_metric="epoch")
    # run.define_metric("test accuracy (epoch avg)", step_metric="epoch")
    run.define_metric("Action Swap Test Loss (epoch avg)", step_metric="epoch")
    run.define_metric("Action Swap Test Accuracy (epoch avg)", step_metric="epoch")
    run.define_metric("MK50 test loss (epoch avg)", step_metric="epoch")
    run.define_metric("MK50 test accuracy (epoch avg)", step_metric="epoch")

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

    train_model()


