#Making the mcq for the full action swap dataset
import pickle
import os
from tqdm.auto import tqdm
import pandas as pd
import csv
import json


PKL_PATH = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/original_hat_actionswap/actionswap_rand_3.pickle"
with open(PKL_PATH, "rb") as f:
    mapping = pickle.load(f)


#JSON File format for saving MCQ:
# human_path: str (eating ice cream/<ID_start_end>)
# background_path: str (eating ice cream/<ID_start_end>)
# choices = []: list of strings

df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
all_labels = df["name"].tolist()

for key, value in tqdm(mapping.items()): #Iterate through the mapping
    human_path = key
    background_path = value[0]

    human_label = human_path.split("/")[0]  # Extract the label from the human path
    background_label = background_path.split("/")[0] # Extract the label from the background path
    
    #Create the choices list
    choices = []
    choices.append(human_label)
    choices.append(background_label)

    #Randomly choose 3 more labels from the all_labels list
    import random
    while len(choices) < 5:
        choice = random.choice(all_labels)
        if choice not in choices:
            choices.append(choice)

    #Shuffle the choices list
    random.shuffle(choices)

    #Create the mcq dict
    mcq = {
        "human_path": human_path,
        "background_path": background_path,
        "choices": choices
    }


    #Save the mcq dict to a json file
    with open("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/original_hat_actionswap/actionswap_mcq_rand3.json", "a") as f:
        json.dump(mcq, f)
        f.write("\n")

