import os
import numpy as np
import torch
import pandas as pd
import csv
from tqdm.auto import tqdm
import json



#json_name = "slow_r50_mimetics final.json"
json_name = "weighted_focus2_mix_hat action swap final.json"

#slow_r50_mix_hat action swap final.json
#slow_humanseg_mix_hat action swap final.json
#sum_concat_mix_hat action swap final.json
#stack_concat_mix_hat action swap final.json
#weighted_focus2_mix_hat action swap final.json

mimetics = False

mimetics_mcq = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq_all_fields.csv"
action_swap_mcq = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"


if mimetics:
    mcq_df = pd.read_csv(mimetics_mcq)
else:
    mcq_df = pd.read_csv(action_swap_mcq)


with open(json_name, 'r') as f:
    data = json.load(f)

human_total = 0
human_total_mcq = 0
background_total = 0
background_total_mcq = 0

for index in tqdm(range(1, len(data)+1)):
    gt_label = data[f"row_{index:06d}"]["gt_label"]
    gt_label_name = data[f"row_{index:06d}"]["gt_label_name"]
    gt_label_background = data[f"row_{index:06d}"]["gt_background_label"]
    gt_label_name_background = data[f"row_{index:06d}"]["gt_label_name_background"]

    predicted_probs = data[f"row_{index:06d}"]["predicted_probs"]
    
    # print("gt_label", gt_label)
    # print("gt_label_name", gt_label_name)
    # print("predicted probs", predicted_probs)

    best_action = max(predicted_probs, key=predicted_probs.get)
    # print("Best action", best_action)

    if gt_label_name == best_action:
        human_total += 1
    if gt_label_name_background == best_action:
        background_total += 1

    csv_row_num = index - 1
    five_choices = []

    #append the choices to choices
    for choice in [1, 2, 3, 4, 5]:
        five_choices.append(mcq_df.iloc[csv_row_num][f"choice_{choice}"])

    # Use a dictionary comprehension to filter
    filtered = {k: predicted_probs[k] for k in five_choices}

    # Get the key with the highest value from the filtered subset
    best_action_mcq = max(filtered, key=filtered.get)

    if best_action_mcq == gt_label_name:
        human_total_mcq += 1
    elif best_action_mcq == gt_label_name_background:
        background_total_mcq += 1

    # print("Choices", five_choices)
    # print("Subset", filtered)
    # print("Correct label from mcq", five_choices[mcq_df.iloc[csv_row_num]["human_choice"] - 1])

    if gt_label_name != five_choices[mcq_df.iloc[csv_row_num]["human_choice"] - 1]:
        print("NOT OK")

    # print("Best action in subset:", best_action)


print("Human Total", human_total)
print("Human Total MCQ", human_total_mcq)
print("Human Accuracy", float(human_total/len(mcq_df)))
print("Human Accuracy MCQ", float(human_total_mcq/len(mcq_df)))


print("Background Total", background_total)
print("Background Total MCQ", background_total_mcq)
print("Background Accuracy", float(background_total/len(mcq_df)))
print("Background Accuracy MCQ", float(background_total_mcq/len(mcq_df)))

