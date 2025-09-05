import os
import json
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mix", type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    mix = args.mix

    #Json, which contains the mcq questions
    json_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/original_hat_actionswap/actionswap_mcq_rand{mix}.json"

    rows = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:  # skip blanks
                rows.append(json.loads(line))


    mapping = {}
    df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
    mapping = {row["name"]:row["id"] for _, row in df.iterrows()} #Map the NAME to the ID
    print("mapping", mapping)


    results_json = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/pretrained_slow_only/hat_actionswap_mix{mix}_results.json"
    with open(results_json, "r") as f:
        results = json.load(f)
    
    pred_human = 0
    pred_background = 0
    total = 0

    for index, item in enumerate(results):
        print("Index", index)
        human_path = item["human_path"]
        background_path = item["background_path"]
        predictions = item["predictions"]

        if human_path != rows[index]["human_path"] or background_path != rows[index]["background_path"]:
            print("Not matching")
            break

        choices = rows[index]["choices"] #Get the names of the choices
        id_choices = [mapping[choice] for choice in choices]
        prob_choices = [predictions[i] for i in id_choices] #Get the probabilities for only the 5 choices
        max_val = max(prob_choices)
        max_index = prob_choices.index(max_val)
        chosen_label = choices[max_index]
        human_label = human_path.split("/")[0]
        background_label = background_path.split("/")[0]

        if chosen_label == human_label:
            pred_human += 1
        elif chosen_label == background_label:
            pred_background += 1
        total += 1
    
    print("human acc", pred_human/total)
    print("background error", pred_background/total)
    print("pred human", pred_human)
    print("pred background", pred_background)
    print("total", total)


if __name__ == "__main__":
    main()