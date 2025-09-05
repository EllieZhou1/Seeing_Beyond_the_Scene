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
    csv_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq.csv"
    mcq_df = pd.read_csv(csv_path)


    mapping = {}
    df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
    mapping = {row["name"]:row["id"] for _, row in df.iterrows()} #Map the NAME to the ID
    print("mapping", mapping)


    results_json = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/pretrained_slow_only/mimetics_results.json"
    with open(results_json, "r") as f:
        results = json.load(f)
    
    pred_human = 0
    total = 0

    for index, item in enumerate(results):
        print("Index", index)
        full_path = item["full_path"]
        predictions = item["predictions"]

        mcq_row = mcq_df.iloc[index]
        label = mcq_row["label"]

        if full_path != mcq_row["full_path"]:
            print("Not matching")
            break

        choices = [mcq_row[f"choice_{i}"] for i in range(1, 6)]
        id_choices = [mapping[choice] for choice in choices]
        prob_choices = [predictions[i] for i in id_choices] #Get the probabilities for only the 5 choices
        max_val = max(prob_choices)
        max_index = prob_choices.index(max_val)
        chosen_label = choices[max_index]

        if chosen_label == label:
            pred_human += 1

        total += 1
    
    print("human acc", pred_human/total)
    print("pred human", pred_human)
    print("total", total)


if __name__ == "__main__":
    main()