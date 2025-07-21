import pandas as pd
import csv

print("imported libraries, starting loading first df")
# Your original data
original_df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/new_minikinetics50_train_all.csv")
original_df = original_df.drop(columns=["hasPerson", "bboxes_info", "bbox_vis_path"])
print(original_df.columns)

print("loaded first df")
# Your new action swap data  
actionswap_df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/places365/new_places365_actionswap_train.csv")
print(actionswap_df.columns)
print("loaded second df")

# Combine and shuffle
all_training_data = pd.concat([original_df, actionswap_df], ignore_index=True)
print("concatted, starting shuffling")

shuffled_training_data = all_training_data.sample(frac=1, random_state=42).reset_index(drop=True)
print("shuffled, starting saving to csv")


# Save combined training set
shuffled_training_data.to_csv('/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/places365/new_mix_train.csv', index=False)
print("saved to csv")
