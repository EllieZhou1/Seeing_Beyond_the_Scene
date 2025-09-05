import pandas as pd
import os

csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/llm_experiments/internvl3_8B_8segments_results.csv"

df = pd.read_csv(csv_path)

count_human = df[df['choice_is_human'] == True].shape[0]
count_bg = df[df['choice_is_bg'] == True].shape[0]
total = len(df)

print(f"Human Accuracy: {count_human}/{total} = {count_human/total:.6f}")
print(f"Background Accuracy: {count_bg}/{total} = {count_bg/total:.6f}")

