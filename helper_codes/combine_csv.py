import os
import pandas as pd
import csv

mcq_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq.csv"
full_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/new_action_swap_all.csv"
output_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"

df_mcq = pd.read_csv(mcq_csv)
df_full = pd.read_csv(full_csv)

# Merge on "action_swap_path"
df_combined = pd.merge(df_mcq, df_full, on="action_swap_path", how="left")

df_combined.to_csv(output_csv, index=False)
