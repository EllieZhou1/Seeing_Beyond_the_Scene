import os
import pandas as pd
import csv

mcq_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq.csv"
full_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_all.csv"
output_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"

df_mcq = pd.read_csv(mcq_csv)
df_full = pd.read_csv(full_csv)

missing_columns = [col for col in df_full.columns if col not in df_mcq.columns]
df_combined = pd.concat([df_mcq, df_full[missing_columns]], axis=1)

df_combined.to_csv(output_csv, index=False)





