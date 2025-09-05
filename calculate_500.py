import json
import pandas as pd

with open("gpt/gpt_actionswap_1.json", "r") as f:
    data = json.load(f)

df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv")

count = 0
human_count = 0
background_count = 0

for index, row in df.iterrows():
    if count > 500:
        break
    #go through first 500 rows

    try:
        response = data[row["action_swap_path"]]["response"]
    except:
        continue

    human_label = data[row["action_swap_path"]]["human_label"]
    background_label = data[row["action_swap_path"]]["background_label"]

    count += 1
    print("response", response)
    print("human label", human_label)
    print("background label", background_label)

    response_cropped = response[3:]
    print("response cropped", response_cropped)
    if response_cropped == human_label:
        human_count += 1
    elif response_cropped == background_label:
        background_count += 1

print(float(human_count/count))
print(float(background_count/count))