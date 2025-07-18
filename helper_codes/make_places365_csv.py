import os
import csv
import pandas as pd

data = []

base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"
data_path = os.path.join(base_dir, "dataset/places365/original_places365")
out_path = os.path.join(base_dir, "dataset/places365.csv")

#Go through the data_path
count = 0

for label_dir in os.listdir(data_path):
    if label_dir.startswith("._"):
        continue

    label_path = os.path.join(data_path, label_dir)
    for img_name in os.listdir(label_path):
        print("img nme", img_name)
        if img_name.endswith(".jpg"):
            img_path = os.path.join(label_path, img_name)
            dictionary = {'label':label_dir, 'full_path':img_path}
            data.append(dictionary)


df = pd.DataFrame(data)
df.to_csv(out_path, index=False)