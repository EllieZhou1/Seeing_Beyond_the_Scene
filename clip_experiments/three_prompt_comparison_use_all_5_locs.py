#Choosing a random label of the 5 to eval on
#Eval on only prompt 2 and 6

import pandas as pd
import json
import os
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel
import random
from tqdm.auto import tqdm
import numpy as np

# Paths
csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"
json_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/temp.json"

# Load data
df = pd.read_csv(csv_path)
with open(json_path, "r") as f:
    location_data = json.load(f)

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Helper to get middle frame from folder
def get_middle_frame(path):
    files = sorted([f for f in os.listdir(path) if f.endswith(".jpg")])
    if not files:
        return None
    middle_idx = len(files) // 2
    return os.path.join(path, files[middle_idx])

# Create prompts
def get_clip_prompts(label_A, label_B):
    def extract_locs(label):
        return location_data.get(label, {}).get("choices", [{}])[0].get("message", {}).get("content", "").split(", ")

    locs_A = extract_locs(label_A)
    locs_B = extract_locs(label_B)

    # return all 5 locations for both label_A and label_B combinations
    prompts = []
    for i in range(5):
        if i < len(locs_A) and i < len(locs_B):
            prompts.append(f"A person doing {label_A} at {locs_B[i]}")
            prompts.append(f"A person doing {label_B} at {locs_A[i]}")
    return prompts


from collections import defaultdict
counts = defaultdict(int)
total=0

# Evaluate CLIP
results = []
counts = defaultdict(int)
scores = defaultdict(list)
total = 0

for idx, row in tqdm(df.iterrows()):
    label_A = row["label_A"]
    label_B = row["label_B"]
    image_folder = row["action_swap_path"]

    img_path = get_middle_frame(image_folder)
    if img_path is None or not os.path.exists(img_path):
        print(f"Skipping {image_folder}, no valid image found.")
        continue

    try:
        image = Image.open(img_path).convert("RGB")
    except Exception as e:
        print(f"Failed to load/process image {img_path}: {e}")
        continue

    prompts = get_clip_prompts(label_A, label_B)

    for i in range(0, len(prompts), 2):
        prompt2 = prompts[i]
        prompt6 = prompts[i+1]
        prompt_pair = [prompt2, prompt6]

        inputs = processor(text=prompt_pair, images=[image], return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image[0]  # shape: (N,)

        best_prompt_pos = torch.argmax(logits_per_image).item()
        for j in range(len(prompt_pair)):
            scores[j].append(logits_per_image[j].item())

        counts[best_prompt_pos] += 1
        total += 1

        item = {
            "index": idx,
            "img_path": img_path,
            "label_A": label_A,
            "label_B": label_B,
            "clip_choice": best_prompt_pos,
            "prompt0": prompt2,
            "prompt1": prompt6,
            "prob0": logits_per_image[0].item(),
            "prob1": logits_per_image[1].item()
        }
        results.append(item)

# # Save results
results_df = pd.DataFrame(results)

out_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clip_experiments/three_prompt_comparison_prompt_2_and_6_test_all_5_locs"
os.makedirs(out_dir, exist_ok=True)
results_df.to_csv(os.path.join(out_dir, "clip_three_prompt_results.csv"), index=False)
print("Saved results to clip_three_prompt_results.csv")


for j in range(len(counts)):
    print(f"Prompt {j} Accuracy:", f"{counts[j]/total:.6f}")


for j in range(len(counts)):
    print(f"Prompt {j} avg score:", f"{np.array(scores[j]).mean():.6f}")