# Initialize Azure OpenAI client
import pandas as pd
import os 
import csv
from tqdm.auto import tqdm
import openai
from openai import OpenAI, AzureOpenAI, AsyncAzureOpenAI
import json
import argparse
import base64
import numpy as np
import wandb

# Helper function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


client = AzureOpenAI(
    api_key="",
    api_version="2024-02-01",
    azure_endpoint="https://api-ai-sandbox.princeton.edu/",
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM model on Action-Swap MCQ")
    parser.add_argument("--start", type=int, required=True, help="Start index for iteration")
    parser.add_argument("--end", type=int, required=True, help="End index for iteration (inclusive)")
    parser.add_argument("--section", type=int, required=True, help="Section #")
    return parser.parse_args()

def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

args = parse_args()

#Section 1: 0-496
#Section 2: 497-993
#Section 3: 994-1490
#Section 4: 1491-1987
#
# Section 5: 1988-2484
run = wandb.init(
    project="Slowfast_Kinetics",
    name=f"GPT Prompt, Section {args.section} ({args.start}-{args.end})",
    mode='online',
    settings=wandb.Settings(_service_wait=300)
)

# For saving prompts filtered by Azure OpenAI content filter
filtered_prompts = []

letter_to_num = {
    "A":1,
    "B":2,
    "C":3,
    "D":4,
    "E":5
}

results = {}

all_predictions = []
all_human_labels = []
all_background_labels = []
total = 0

run.log({
    "Section":args.section,
    "Start":args.start,
    "End":args.end
})

action_swap_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"
df = pd.read_csv(action_swap_csv)

for index, row in tqdm(df.iloc[args.start:args.end+1].iterrows(), total=(args.end-args.start+1)):
    path = row['action_swap_path']
    human_label = row['label_A']
    background_label = row['label_B']
    human_choice = row['human_choice']
    background_choice = row['background_choice']
    choice_1 = row['choice_1']
    choice_2 = row['choice_2']
    choice_3 = row['choice_3']
    choice_4 = row['choice_4']
    choice_5 = row['choice_5']

    total_frames = row['num_files_A']
    indices = sample_indices(8, total_frames)

    image_paths = [os.path.join(path, f"{index:06d}.jpg") for index in indices]

    # Encode all images to base64
    encoded_images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        encoded_images.append(encode_image_to_base64(image_path))
        prompt_text = f'Please just look at the background and not the person. Based on the background scene, what is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B){choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'

    # Prepare the content for the API call
    content = [
            {
                "type": "text",
                "text": prompt_text
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[0]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[1]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[2]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[3]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[4]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[5]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[6]}",
                    "detail": "low"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{encoded_images[7]}",
                    "detail": "low"
                }
            }
        ]

    try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1000,
            temperature=0.7
        )
    except openai.BadRequestError as e:
        if "content_filter" in str(e):
            filtered_prompts.append({
                "path": path,
                "prompt": prompt_text,
                "error": str(e)
            })
            print(f"⚠️ Prompt filtered at {path}, skipping.")
            continue
        else:
            raise

    try:
        first_char = response.choices[0].message.content.strip()[0]
        llm_choice = letter_to_num[first_char]
    except (IndexError, KeyError) as e:
        filtered_prompts.append({
            "path": path,
            "prompt": prompt_text,
            "response": response.choices[0].message.content if response.choices else "",
            "error": f"{type(e).__name__}: {str(e)}"
        })
        print(f"⚠️ Invalid response at {path}, skipping.")
        continue

    print("LLM CHOICE", llm_choice)


    all_predictions.append(llm_choice)
    all_human_labels.append(human_choice)
    all_background_labels.append(background_choice)

    results[path] = {
        "prompt":prompt_text,
        "response":response.choices[0].message.content,
        "prompt_tokens":response.usage.prompt_tokens,
        "completion_tokens":response.usage.completion_tokens,
        "total_tokens":response.usage.total_tokens,
        "action_swap_path":path,
        "human_label":human_label,
        "background_label":background_label
    }

    print("Prompt Text:", prompt_text)
    print("\n"+response.choices[0].message.content)


num_successful = len(results)

print("All predictions", all_predictions)
print("All Human labels", all_human_labels)
print("All bg labels", all_background_labels)


np_pred = np.array(all_predictions)
np_human = np.array(all_human_labels)
np_bg = np.array(all_background_labels)

print("Total Human", (np_pred==np_human).sum())
print("Total Background", (np_pred==np_bg).sum())
print("Human Accuracy", round((np_pred == np_human).mean(), 6))
print("Background Accuracy", round((np_pred == np_bg).mean(), 6))

run.log({
    "Total Human": (np_pred==np_human).sum(),
    "Total Background": (np_pred==np_bg).sum(),
    "Human Accuracy": round((np_pred == np_human).mean(), 6),
    "Background Accuracy": round((np_pred == np_bg).mean(), 6),
    "Number of Successful Prompts":num_successful
})

with open(f"gpt/gpt_actionswap_background_try_2_section_{args.section}.json", "w") as f:
    json.dump(results, f, indent=2)

# Save filtered prompts for inspection
with open(f"gpt/filtered_prompts_background_try_2_section_{args.section}.json", "w") as f:
    json.dump(filtered_prompts, f, indent=2)