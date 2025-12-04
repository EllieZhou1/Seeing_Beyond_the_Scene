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
    parser.add_argument("--prompt", type=int, required=True, help="Number of the prompt you want to ask the LLM")
    # parser.add_argument("--start", type=int, required=True, help="Start index for iteration")
    # parser.add_argument("--end", type=int, required=True, help="End index for iteration (inclusive)")
    # parser.add_argument("--section", type=int, required=True, help="Section #")
    return parser.parse_args()

def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

args = parse_args()
prompt_num = args.prompt

#Section 1: 0-496
#Section 2: 497-993
#Section 3: 994-1490
#Section 4: 1491-1987
#
# Section 5: 1988-2484
run = wandb.init(
    project="Slowfast_Kinetics",
    name=f"GPT on Mimetics, BEST PROMPT {args.prompt}",
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
total = 0

run.log({
    "Prompt":args.prompt
})

action_swap_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq_all_fields.csv"
df = pd.read_csv(action_swap_csv)

for index, row in tqdm(df.iterrows()):
    path = row['full_path']
    human_label = row['label']
    human_choice = row['human_choice']
    choice_1 = row['choice_1']
    choice_2 = row['choice_2']
    choice_3 = row['choice_3']
    choice_4 = row['choice_4']
    choice_5 = row['choice_5']

    total_frames = row['num_files']
    indices = sample_indices(8, total_frames)

    image_paths = [os.path.join(path, f"{index:06d}.jpg") for index in indices]

    # Encode all images to base64
    encoded_images = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        encoded_images.append(encode_image_to_base64(image_path))

        if prompt_num == 1: #Automated Prompt Tune #1
            prompt_text = f'What is the action being performed? Answer with only the letter and choice A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'
        elif prompt_num == 2:
            prefix = "A photo of a human"
            prompt_text = f'What is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {prefix} {choice_1} B) {prefix} {choice_2} C) {prefix} {choice_3} D) {prefix} {choice_4} E) {prefix} {choice_5}'
        elif prompt_num == 3:
            prompt_text = f'Focus only on the person and their motion. Ignore the background, scene, or surroundings. Based on the person’s posture, appearance, and movement, what is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B){choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'
        elif prompt_num == 4: #Automated Prompt Tune #2
            prompt_text = f"Focus on the human's movements and actions. What activity is the person engaged in? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        elif prompt_num == 5:
            prompt_text = f"Identify the activity based on the person's movements, ignoring the surroundings. What is the person doing? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        elif prompt_num == 6:
            prompt_text = f"Describe the human action in the video, disregarding the background. What is the person doing? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        elif prompt_num == 7:
            prompt_text = f"Analyze the person's movements to determine the action, ignoring the background context. What activity is being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        elif prompt_num == 8:
            prompt_text = f"Concentrate solely on the person's movements to identify the action. What is the person doing, irrespective of the background? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        elif prompt_num == 9:
            prompt_text = f"Describe the human action in the video, disregarding the background. What is the person doing? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"

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

    # print("LLM CHOICE", llm_choice)


    all_predictions.append(llm_choice)
    all_human_labels.append(human_choice)

    results[path] = {
        "prompt":prompt_text,
        "response":response.choices[0].message.content,
        "prompt_tokens":response.usage.prompt_tokens,
        "completion_tokens":response.usage.completion_tokens,
        "total_tokens":response.usage.total_tokens,
        "full_path":path,
        "human_label":human_label,
    }

    # print("Prompt Text:", prompt_text)
    # print("\n"+response.choices[0].message.content)


num_successful = len(results)

print("All predictions", all_predictions)
print("All Human labels", all_human_labels)


np_pred = np.array(all_predictions)
np_human = np.array(all_human_labels)

print("Total Human", (np_pred==np_human).sum())
print("Human Accuracy", round((np_pred == np_human).mean(), 6))

run.log({
    "Total Human": (np_pred==np_human).sum(),
    "Human Accuracy": round((np_pred == np_human).mean(), 6),
    "Number of Successful Prompts":num_successful
})


dirname = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/gpt/best_prompt_run_on_mimetics"
os.makedirs(dirname, exist_ok=True)

with open(os.path.join(dirname, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

# Save filtered prompts for inspection
with open(os.path.join(dirname, "filtered_prompts.json"), "w") as f:
    json.dump(filtered_prompts, f, indent=2)
