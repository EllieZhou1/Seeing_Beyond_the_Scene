#Evaluate the manual prompts on 75% Mini Action Swap dataset
import pandas as pd
import os 
import csv
from tqdm.auto import tqdm
import openai
from openai import OpenAI, AsyncAzureOpenAI
import json
import argparse
import base64
import asyncio
import numpy as np
import wandb

# Helper function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


client = AsyncAzureOpenAI( #For image-text prompts
    api_key="",
    api_version="2024-02-01",
    azure_endpoint="https://api-ai-sandbox.princeton.edu/"
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM model on Action-Swap MCQ")
    parser.add_argument("--prompt", type=int, required=True, help="Number of the prompt you want to ask the LLM")
    parser.add_argument("--start", type=int, required=True, help="Start index for iteration")
    parser.add_argument("--end", type=int, required=True, help="End index for iteration (inclusive)")
    return parser.parse_args()

def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]


async def main():
    args = parse_args()
    prompt_num = args.prompt

    run = wandb.init(
        project="Slowfast_Kinetics",
        name=f"GPT Prompt {args.prompt}, ({args.start}-{args.end})",
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


    pred_human = 0
    pred_background = 0
    total = 0

    run.log({
        "Prompt":args.prompt,
        "Start":args.start,
        "End":args.end
    })

    action_swap_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/mini_action_swap_mcq_75.csv"
    df = pd.read_csv(action_swap_csv)



    tasks = []
    path_names = []

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

        if prompt_num == 1: #Automated Prompt Tune #1
            prompt_text = f'What is the action being performed? Answer with only the letter and choice A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'
        elif prompt_num == 2:
            prefix = "A video of a human"
            prompt_text = f'What is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {prefix} {choice_1} B) {prefix} {choice_2} C) {prefix} {choice_3} D) {prefix} {choice_4} E) {prefix} {choice_5}'
        elif prompt_num == 3:
            prompt_text = f'Focus only on the person and their motion. Ignore the background, scene, or surroundings. Based on the person’s posture, appearance, and movement, what is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B){choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'
        # elif prompt_num == 4: #Automated Prompt Tune #2
        #     prompt_text = f"Focus on the human's movements and actions. What activity is the person engaged in? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        # elif prompt_num == 5:
        #     prompt_text = f"Identify the activity based on the person's movements, ignoring the surroundings. What is the person doing? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        # elif prompt_num == 6:
        #     prompt_text = f"Describe the human action in the video, disregarding the background. What is the person doing? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        # elif prompt_num == 7:
        #     prompt_text = f"Analyze the person's movements to determine the action, ignoring the background context. What activity is being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        # elif prompt_num == 8:
        #     prompt_text = f"Concentrate solely on the person's movements to identify the action. What is the person doing, irrespective of the background? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        # elif prompt_num == 9: #best prompt
        #     prompt_text = f"Describe the human action in the video, disregarding the background. What is the person doing? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"
        elif prompt_num == 4: #Background biased prompt
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

        # try:
        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=1000,
            temperature=0
        )

        tasks.append(response)

        results[path] = {
            "prompt":prompt_text,
            "action_swap_path":path,
            "human_label":human_label,
            "background_label":background_label,
            "human_choice":human_choice,
            "background_choice":background_choice,
            "choices":[choice_1, choice_2, choice_3, choice_4, choice_5],
            "flagged":False
        }
        path_names.append(path)


        if len(tasks) == 10:
            print("Processing batch of 10 tasks...")
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            tasks = []
            for indexx, r in enumerate(responses): #Go through each respose
                if isinstance(r, openai.BadRequestError) and "content_filter" in str(r):
                    print("Content filter triggered for path:", path_names[indexx])
                    filtered_prompts.append({
                        "prompt":results[path_names[indexx]]["prompt"],
                        "action_swap_path":results[path_names[indexx]]["action_swap_path"],
                        "human_label":results[path_names[indexx]]["human_label"],
                        "background_label":results[path_names[indexx]]["background_label"],
                        "human_choice":results[path_names[indexx]]["human_choice"],
                        "background_choice":results[path_names[indexx]]["background_choice"],
                        "choices":results[path_names[indexx]]["choices"],
                        "flagged":True
                    })
                    results[path_names[indexx]]["flagged"] = True
                    continue

                total += 1

                if r.choices[0].message.content:
                    results[path_names[indexx]]["response"] = r.choices[0].message.content
                    results[path_names[indexx]]["prompt_tokens"] = r.usage.prompt_tokens
                    results[path_names[indexx]]["completion_tokens"] = r.usage.completion_tokens
                    results[path_names[indexx]]["total_tokens"] = r.usage.total_tokens

                    llm_choice = r.choices[0].message.content.strip()[0] # Get the first character of the response
                    if llm_choice not in letter_to_num:
                        print("Invalid choice from LLM:", llm_choice)
                        continue
                    
                    llm_choice = letter_to_num[llm_choice] #Convert letter to number
                    if llm_choice == results[path_names[indexx]]["human_choice"]:
                        pred_human += 1
                    elif llm_choice == results[path_names[indexx]]["background_choice"]:
                        pred_background += 1
        
            path_names = []


    if tasks:  # Process any remaining tasks
        print("Processing remaining tasks...")
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        tasks = []
        for indexx, r in enumerate(responses): #Go through each respose
            if isinstance(r, openai.BadRequestError) and "content_filter" in str(r):
                print("Content filter triggered for path:", path_names[indexx])
                filtered_prompts.append({
                    "prompt":results[path_names[indexx]]["prompt"],
                    "action_swap_path":results[path_names[indexx]]["action_swap_path"],
                    "human_label":results[path_names[indexx]]["human_label"],
                    "background_label":results[path_names[indexx]]["background_label"],
                    "human_choice":results[path_names[indexx]]["human_choice"],
                    "background_choice":results[path_names[indexx]]["background_choice"],
                    "choices":results[path_names[indexx]]["choices"],
                    "flagged":True
                })
                results[path_names[indexx]]["flagged"] = True
                continue


            total += 1
            if r.choices[0].message.content:
                results[path_names[indexx]]["response"] = r.choices[0].message.content
                results[path_names[indexx]]["prompt_tokens"] = r.usage.prompt_tokens
                results[path_names[indexx]]["completion_tokens"] = r.usage.completion_tokens
                results[path_names[indexx]]["total_tokens"] = r.usage.total_tokens

                llm_choice = r.choices[0].message.content.strip()[0] # Get the first character of the response
                if llm_choice not in letter_to_num:
                    print("Invalid choice from LLM:", llm_choice)
                    continue
                
                llm_choice = letter_to_num[llm_choice] #Convert letter to number
                if llm_choice == results[path_names[indexx]]["human_choice"]:
                    pred_human += 1
                elif llm_choice == results[path_names[indexx]]["background_choice"]:
                    pred_background += 1
    
        path_names = []


    print("Total Human", pred_human)
    print("Total Background", pred_background)
    print("Total that was not filtered out", total)
    print("Filtered prompts count", len(filtered_prompts))
    print("REAL TOTAL", len(df))
    print("Human Accuracy", pred_human/total)
    print("Background Accuracy", pred_background/total)

    run.log({
        "human_accuracy": pred_human / total,
        "background_accuracy": pred_background / total,
        "Total not filtered": total,
        "filtered_prompts_count": len(filtered_prompts),
        "REAL TOTAL": len(df),
        "pred_human": pred_human,
        "pred_background": pred_background,
    })

    dirname = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/gpt/4o-mini_manual_prompt_{prompt_num}_75_mini_actionswap"

    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save filtered prompts for inspection
    with open(os.path.join(dirname, "filtered_prompts.json"), "w") as f:
        json.dump(filtered_prompts, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
