#Evaluate the filtered prompts on 75% Mini Action Swap dataset
import pandas as pd
import os
from tqdm.auto import tqdm
import openai
from openai import OpenAI
import json
import argparse
import base64
import wandb

# Helper function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

client = OpenAI(
    api_key=""
)

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM model on Action-Swap MCQ")
    parser.add_argument("--prompt", type=int, required=True, help="Number of the prompt you want to ask the LLM")
    return parser.parse_args()

def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

def main():
    args = parse_args()
    prompt_num = args.prompt

    run = wandb.init(
        project="Slowfast_Kinetics",
        name=f"GPT Prompt {args.prompt}, Run Filtered Prompts on OpenAI",
        mode='online',
        settings=wandb.Settings(_service_wait=300)
    )

    pred_human = 0
    pred_background = 0
    total = 0

    #Load the json file
    with open(f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/gpt/4o-mini_manual_prompt_{prompt_num}_75_mini_actionswap/filtered_prompts.json", "r") as f:
        json_file = json.load(f)

    mapping={"A":1, "B":2, "C":3, "D":4, "E":5}
    results = {}


    for entry in tqdm(json_file, desc="Processing prompts"):
        total += 1
        prompt_text = entry["prompt"]
        action_swap_path = entry["action_swap_path"]
        human_label = entry["human_label"]
        background_label = entry["background_label"]
        human_choice = entry["human_choice"]
        background_choice = entry["background_choice"]
        choices = entry["choices"]

        total_frames = sum(1 for f in os.listdir(action_swap_path) if os.path.isfile(os.path.join(action_swap_path, f))) - 1
        indices = sample_indices(8, total_frames)
        image_paths = [os.path.join(action_swap_path, f"{index:06d}.jpg") for index in indices]

        # Encode all images to base64
        encoded_images = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            encoded_images.append(encode_image_to_base64(image_path))

        #Create the message content with prompt & images
        content = [
                {
                    "type": "text",
                    "text": prompt_text
                }
                ] + \
                [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{encoded_images[jj]}",
                            "detail": "low"
                        }
                    } for jj in range(8)
                ]

        response = client.chat.completions.create(
            model='gpt-4o-mini',
            messages=[
                {
                    "role":"user", 
                    "content":content
                }
            ],
            temperature=0, # temperature = how creative/random the model is in generating response - 0 to 1 with 1 being most creative
            max_tokens=1000, # max_tokens = token limit on context to send to the mode
        )
        first_char = response.choices[0].message.content.strip()[0]
        
        if first_char in mapping:
            llm_choice = mapping[first_char]
            if llm_choice == human_choice:
                pred_human += 1
            elif llm_choice == background_choice:
                pred_background += 1

        results[action_swap_path] = {
            "prompt":prompt_text,
            "response":response.choices[0].message.content,
            "prompt_tokens":response.usage.prompt_tokens,
            "completion_tokens":response.usage.completion_tokens,
            "total_tokens":response.usage.total_tokens,
            "action_swap_path":action_swap_path,
            "human_label":human_label,
            "background_label":background_label,
            "human_choice":human_choice,
            "background_choice":background_choice,
            "choices":choices,
            "llm_choice":llm_choice if first_char in mapping else "Invalid"
        }

    
    with open(f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/gpt/4o-mini_manual_prompt_{prompt_num}_75_mini_actionswap/filtered_prompts_results.json", "w") as f:
        json.dump(results, f, indent=4)

    print("Human total:", pred_human)
    print("Background total:", pred_background)
    print("Total:", total)
    print("Human Accuracy:", pred_human/total)
    print("Background Error:", pred_background/total)
    
    run.log({
        "Human total": pred_human,
        "Background total": pred_background,
        "Total": total,
        "Human Accuracy": pred_human/total,
        "Background Error": pred_background/total
    })

main()

        















    
    



