import openai
import json
from openai import AzureOpenAI
from openai import OpenAI, AsyncAzureOpenAI
import wandb
import argparse
import os
import numpy as np
from tqdm.auto import tqdm
import base64
import pandas as pd
import csv

instruction = "Your task is to help me design an ideal action classification prompt for a vision/language model to minimize the background bias while improving the accuracy. The provided video shows a human performing an action in an unrelated background (e.g., a video of a person playing tennis on a golf course). The model should classify the video as 'playing tennis' based on the action in the video and ignore the background. The model’s accuracy is the percentage of correctly classified videos. The background bias is how often the model classifies based on background instead of the action. You can test your prompt by outputting a single new line starting with 'PROMPT:'. Do not list options—the system will provide them automatically. Try to keep the prompt as short and simple as possible, but be creative. It might be reasonable to summarize the insights of previous attempts and to outline your goals before responding with a new prompt, but make sure that only the prompt starts with 'PROMPT:'. In response to the prompt, you will be told the accuracy and background bias. Then you will refine the prompt, and we will continue until I say stop. Let’s go!"

def parse_args():
    parser = argparse.ArgumentParser(description="Run LLM model on Action-Swap MCQ - AUTOMATED VERSION")
    parser.add_argument("--start", type=int, required=True, help="Start index for iteration")
    parser.add_argument("--end", type=int, required=True, help="End index for iteration (inclusive)")
    parser.add_argument("--section", type=int, required=True, help="Section #")
    return parser.parse_args()

# Helper function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

prompt_log_path = "gpt/action_swap_results/generated_prompts_log.txt"


action_swap_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"
df = pd.read_csv(action_swap_csv)

args = parse_args()
run = wandb.init(
    project="Slowfast_Kinetics",
    name=f"GPT Automated Prompt Tuning, Section {args.section} ({args.start}-{args.end})",
    mode='online',
    settings=wandb.Settings(_service_wait=300)
)


def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

def evaluate_prompt_on_dataset(prompt_text, prompt_num):
    filtered_prompts = [] # For saving prompts filtered by Azure OpenAI content filter

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
        "Current Prompt":prompt_num,
        "Section":args.section,
        "Start":args.start,
        "End":args.end
    })

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

        image_paths = [os.path.join(path, f"{indx:06d}.jpg") for indx in indices]

        # Encode all images to base64
        encoded_images = []
        for image_path in image_paths:
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            encoded_images.append(encode_image_to_base64(image_path))

        prefix = "A photo of a human"
        full_prompt_text = prompt_text + f"Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. You cannot give a blank answer. A) {prefix} {choice_1} B) {prefix} {choice_2} C) {prefix} {choice_3} D) {prefix} {choice_4} E) {prefix} {choice_5}"

        # Prepare the content for the API call
        content = [
                {
                    "type": "text",
                    "text": full_prompt_text
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
                    "prompt": full_prompt_text,
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
                "prompt": full_prompt_text,
                "response": response.choices[0].message.content if response.choices else "",
                "error": f"{type(e).__name__}: {str(e)}"
            })
            print(f"⚠️ Invalid response because of index error or key error at {path}, skipping.")
            continue


        all_predictions.append(llm_choice)
        all_human_labels.append(human_choice)
        all_background_labels.append(background_choice)

        results[path] = {
            "prompt":full_prompt_text,
            "response":response.choices[0].message.content,
            "prompt_tokens":response.usage.prompt_tokens,
            "completion_tokens":response.usage.completion_tokens,
            "total_tokens":response.usage.total_tokens,
            "action_swap_path":path,
            "human_label":human_label,
            "background_label":background_label
        }

        # print("Prompt Text:", prompt_text)
        # print("\n"+response.choices[0].message.content)
        # print("LLM CHOICE", llm_choice)
    
    num_successful = len(results)

    print(f"============== FINISHED WITH PROMPT {prompt_num} ============")
    print("All predictions", all_predictions)
    print("All Human labels", all_human_labels)
    print("All bg labels", all_background_labels)

    np_pred = np.array(all_predictions)
    np_human = np.array(all_human_labels)
    np_bg = np.array(all_background_labels)

    print("Total Human", (np_pred==np_human).sum())
    print("Total Background", (np_pred==np_bg).sum())

    human_accuracy = round((np_pred == np_human).mean(), 6)
    background_accuracy = round((np_pred == np_bg).mean(), 6)

    print("Human Accuracy", human_accuracy)
    print("Background Accuracy", background_accuracy)

    run.log({
        f"Prompt {prompt_num} Total Human": (np_pred==np_human).sum(),
        f"Prompt {prompt_num} Total Background": (np_pred==np_bg).sum(),
        f"Prompt {prompt_num} Human Accuracy": round((np_pred == np_human).mean(), 6),
        f"Prompt {prompt_num} Background Accuracy": round((np_pred == np_bg).mean(), 6),
        f"Prompt {prompt_num} Number of Successful Prompts":num_successful
    })

    with open(f"gpt/action_swap_results/gpt_actionswap_prompt_{prompt_num}_section_{args.section}.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save filtered prompts for inspection
    with open(f"gpt/action_swap_results/filtered_prompts_prompt_{prompt_num}_section_{args.section}.json", "w") as f:
        json.dump(filtered_prompts, f, indent=2)

    return human_accuracy*100, background_accuracy*100


# === SETUP CLIENT ===
client = AzureOpenAI(
    api_key="",
    api_version="2024-02-01",
    azure_endpoint="https://api-ai-sandbox.princeton.edu/"
)

#Prompt 2 and 3 were manually curated

# === INITIAL PROMPT STATE ===

messages = [
    {'role': 'user', 'content': instruction},
    {"role": "assistant", "content": "Prompt: What is the action being performed?"}, 
    {"role": 'user', 'content': "Accuracy: 36.40%, Background Bias: 46.68%"}, #Prompt 1
    {'role': 'assistant', 'content': "Prompt: Focus on the human's movements and actions. What activity is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: 38.4289%, Background Bias: 50.5308%"}, #Prompt 4
    {"role": 'assistant', 'content':"Prompt: Identify the activity based on the person's movements, ignoring the surroundings. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 38.0042%, Background Bias: 50.3185%"},#Prompt 5
    {"role": 'assistant', 'content': "Prompt: Describe the human action in the video, disregarding the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 40.0424%, Background Bias: 45.339%"},#Prompt 6
    {"role": 'assistant', 'content': "Prompt: Analyze the person's movements to determine the action, ignoring the background context. What activity is being performed?"},
    {"role": 'user', 'content': "Accuracy: 39.4068%, Background Bias: 48.3051%"}, #"Prompt 7"
    {"role": 'assistant', 'content': "Prompt: Observe the individual's movements and actions to identify the activity, without considering the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 37.2881%, Background Bias: 51.0593%"}, #Prompt 8
    {"role": 'assistant', 'content': "Prompt: Concentrate on the person's actions and movements to determine the activity, setting aside the background. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: 36.8644%, Background Bias: 51.9068%"}, #Prompt 9
    {"role": 'assistant', 'content':"Prompt: Examine the individual's movements and actions to identify the activity, ignoring the background details. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 37.3134%, Background Bias: 51.1727%"}, #Prompt 10
    {"role": 'assistant', 'content': "Prompt: Focus on the person's actions and movements to determine the activity, disregarding the background. What is the person engaged in?"}, 
    {"role": 'user', 'content': "Accuracy: 38.0042%, Background Bias: 50.5308%"}, #Prompt 11
    {"role": 'assistant', 'content':"Prompt: Analyze the individual's movements and actions to identify the activity, without considering the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 36.8644%, Background Bias: 52.1186%"}, #Prompt 12
    {"role": 'assistant', 'content': "Prompt: Observe the person's actions and movements to determine the activity, setting aside the background context. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: 38.7712%, Background Bias: 51.4831%"}, #Prompt 13
    {"role": 'assistant', 'content':"Prompt: Focus on the individual's movements and actions to identify the activity, ignoring the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 39.4456%, Background Bias: 49.6802%"}, #14
    {"role": 'assistant', 'content':"Prompt: Concentrate on the person's actions and movements to determine the activity, disregarding the background details. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: nan%, Background Bias: nan%"}, #15
    {"role": 'assistant', 'content':"Prompt: Examine the individual's movements and actions to identify the activity, without considering the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 37.2881%, Background Bias: 51.6949%"}, #16
    {"role": 'assistant', 'content':"Prompt: Focus on the person's actions and movements to determine the activity, setting aside the background context. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: 37.3673%, Background Bias: 51.1677%"},#17
    {"role": 'assistant', 'content':"Prompt: Observe the individual's movements and actions to identify the activity, ignoring the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 39.4904%, Background Bias: 50.7431%"},#18
    {"role": 'assistant', 'content':"Prompt: Concentrate on the person's movements and actions to determine the activity, disregarding the background details. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: nan%, Background Bias: nan%"},#19
    {"role": 'assistant', 'content':"Prompt: Analyze the individual's movements and actions to identify the activity, without considering the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 37.4468%, Background Bias: 50.6383%"},#20
    {"role": 'assistant', 'content':"Prompt: Focus on the person's actions and movements to determine the activity, setting aside the background context. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: 36.3057%, Background Bias: 51.3800%"},#21
    {"role": 'assistant', 'content':"Prompt: Observe the individual's movements and actions to identify the activity, ignoring the background. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 39.4068%, Background Bias: 50.8475%"},#22
    {"role": 'assistant', 'content': "Prompt: Concentrate on the person's movements and actions to determine the activity, disregarding the background details. What is the person engaged in?"},
    {"role": 'user', 'content': "Accuracy: nan%, Background Bias: nan%"}, #23
    {"role": 'assistant', 'content': "Prompt: Analyze the person's movements and actions to identify the activity, ignoring the background context. What is the person doing?"},
    {"role": 'user', 'content': "Accuracy: 37.0213%, Background Bias: 50.2128%"} #24
]

# === TUNING LOOP ===

i = 25
TOTAL_ITERATIONS = 20

for j in range(i, i + TOTAL_ITERATIONS):
    response = client.chat.completions.create(
        model='gpt-4o',
        temperature=0,
        max_tokens=1000,
        top_p=0.5,
        messages=messages
    )
    assistant_reply = response.choices[0].message.content.strip()
    print(f"\n Prompt {j}:")
    print("GPT-4o Suggestion for Prompt:", assistant_reply)

    # Step 2: Extract the actual prompt
    if "PROMPT:" in assistant_reply:
        new_prompt = assistant_reply.split("PROMPT:")[-1].strip()
    elif "Prompt:" in assistant_reply:
        new_prompt = assistant_reply.split("Prompt:")[-1].strip()
    else:
        print("No 'PROMPT:' or 'Prompt:' found, skipping.")
        break

    # Step 3: Evaluate prompt on dataset (returns accuracy and bias)
    accuracy, background_bias = evaluate_prompt_on_dataset(prompt_text=new_prompt, prompt_num=j)

    print(f"Accuracy: {accuracy:.4f}%, Background Bias: {background_bias:.4f}%")

    with open(prompt_log_path, "a") as f:
        f.write(f"Prompt {j}: {assistant_reply}\n")
        f.write(f"Accuracy: {accuracy:.4f}%, Background Bias: {background_bias:.4f}%\n\n")

    # Step 4: Feed metrics back to GPT
    messages.append({'role': 'assistant', 'content': f"Prompt: {new_prompt}"})
    messages.append({'role': 'user', 'content': f"Instructions: {instruction}. The previous result: {accuracy:.4f}%, Background Bias: {background_bias:.4f}% What's the next prompt? Start with the phrase 'Prompt:'"})