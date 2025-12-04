#Evaluate the manual prompts on 75% Mini Action Swap dataset
import pandas as pd
import os
from tqdm.auto import tqdm
import openai
from openai import AzureOpenAI
import json
import argparse
import base64
import wandb

# Helper function to encode image to base64
def encode_image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")


client = AzureOpenAI( #For image-text prompts
    api_key="",
    api_version="2024-02-01",
    azure_endpoint="https://api-ai-sandbox.princeton.edu/"
)

# def parse_args():
#     parser = argparse.ArgumentParser(description="Run LLM model on Action-Swap MCQ")
#     parser.add_argument("--prompt", type=int, required=True, help="Number of the prompt you want to ask the LLM")
#     parser.add_argument("--start", type=int, required=True, help="Start index for iteration")
#     parser.add_argument("--end", type=int, required=True, help="End index for iteration (inclusive)")
#     return parser.parse_args()
import time, random
from typing import Optional

# ---- THROTTLER: limit requests/sec to avoid 429s ----
class RPSThrottler:
    def __init__(self, rps: float = 0.5):  # 0.5 rps = 1 call every 2s (safe default)
        self.min_interval = 1.0 / rps if rps > 0 else 0
        self._next_ok = 0.0

    def wait(self):
        now = time.monotonic()
        if now < self._next_ok:
            time.sleep(self._next_ok - now)
        self._next_ok = time.monotonic() + self.min_interval

throttler = RPSThrottler(rps=0.5)  # tune this upward once you know your quota

# ---- RETRY WRAPPER: exponential backoff + jitter + Retry-After ----
def chat_with_retry(client, **kwargs):
    max_retries = 6
    base = 1.5  # backoff base
    for attempt in range(max_retries):
        try:
            throttler.wait()  # enforce RPS before each call
            return client.chat.completions.create(**kwargs)
        except openai.RateLimitError as e:
            # Honor Retry-After if available
            wait_s: Optional[float] = None
            try:
                resp = getattr(e, "response", None)
                if resp is not None:
                    ra = resp.headers.get("Retry-After")
                    if ra:
                        wait_s = float(ra)
            except Exception:
                pass
            if wait_s is None:
                # exponential backoff with jitter
                wait_s = (base ** attempt) + random.uniform(0, 0.5)

            if attempt < max_retries - 1:
                print(f"[429] Rate limit hit. Sleeping {wait_s:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue
            raise  # out of retries
        except openai.APIStatusError as e:
            # Some Azure 429s/5xx also surface here; treat 429 similarly
            status = getattr(e, "status_code", None)
            if status == 429 and attempt < max_retries - 1:
                wait_s = (base ** attempt) + random.uniform(0, 0.5)
                print(f"[429/APIStatus] Sleeping {wait_s:.2f}s (attempt {attempt+1}/{max_retries})")
                time.sleep(wait_s)
                continue
            raise


def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]


def main(prompt_num):
    prompt_num = prompt_num

    run = wandb.init(
        project="Slowfast_Kinetics",
        name=f"GPT Eval 20 Prompts on 75% HAT Action Swap",
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

    total_enqueued = 0          # tasks attempted
    total_processed_ok = 0      # responses that went through (not filtered out)
    filtered_count = 0          # content-filter rejects

    action_swap_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/mini_action_swap_mcq_75.csv"
    df = pd.read_csv(action_swap_csv)


    prompts = {
        1: "Focus only on the person’s movements and actions. What activity is the person doing, regardless of the background?",
        2: "Ignore the background. Based only on the person’s movements, what action are they performing?",
        3: "Describe only the main action the person is doing, without considering the background or location.",
        4: "Based solely on the person’s body movements, what action are they performing in this video? Ignore the background.",
        5: "Ignore the setting. What is the person doing, based only on their actions and movements?",
        6: "Disregard the background. Identify the action the person is performing by observing their movements only.",
        7: "Only consider the person’s actions and body movements. What activity are they doing, without using any clues from the background?",
        8: "Focus only on the person’s motion and behavior. What action are they performing, ignoring all background details?",
        9: "Watch the person’s movements and actions only. What are they doing, without using any information from the background?",
        10: "Based only on the person’s physical actions, what activity are they performing? Do not use any background information.",
        11: "Ignore everything except the person’s movements. What action are they performing?",
        12: "Looking only at the person’s actions, what are they doing in this video? Ignore the surroundings.",
        13: "Ignore the environment. What is the person doing, based only on their actions in the video?",
        14: "Focus only on the person’s movements in the video. What action are they performing, without considering the background?",
        15: "Ignore where the video takes place. What action is the person doing, based only on their movements?",
        16: "Disregard the location and background. What is the person doing, based only on their actions?",
        17: "Without using any clues from the background or location, what action is the person performing in this video?",
        18: "Ignore the background and setting. What action is the person performing, based only on their movements?",
        19: "What is the person doing in this video, based only on their actions and not the background?",
        20: "Describe the action the person is performing, using only their movements and ignoring the background."
    }

    #Get the propmt from the dictionary
    prompt_text = prompts.get(prompt_num)
    if prompt_text is None:
        raise ValueError(f"Invalid prompt_num: {prompt_num}")

    print(f"#### STARTING PROMPT {prompt_num} ########")
    print(prompt_text)

    for index, row in tqdm(df.iterrows(), total=len(df)):
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


        ######## for manual prompt ##########
        # if prompt_num == 1: #Automated Prompt Tune #1
        #     prompt_text = f'What is the action being performed? Answer with only the letter and choice A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'
        # elif prompt_num == 2:
        #     prefix = "A video of a human"
        #     prompt_text = f'What is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {prefix} {choice_1} B) {prefix} {choice_2} C) {prefix} {choice_3} D) {prefix} {choice_4} E) {prefix} {choice_5}'
        # elif prompt_num == 3:
        #     prompt_text = f'Focus only on the person and their motion. Ignore the background, scene, or surroundings. Based on the person’s posture, appearance, and movement, what is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B){choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'
        # elif prompt_num == 4: #Background biased prompt
        #     prompt_text = f'Please just look at the background and not the person. Based on the background scene, what is the action being performed? Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B){choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'

        ####### for eval the 20 automated prompt ####


        full_prompt_text = prompt_text + f" Answer with only the letter and choice. Your response must begin with one of these capital letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}"

        # Prepare the content for the API call
        content = [
                {
                    "type": "text",
                    "text": full_prompt_text
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

        # Attempt the request synchronously
        total_enqueued += 1

        # Prepare a result shell before the call
        results[path] = {
            "prompt": full_prompt_text,
            "action_swap_path": path,
            "human_label": human_label,
            "background_label": background_label,
            "human_choice": human_choice,
            "background_choice": background_choice,
            "choices": [choice_1, choice_2, choice_3, choice_4, choice_5],
            "flagged": False
        }

        try:
            # r = client.chat.completions.create(
            #     model='gpt-4o-mini',
            #     messages=[{"role": "user", "content": content}],
            #     max_tokens=1000,
            #     temperature=0
            # )
            r = chat_with_retry(
                client,
                model='gpt-4o-mini',
                messages=[{"role": "user", "content": content}],
                max_tokens=1000,
                temperature=0
            )
        except openai.BadRequestError as e:
            msg = str(e)
            if "content_filter" in msg:
                print("Content filter triggered for path:", path)
                filtered_prompts.append({
                    "prompt": results[path]["prompt"],
                    "action_swap_path": results[path]["action_swap_path"],
                    "human_label": results[path]["human_label"],
                    "background_label": results[path]["background_label"],
                    "human_choice": results[path]["human_choice"],
                    "background_choice": results[path]["background_choice"],
                    "choices": results[path]["choices"],
                    "flagged": True,
                    "error": msg
                })
                results[path]["flagged"] = True
                filtered_count += 1
                continue

        # Success path
        if r.choices and r.choices[0].message and r.choices[0].message.content:
            text = r.choices[0].message.content
            results[path]["response"] = text
            if hasattr(r, "usage") and r.usage:
                results[path]["prompt_tokens"] = r.usage.prompt_tokens
                results[path]["completion_tokens"] = r.usage.completion_tokens
                results[path]["total_tokens"] = r.usage.total_tokens

            letter = text.strip()[:1]
            total_processed_ok += 1
            if letter not in {"A","B","C","D","E"}:
                print("Invalid choice from LLM:", letter)
                continue

            pred = {"A":1,"B":2,"C":3,"D":4,"E":5}[letter]
            if pred == human_choice:
                pred_human += 1
            elif pred == background_choice:
                pred_background += 1

        print("Index ", index, "pred_human: ", pred_human, "pred_background", pred_background, "filtered", filtered_count)


    print("Total enqueued", total_enqueued)
    print("Processed OK (parseable)", total_processed_ok)
    print("Filtered", filtered_count)
    print("Total Human", pred_human)
    print("Total Background", pred_background)

    den = total_processed_ok if total_processed_ok else 1
    human_acc = pred_human / den
    bg_acc = pred_background / den
    print("Human Accuracy", human_acc)
    print("Background Accuracy", bg_acc)

    run.log({
        f"prompt_{prompt_num}_human_accuracy": human_acc,
        f"prompt_{prompt_num}_background_accuracy": bg_acc,
        f"prompt_{prompt_num}_total_enqueued": total_enqueued,
        f"prompt_{prompt_num}_total_processed_ok": total_processed_ok,
        f"prompt_{prompt_num}_filtered_prompts_count": filtered_count,
        f"prompt_{prompt_num}_REAL TOTAL": len(df),
        f"prompt_{prompt_num}_pred_human": pred_human,
        f"prompt_{prompt_num}_pred_background": pred_background,
    })

    dirname = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/gpt/4o-mini_automated_prompt_{prompt_num}_75_mini_actionswap"

    os.makedirs(dirname, exist_ok=True)
    with open(os.path.join(dirname, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Save filtered prompts for inspection
    with open(os.path.join(dirname, "filtered_prompts.json"), "w") as f:
        json.dump(filtered_prompts, f, indent=2)


if __name__ == "__main__":
    for i in range(15, 21):
        main(i)
