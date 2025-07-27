import transformers
import subprocess

import pandas as pd
import csv

print("transformers version", transformers.__version__)

from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import os

os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/qwen_llm")
from qwen25_vl.qwen_vl_utils.src.qwen_vl_utils.vision_process import process_vision_info
from tqdm import tqdm


def frames_to_video(frame_dir, output_path, framerate=1):
    # Pattern: assumes frames like frame_000001.jpg
    input_pattern = os.path.join(frame_dir, '%06d.jpg')

    # Build ffmpeg command
    cmd = [
        'ffmpeg',
        '-framerate', str(framerate),
        '-i', input_pattern,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_path
    ]
    # Run the command
    subprocess.run(cmd, check=True)

# default: Load the model on the available device(s)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct", torch_dtype="auto", device_map="auto"
)

# We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-7B-Instruct",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="flash_attention_2",
#     device_map="auto",
# )

# default processor
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct", min_pixels=min_pixels, max_pixels=max_pixels)

csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq.csv"
df = pd.read_csv(csv_path)

output_rows = []
max_frames = 16

for _, row in tqdm(df.iterrows()):
    video_dir = row['action_swap_path']
    choices = [row[f'choice_{i}'] for i in range(1, 6)]
    question = f"What is the action being performed? Please answer with only the letter. A) {choices[0]} B) {choices[1]} C) {choices[2]} D) {choices[3]} E) {choices[4]}"

    all_frames = sorted([
        os.path.join(video_dir, fname)
        for fname in os.listdir(video_dir)
        if fname.endswith((".jpg", ".jpeg", ".png"))
    ])
    step = max(1, len(all_frames) // max_frames)
    frame_paths = all_frames[::step][:max_frames]

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {
            "role": "user",
            "content": [
                {"type": "video", "video": frame_paths},
                {"type": "text", "text": question},
            ]
        },
    ]

    print(f"Processing: {video_dir}")
    device = 'cuda'
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    print("Created text")
    image_inputs, video_inputs = process_vision_info(messages)
    print("Created image and video inputs")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(device)
    print("Using device:", model.device)
    print("CUDA available?", torch.cuda.is_available())
    print("Created image and video inputs")

    generated_ids = model.generate(**inputs, max_new_tokens=128)
    print("Created generated ids")
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("Model response:", output_text)

    choice = None
    for c in choices:
        if c in output_text:
            choice = c
            break

    choice_is_human = choice == row['label_A']
    choice_is_bg = choice == row['label_B']

    print("Choice is human", choice_is_human)
    print("Choice is bg", choice_is_bg)

    output_rows.append({
        'action_swap_path': row['action_swap_path'],
        'label_A': row['label_A'],
        'label_B': row['label_B'],
        'choice_1': row['choice_1'],
        'choice_2': row['choice_2'],
        'choice_3': row['choice_3'],
        'choice_4': row['choice_4'],
        'choice_5': row['choice_5'],
        'human_choice': row['label_A'],
        'background_choice': row['label_B'],
        'choice': choice,
        'choice_is_human': choice_is_human,
        'choice_is_bg': choice_is_bg
    })

output_csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/qwen_llm/qwen_results.csv"
keys = ['action_swap_path', 'label_A', 'label_B', 'choice_1', 'choice_2', 'choice_3', 'choice_4', 'choice_5', 'human_choice', 'background_choice', 'choice', 'choice_is_human', 'choice_is_bg']
with open(output_csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=keys)
    writer.writeheader()
    writer.writerows(output_rows)