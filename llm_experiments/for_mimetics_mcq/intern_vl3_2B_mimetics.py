import numpy as np
import torch
import torchvision.transforms as T
from decord import VideoReader, cpu
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import math
import wandb
from collections import defaultdict

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
    if bound:
        start, end = bound[0], bound[1]
    else:
        start, end = -100000, 100000
    start_idx = max(first_idx, round(start * fps))
    end_idx = min(round(end * fps), max_frame)
    seg_size = float(end_idx - start_idx) / num_segments
    frame_indices = np.array([
        int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
        for idx in range(num_segments)
    ])
    return frame_indices


def load_video(frame_dir, bound=None, input_size=448, max_num=1, num_segments=32):
    frame_paths = sorted([
        os.path.join(frame_dir, f)
        for f in os.listdir(frame_dir)
        if f.endswith(('.jpg', '.png'))
    ])
    max_frame = len(frame_paths) - 1
    fps = 30  # estimate manually if unknown

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)

    for idx in frame_indices:
        img = Image.open(frame_paths[idx]).convert('RGB')
        tiles = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(tile) for tile in tiles]
        pixel_values = torch.stack(pixel_values)
        pixel_values_list.append(pixel_values)
        num_patches_list.append(pixel_values.shape[0])

    pixel_values = torch.cat(pixel_values_list)
    return pixel_values, num_patches_list

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


total_per_label = defaultdict(int) # correct_per_label['label'] = # of times predicted correctly
correct_per_label = defaultdict(int) #total_per_label['label'] = total number of times the label occurred in mimetics


def run_dif_seg(model, model_name, num_seg, tokenizer, generation_config):
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"
    csv_path = os.path.join(base_dir, "dataset/mimetics/mimetics_mcq.csv")
    df = pd.read_csv(csv_path)
    print("Read CSV")

    pred_human = 0
    pred_bg = 0
    total = 0

    mapping = {
        1:'A',
        2:'B',
        3:'C',
        4:'D',
        5:'E'
    }

    rows = []

    for idx, row in tqdm(df.iterrows()):
        full_path = row['full_path']
        label = row['label']
        human_choice = row['human_choice']
        choice_1 = row['choice_1']
        choice_2 = row['choice_2']
        choice_3 = row['choice_3']
        choice_4 = row['choice_4']
        choice_5 = row['choice_5']
        
        pixel_values, num_patches_list = load_video(frame_dir=full_path, num_segments=num_seg, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + f'What is the action being performed? A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'

        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)

        choice = response[0]
        if choice == mapping[human_choice]:
            pred_human += 1
            correct_per_label[label] += 1

        total += 1
        total_per_label[label] += 1

        new_row = row.copy()
        new_row['choice'] = choice
        new_row['choice_is_human'] = (choice==mapping[human_choice])
        rows.append(new_row)
        del pixel_values, num_patches_list
        torch.cuda.empty_cache()

    print(f"Predicted Human: {pred_human}/{total} = {pred_human/total:.6f}")

    dir_name = os.path.join(base_dir, f"llm_experiments/for_mimetics_mcq/results/{model_name}/{num_seg}_segments")
    os.makedirs(dir_name, exist_ok=True)

    label_to_accuracy = []
    with open(os.path.join(dir_name, f"name_{model_name}_seg_{num_seg}_mimetics_mcq_summary.txt"), "w") as f:
        f.write(f"Predicted Human: {pred_human}/{total} = {pred_human/total:.6f}\n")
        f.write("Per-label Accuracy:\n")
        for label, correct_count in correct_per_label.items():
            total_count = total_per_label.get(label, 0)
            acc = correct_count / total_count if total_count > 0 else 0.0
            f.write(f"\n {label}, {acc:.6f}, ({correct_count}/{total_count})")

            label_to_accuracy.append({'label': label, 'accuracy': f"{acc:.6f}"})

    df_result = pd.DataFrame(rows)
    df_result.to_csv(os.path.join(dir_name, f"name_{model_name}_seg_{num_seg}_mimetics_mcq_results.csv"), index=False)

    df_per_label = pd.DataFrame(label_to_accuracy)
    df_per_label.to_csv(os.path.join(dir_name, f"name_{model_name}_seg_{num_seg}_mimetics_mcq_results_per_label.csv"), index=False)


def run_model(model_name):
    base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"
    path = os.path.join(base_dir, "llm_experiments", model_name)
    device_map = split_model(path)
    print("Started creating model")
    model = AutoModel.from_pretrained(
        path,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False, #Changed from unspecified to specify as false
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        device_map=device_map,
        trust_remote_code=True).eval()

    print("Finished creating model")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print("Finished creating tokenizer")

    # set the max number of tiles in `max_num`
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    print("Finished creating generation config")

    segments = [1, 2, 4, 8, 16]

    for num_seg in segments:
        print(f"Running model {model_name} with {num_seg} segments")
        run_dif_seg(model, model_name, num_seg, tokenizer, generation_config)


run = wandb.init(
    project="Slowfast_Kinetics",
    name="InternVL3-2B, Mimetics MCQ",
    mode='online',
    settings=wandb.Settings(_service_wait=300)
)
num_gpus = torch.cuda.device_count()
wandb.log({"num_gpus": num_gpus})

run_model(model_name="InternVL3-2B")