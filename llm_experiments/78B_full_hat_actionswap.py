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
import mmcv
from typing import List, Tuple
import json
from scipy import ndimage


ORIGINAL_DIR = "/n/fs/visualai-scr/Data/Kinetics_cvf/frames_highres/val"
SEG_DIR = "/n/fs/visualai-scr/Data/HAT4/seg"
INPAINT_DIR = "/n/fs/visualai-scr/Data/HAT4/inpaint"
THRESH = 128  # segmentation threshold (0..255)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run action swap experiments")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to run (eg: InternVL3-1B)")
    parser.add_argument("--mix", type=int, help="The HAT mix to test on (1, 2, or 3)")
    return parser.parse_args()


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


## my code ###
def list_sorted_files(folder: str, ext: str) -> List[str]:
    """Return sorted list of full paths ending with ext in folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ext)]
    files.sort()  # OK because files are zero-padded like 000001
    return files

## my code ###
def center_of_mass_or_center(mask: np.ndarray) -> Tuple[float, float]:
    """Return (row, col) center of mass; fall back to geometrical center if empty."""
    if mask.sum() > 0:
        c = ndimage.measurements.center_of_mass(mask)
        return float(c[0]), float(c[1])
    h, w = mask.shape[:2]
    return h / 2.0, w / 2.0

## my code ###
def paste_foreground_on_background(fg_img: Image.Image, fg_mask: np.ndarray, bg_img: Image.Image, move_rc: Tuple[int, int]) -> Image.Image:
    """Paste fg_img onto bg_img using binary mask and (row, col) shift."""
    fg_img = Image.fromarray(fg_img).convert("RGB")
    bg_img = Image.fromarray(bg_img).convert("RGB")
    r_shift, c_shift = move_rc
    # Ensure mask is 0..255 uint8
    mask = (fg_mask > THRESH).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).convert("L")
    bg_img = bg_img.copy()
    bg_img.paste(fg_img, (int(c_shift), int(r_shift)), mask_pil)
    return bg_img

def make_actionswap_image(fg_rgb_path, fg_seg_path, bg_rgb_path, bg_seg_path):
    """
        Args: 
            fg_rgb_path (str): Path to the foreground RGB image (eg: hitting baseball/<ID_start_end>/000001.jpg)
            fg_seg_path (str): Path to the foreground segmentation mask (eg: hitting baseball/<ID_start_end>/000001.jpg)
            bg_rgb_path (str): Path to the background RGB image (eg: eating ice cream/<ID_start_end>/000001.jpg)
            bg_seg_path (str): Path to the background segmentation mask (eg: eating ice cream/<ID_start_end>/000001.jpg)
        Returns:
            a pil image of the action swap (foreground pasted on background)
    """


    # Load images and masks
    fg_img = np.array(Image.open(fg_rgb_path).convert("RGB"))
    fg_mask =np.array(Image.open(fg_seg_path).convert("L"))  # single-channel mask 0..255
    bg_img = np.array(Image.open(bg_rgb_path).convert("RGB"))
    bg_mask = np.array(Image.open(bg_seg_path).convert("L"))

    h, w = fg_img.shape[0], fg_img.shape[1]

    if h < w: #Scale the height of background to match the short side (height of fg)
        new_h = h
        new_w = int(round(h / bg_img.shape[0] * bg_img.shape[1]))
        bg_img = mmcv.imresize(bg_img, (new_w, new_h))
        bg_mask = mmcv.imresize(bg_mask, (new_w, new_h))
    else: #Scale the width of background to match the short side (width of fg)
        new_w = w
        new_h = int(round(w / bg_img.shape[1] * bg_img.shape[0]))
        bg_img = mmcv.imresize(bg_img, (new_w, new_h))
        bg_mask = mmcv.imresize(bg_mask, (new_w, new_h))

    # Compute centers
    fg_center_r, fg_center_c = center_of_mass_or_center(np.array(fg_mask))
    bg_center_r, bg_center_c = center_of_mass_or_center(np.array(bg_mask))
    move = (int(round(bg_center_r - fg_center_r)), int(round(bg_center_c - fg_center_c)))

    # Paste
    composed = paste_foreground_on_background(fg_img, fg_mask, bg_img, move)
    # composed.save("image.jpg")  # Save for debugging
    return composed


### my code for loading actionswap video ###
def load_actionswap_video(human_dir, background_dir, bound=None, input_size=448, max_num=1, num_segments=32):
    """
    Args:
        human_path (str): Path to the human video (eg: hitting baseball/<ID_start_end>)
        background_path (str): Path to the background video (eg: eating ice cream/<ID_start_end>)
        num_segments (int): Number of segments to sample from the video
    Returns:
        pixel_values (torch.Tensor): Tensor of shape (num_segments * max_num, 3, input_size, input_size)
        num_patches_list (list): List of length num_segments, where each element is the number of patches in that segment
    """
    human_path = os.path.join(ORIGINAL_DIR, human_dir)
    human_seg_path = os.path.join(SEG_DIR, human_dir)
    background_path = os.path.join(INPAINT_DIR, background_dir)
    background_seg_path = os.path.join(SEG_DIR, background_dir)

    fg_rgb_files = list_sorted_files(human_path, ".jpg")
    fg_seg_files = list_sorted_files(human_seg_path, ".png")
    bg_rgb_files = list_sorted_files(background_path, ".jpg")
    bg_seg_files = list_sorted_files(background_seg_path, ".png")
    

    max_frame_human = len(fg_rgb_files) - 1
    max_frame_bg = len(bg_rgb_files) - 1

    fps = 30  # estimate manually if unknown

    pixel_values_list, num_patches_list = [], []
    transform = build_transform(input_size=input_size)
    frame_indices_human = get_index(bound, fps, max_frame_human, first_idx=0, num_segments=num_segments)
    frame_indices_bg = get_index(bound, fps, max_frame_bg, first_idx=0, num_segments=num_segments)

    print("Frame indices human", frame_indices_human)
    print("Frame indices bg", frame_indices_bg)

    for i in range (num_segments):
        idx_human = frame_indices_human[i]
        idx_background = frame_indices_bg[i]

        image = make_actionswap_image(fg_rgb_files[idx_human], fg_seg_files[idx_human], bg_rgb_files[idx_background], bg_seg_files[idx_background])
        tiles = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
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

def run_dif_seg(model, model_name, num_seg, tokenizer, generation_config, mix):
    # If you have an 80G A100 GPU, you can put the entire model on a single GPU.
    # Otherwise, you need to load a model using multiple GPUs, please refer to the `Multiple GPUs` section.
    base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics"
    mcq_path = os.path.join(base_dir, "dataset", "original_hat_actionswap", f"actionswap_mcq_rand{mix}.json")

    data = []
    with open(mcq_path, "r") as f:
        for line in f:
            if line.strip():  # skip empty lines
                data.append(json.loads(line))

    print("Loaded the mcq json")

    pred_human = 0
    pred_bg = 0
    total = 0

    rows = []

    for item in tqdm(data):
        human_path = item['human_path']
        background_path = item['background_path']
        human_label = human_path.split("/")[0]
        background_label = background_path.split("/")[0]
        choices = item['choices']
        choice_1, choice_2, choice_3, choice_4, choice_5 = choices

        # print("Human path:", human_path)
        # print("Background path:", background_path)
        # print("human_label:", human_label)
        # print("background_label:", background_label)
        # print("Choices:", choices)

        mapping = {
            choice_1: 'A',
            choice_2: 'B',
            choice_3: 'C',
            choice_4: 'D',
            choice_5: 'E'
        }
        
        pixel_values, num_patches_list = load_actionswap_video(human_dir=human_path, background_dir=background_path, num_segments=num_seg, max_num=1)
        pixel_values = pixel_values.to(torch.bfloat16).cuda()
        video_prefix = ''.join([f'Frame{i+1}: <image>\n' for i in range(len(num_patches_list))])
        question = video_prefix + f'What is the action being performed? Your reseponse must begin with one of the following letters: A, B, C, D, or E. A) {choice_1} B) {choice_2} C) {choice_3} D) {choice_4} E) {choice_5}'

        response, history = model.chat(tokenizer, pixel_values, question, generation_config,
                                    num_patches_list=num_patches_list, history=None, return_history=True)


        # print("Response:", response)

        choice = response[0] #Choice is A, B, C, D, or E
        print("Choice:", choice)
        if choice == mapping[human_label]:
            pred_human += 1
            # print("Predicted human action")

        elif choice == mapping[background_label]:
            pred_bg += 1
            # print("Predicted Background action")

        total += 1

        rows.append({
            "human_path": human_path,
            "background_path": background_path,
            "choices": choices,
            "predicted_choice": choice
        })

        del pixel_values, num_patches_list  # free the memory
        torch.cuda.empty_cache()

    print(f"Predicted Human {num_seg} seg: {pred_human}/{total} = {pred_human/total:.6f}")
    print(f"Predicted Background {num_seg} seg: {pred_bg}/{total} = {pred_bg/total:.6f}")

    run.log({
        f"accuracy_human_{num_seg}_seg": pred_human / total,
        f"accuracy_background_{num_seg}_seg": pred_bg / total,
        f"pred_human_{num_seg}_seg": pred_human,
        f"pred_bg_{num_seg}_seg": pred_bg,
        f"total_{num_seg}_seg": total,
    })

    #Save results to a json

    os.makedirs(os.path.join(base_dir, "llm_experiments", "full_hat"), exist_ok=True)
    output_path = os.path.join(base_dir, "llm_experiments", "full_hat", f"{model_name}_segments_{num_seg}_mix_{mix}.json")
    with open(output_path, "w") as f:
        json.dump(rows, f, indent=4)

    print(f"Results saved to {output_path}")


def run_model(model_name, mix):
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
        trust_remote_code=True,
        device_map=device_map).eval() #got rid of to cuda

    print("Finished creating model")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    print("Finished creating tokenizer")

    # set the max number of tiles in `max_num`
    generation_config = dict(max_new_tokens=1024, do_sample=False)
    print("Finished creating generation config")

    segments = [1, 2, 4, 8, 16]

    for num_seg in segments:
        print(f"Running model {model_name} with {num_seg} segments")
        run_dif_seg(model, model_name, num_seg, tokenizer, generation_config, mix)


args = parse_args()
model_name = args.model_name
mix = args.mix

run = wandb.init(
    project="Slowfast_Kinetics",
    name=f"{model_name} - Mix {mix}",
    mode='online',
    # mode='disabled',
    settings=wandb.Settings(_service_wait=300)
)
num_gpus = torch.cuda.device_count()
wandb.log({"num_gpus": num_gpus, "model_name": model_name, "mix":mix})

run_model(model_name=model_name, mix=mix)