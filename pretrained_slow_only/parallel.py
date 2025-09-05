import os
# ---- set CPU + threads BEFORE torch import if possible ----
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")        # force no GPU
os.environ.setdefault("OMP_NUM_THREADS", "8")            # tune for your machine
os.environ.setdefault("MKL_NUM_THREADS", "8")

import pickle
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import torch
from typing import List, Tuple
import json
from scipy import ndimage
import glob
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ShortSideScale
import mmcv
from concurrent.futures import ProcessPoolExecutor, as_completed   # ### NEW

# ----------------- your constants -----------------
ORIGINAL_DIR = "/n/fs/visualai-scr/Data/Kinetics_cvf/frames_highres/val"
SEG_DIR = "/n/fs/visualai-scr/Data/HAT4/seg"
INPAINT_DIR = "/n/fs/visualai-scr/Data/HAT4/inpaint"
THRESH = 128
USE_PKL = True

def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run action swap experiments")
    parser.add_argument("--mix", type=int, required=True, help="The HAT mix to test on (1, 2, or 3)")
    parser.add_argument("--workers", type=int, default=8, help="Processes for data prep")  # ### NEW
    parser.add_argument("--batch_size", type=int, default=8, help="CPU inference batch size")  # ### NEW
    return parser.parse_args()

def center_of_mass_or_center(mask: np.ndarray) -> Tuple[float, float]:
    if mask.sum() > 0:
        c = ndimage.measurements.center_of_mass(mask)
        return float(c[0]), float(c[1])
    h, w = mask.shape[:2]
    return h / 2.0, w / 2.0

def paste_foreground_on_background(fg_img: np.ndarray, fg_mask: np.ndarray,
                                   bg_img: np.ndarray, move_rc: Tuple[int, int]) -> Image.Image:
    fg_img = Image.fromarray(fg_img).convert("RGB")
    bg_img = Image.fromarray(bg_img).convert("RGB")
    r_shift, c_shift = move_rc
    mask = (fg_mask > THRESH).astype(np.uint8) * 255
    mask_pil = Image.fromarray(mask).convert("L")
    bg_img = bg_img.copy()
    bg_img.paste(fg_img, (int(c_shift), int(r_shift)), mask_pil)
    return bg_img

def get_image(human_path, bg_path, human_idx, background_idx) -> Image.Image:
    fg_rgb_path = os.path.join(ORIGINAL_DIR, human_path, f"{human_idx:06d}.jpg")
    fg_seg_path = os.path.join(SEG_DIR, human_path, f"{human_idx:06d}.png")
    bg_rgb_path = os.path.join(INPAINT_DIR, bg_path, f"{background_idx:06d}.jpg")
    bg_seg_path = os.path.join(SEG_DIR, bg_path, f"{background_idx:06d}.png")

    fg_img = np.array(Image.open(fg_rgb_path).convert("RGB"))
    fg_mask = np.array(Image.open(fg_seg_path).convert("L"))
    bg_img = np.array(Image.open(bg_rgb_path).convert("RGB"))
    bg_mask = np.array(Image.open(bg_seg_path).convert("L"))

    h, w = fg_img.shape[:2]
    if h < w:
        new_h = h
        new_w = int(round(h / bg_img.shape[0] * bg_img.shape[1]))
        bg_img = mmcv.imresize(bg_img, (new_w, new_h))
        bg_mask = mmcv.imresize(bg_mask, (new_w, new_h))
    else:
        new_w = w
        new_h = int(round(w / bg_img.shape[1] * bg_img.shape[0]))
        bg_img = mmcv.imresize(bg_img, (new_w, new_h))
        bg_mask = mmcv.imresize(bg_mask, (new_w, new_h))

    fg_center_r, fg_center_c = center_of_mass_or_center(np.array(fg_mask))
    bg_center_r, bg_center_c = center_of_mass_or_center(np.array(bg_mask))
    move = (int(round(bg_center_r - fg_center_r)), int(round(bg_center_c - fg_center_c)))

    composed = paste_foreground_on_background(fg_img, fg_mask, bg_img, move)
    return composed

# transforms (operate on [C, T, H, W] torch tensor)
side_size = 256
mean = [0.45, 0.45, 0.45]
std = [0.225, 0.225, 0.225]
crop_size = 256
num_frames_slow = 8

transform = Compose([
    Lambda(lambda x: x / 255.0),
    NormalizeVideo(mean, std),
    ShortSideScale(size=side_size),
    CenterCropVideo(crop_size),
])

def sample_indices(n, total_frames):
    # guard for small folders
    total_frames = max(total_frames, n)
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

# ---------------- parallel worker: returns one preprocessed sample ----------------
def build_one_sample(args):
    """
    Runs in a separate process. Returns:
      (human_label, background_label, np.float32 array of shape [3, 8, 256, 256])
    """
    (human_path, background_path) = args

    full_human_path = os.path.join(ORIGINAL_DIR, human_path)
    full_background_path = os.path.join(INPAINT_DIR, background_path)

    num_files_A = len(glob.glob(os.path.join(full_human_path, "*")))
    num_files_B = len(glob.glob(os.path.join(full_background_path, "*")))

    human_label = human_path.split("/")[0]
    background_label = background_path.split("/")[0]

    human_indices = sample_indices(8, num_files_A)
    background_indices = sample_indices(8, num_files_B)

    frames = []
    for i in range(8):
        hi = human_indices[i]
        bi = background_indices[i]
        image = get_image(human_path, background_path, hi, bi)  # PIL
        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # [C,H,W] uint8
        frames.append(image_tensor)

    video_tensor = torch.stack(frames, dim=1).float()   # [C,T,H,W]
    video_tensor = transform(video_tensor)              # [C,T,256,256]
    return human_label, background_label, video_tensor.numpy().astype(np.float32)  # send as numpy (pickle-friendly)

# ---------------- main ----------------
def main():
    args = parse_args()

    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "8"))))  # ### NEW
    torch.set_num_interop_threads(1)

    PKL_PATH = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/original_hat_actionswap/actionswap_rand_{args.mix}.pickle"
    labels_df = pd.read_csv("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/kinetics_400_labels.csv")
    all_labels = labels_df["name"].tolist()

    # CPU only
    device = torch.device("cpu")
    print("Using device:", device)

    print("Loading model (slow_r50 on CPU)...")
    model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=True)
    model.to(device).eval()
    print("Model ready.")

    # load mapping
    if USE_PKL:
        with open(PKL_PATH, "rb") as f:
            mapping = pickle.load(f)
    else:
        mapping = {
            "hitting baseball/idueIYDAbZc_000149_000159": ["eating ice cream/0fCDlKYkRxc_000081_000091", -1, -1]
        }

    # prepare jobs list
    jobs = [(k, v[0]) for k, v in mapping.items()]
    json_path = f"/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/pretrained_slow_only/hat_actionswap_mix{args.mix}_results.json"
    results_json = []

    human_total = 0
    background_total = 0
    total = 0

    # ---------- PARALLEL DATA PREP ----------
    # Build preprocessed samples in parallel processes; each returns (human_label, background_label, np_video)
    preprocessed = []  # will store tuples
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = {ex.submit(build_one_sample, job): job for job in jobs}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Preprocessing"):
            try:
                item = fut.result()
                preprocessed.append(item)
            except Exception as e:
                print("Preprocess failed for", futures[fut], "with", repr(e))

    # ---------- CPU INFERENCE IN BATCHES ----------
    def chunks(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i:i+n]

    with torch.inference_mode():
        for batch in tqdm(list(chunks(preprocessed, args.batch_size)), desc="Inference"):
            if not batch: 
                continue
            human_labels = [b[0] for b in batch]
            background_labels = [b[1] for b in batch]
            videos_np = np.stack([b[2] for b in batch], axis=0)  # [B, C, T, 256, 256]
            videos = torch.from_numpy(videos_np).to(device)

            outputs = model(videos)  # [B, num_classes]
            preds = outputs.argmax(dim=1).tolist()
            outputs_list = outputs.tolist()

            for i in range(len(batch)):
                predicted_label = all_labels[preds[i]]
                if predicted_label == human_labels[i]:
                    human_total += 1
                elif predicted_label == background_labels[i]:
                    background_total += 1
                total += 1

                # store raw logits/scores
                results_json.append({
                    "human_label": human_labels[i],
                    "background_label": background_labels[i],
                    "predictions": outputs_list[i]
                })

    print("Human total", human_total)
    print("Background total", background_total)
    print("Total", total)
    if total > 0:
        print("Human acc", human_total / total)
        print("Background error", background_total / total)

    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=4)
    print("Wrote", json_path)

if __name__ == "__main__":
    main()