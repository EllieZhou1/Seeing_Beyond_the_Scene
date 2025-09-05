import os
import pickle
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
import mmcv


# ---------------------------
# Config: set your source dirs
# ---------------------------
PKL_PATH = "../dataset/original_hat_actionswap/actionswap_rand_1.pickle"
ORIGINAL_DIR = "/n/fs/visualai-scr/Data/HAT2/original"
SEG_DIR = "/n/fs/visualai-scr/Data/HAT2/seg"
INPAINT_DIR = "/n/fs/visualai-scr/Data/HAT2/inpaint"
OUT_DIR = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/helper_codes"

# Choose a foreground (human) clip and a background clip
HUMAN_DIR = "hitting baseball/idueIYDAbZc_000149_000159"
BACKGROUND_DIR = "eating ice cream/0fCDlKYkRxc_000081_000091"  # will be overridden by PKL if desired
USE_PKL_MAPPING = False  # set True to take BACKGROUND_DIR from the pickle mapping

# Filenames are assumed zero-padded like 000001.jpg / 000001.png
FG_RGB_EXT = ".jpg"
FG_SEG_EXT = ".png"
BG_RGB_EXT = ".jpg"
BG_SEG_EXT = ".png"
THRESH = 128  # segmentation threshold (0..255)

# ----------------------------------
# Helpers
# ----------------------------------

def list_sorted_files(folder: str, ext: str) -> List[str]:
    """Return sorted list of full paths ending with ext in folder."""
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith(ext)]
    files.sort()  # OK because files are zero-padded like 000001
    return files

def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

def center_of_mass_or_center(mask: np.ndarray) -> Tuple[float, float]:
    """Return (row, col) center of mass; fall back to geometrical center if empty."""
    if mask.sum() > 0:
        c = ndimage.measurements.center_of_mass(mask)
        return float(c[0]), float(c[1])
    h, w = mask.shape[:2]
    return h / 2.0, w / 2.0

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


def main():

    # Load mapping (optional)
    mapping = None
    if os.path.isfile(PKL_PATH):
        with open(PKL_PATH, "rb") as f:
            mapping = pickle.load(f)

    human_path = os.path.join(ORIGINAL_DIR, HUMAN_DIR)
    human_seg_path = os.path.join(SEG_DIR, HUMAN_DIR)

    # Background from explicit dir or mapping
    bg_dir = BACKGROUND_DIR
    if USE_PKL_MAPPING:
        if mapping is None:
            raise RuntimeError("USE_PKL_MAPPING=True but PKL_PATH not found or invalid")
        if HUMAN_DIR not in mapping:
            raise KeyError(f"{HUMAN_DIR} not in mapping pickle")
        bg_dir = mapping[HUMAN_DIR][0]

    background_inpaint_path = os.path.join(INPAINT_DIR, bg_dir)
    background_seg_path = os.path.join(SEG_DIR, bg_dir)

    # List frames
    fg_rgb_files = list_sorted_files(human_path, FG_RGB_EXT)
    fg_seg_files = list_sorted_files(human_seg_path, FG_SEG_EXT)
    bg_rgb_files = list_sorted_files(background_inpaint_path, BG_RGB_EXT)
    bg_seg_files = list_sorted_files(background_seg_path, BG_SEG_EXT)

    # print("Counts:")
    # print("  FG RGB:", len(fg_rgb_files))
    # print("  FG SEG:", len(fg_seg_files))
    # print("  BG RGB:", len(bg_rgb_files))
    # print("  BG SEG:", len(bg_seg_files))

    if len(fg_rgb_files) == 0 or len(fg_seg_files) == 0:
        raise RuntimeError("No foreground frames or segs found.")
    if len(bg_rgb_files) == 0 or len(bg_seg_files) == 0:
        raise RuntimeError("No background frames or segs found.")



    if len(fg_rgb_files) != len(fg_seg_files):
        print("Warning: FG RGB and SEG file lists differ; using min length.")

    #Get 8 indices, uniformly sampled across the video
    human_indices = sample_indices(8, len(fg_rgb_files))
    bg_indices = sample_indices(8, len(bg_rgb_files))

    for i in range(0, 8):
        print("i=", i)
        human_idx = human_indices[i]
        background_idx = bg_indices[i]

        print(human_idx, background_idx)
        fg_rgb_path = fg_rgb_files[human_idx - 1]
        fg_seg_path = fg_seg_files[human_idx - 1]
        bg_rgb_path = bg_rgb_files[background_idx - 1]
        bg_seg_path = bg_seg_files[background_idx - 1]

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

        # Save
        out_path = os.path.join(OUT_DIR, f"swap_{i:06d}.jpg")
        composed.save(out_path, quality=95)

    print(f"Saved swapped frames to: {OUT_DIR}")


if __name__ == "__main__":
    main()