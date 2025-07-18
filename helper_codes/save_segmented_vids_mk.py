#Created by Ellie Zhou on July 2, 2025

#Making a new "dataset" with segmented MiniKinetics50 videos

#1. Go through each video of dataset/mini50_clean_train.csv
#2. Run YOLO + segmentation on it
#3. If it could not find a person:
    # Continue
#4. If it did find a person:
    # a) Add the video frames to the directory I just made
        # in dataset/segmented_minikinetics50/train, create (if not created yet)
        # or go to the directory corresponding to the label
            # Create a new subdirectory with the youtube id only
            # Add in the segmented images
    # b) Add a new col in the new csv for the new vid path
    #       Write the row into the new csv

import os
import pandas as pd
import tempfile
import shutil
import numpy as np
import torch
import sys
from PIL import Image
import wandb

import wandb
from ultralytics import YOLO

os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2")
sys.path.insert(0, "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2")
# print("current working direcotry", os.getcwd())
from sam2.build_sam import build_sam2_video_predictor
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")

output_root_seg = "dataset/minikinetics50/new_segmented_minikinetics50_train"
output_root_binarymask = "dataset/minikinetics50/new_binarymasks_minikinetics50_train"

og_csv_path = "dataset/minikinetics50/minikinetics50_train_all.csv"
final_csv_path_only_hasHuman = "dataset/action_swap_only_hasHuman.csv" #the csv where we will put ONLY the 
#samples where a human was found
final_csv_path = "dataset/action_swap.csv" #contains both segmented samples where a human was found and 
#also samples where a human was not found (in which case it would just be a black box)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device", device)


# wandb.init(
#     project="SlowFast_Kinetics",  # Change to your project name
#     name="save_segmented_vids_for_HAT_minikinetics50_validation",             # Optional: run name
#     config={
#         "og_csv_path": og_csv_path,
#         "output_root": output_root,
#         "final_csv_path": final_csv_path,
#         "device": device
#     }
# )

#yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print("Loading YOLO model")
yolo = YOLO("yolov5su.pt")  # Let Ultralytics handle CUDA automatically
print("Here1")
class_names = yolo.names  # dict: {0: 'person', 1: 'bicycle', ...}

sam2_checkpoint = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2/checkpoints/sam2.1_hiera_large.pt"

#model_cfg, expects a config path, not an absolute path
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2/sam2")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")

df = pd.read_csv(og_csv_path)
print("Here2")

# Compute indices for 8 and 32 evenly spaced frames
def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

#Given a row in csv, sample indices, run yolo & video segmentation on it
#Returns
#   - segmented frames: a list of numpy arrays of shape [h, w, 3]
#   - hasPerson: a boolean
def run_yolo_and_seg(row):
    segmented_frames = [] #A list of PIL images containing all the segmented frames

    video_path = row['action_swap_path']
    sample_name = os.path.basename(video_path)  # 'sample_000000_deadlifting_to_juggling soccer ball_...'
    sample_idx = int(sample_name.split('_')[1])

    label = row['label']
    youtube_id = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']
    split = row['split']
    full_path = row['full_path']
    num_files = row['num_files']

    indices = list(range(1, num_files + 1))
    hasPerson = False

    #Make a temporary directory storing video frames at certain indices
    #Make a numpy array "frames" to store the video frames at certain indices
    temp_dir = tempfile.mkdtemp()
    frames = []

    for counter, i in enumerate(indices):
        source = os.path.join(video_path, f"{i:06d}.jpg")
        dest = os.path.join(temp_dir, f"{counter:06d}.jpg")
        os.symlink(source, dest)

        img = Image.open(source).convert('RGB')  # Load as RGB
        img_np = np.array(img) #convert to np image
        frames.append(img_np)

    video_np = np.stack(frames, axis=0)
    print("Here 3")

    img0_np = video_np[0]
    results = yolo.predict(img0_np, device='cuda' if torch.cuda.is_available() else 'cpu') #get bbox for the first frame
    print("finished yolo")
    # print("trtying to get bounding box")
    boxes = results[0].boxes
    xyxy = boxes.xyxy.cpu().numpy()
    confidence = boxes.conf.cpu().numpy()
    class_id = boxes.cls.cpu().numpy()
    bboxes = np.concatenate([xyxy, confidence[:, None], class_id[:, None]], axis=1)
    # print("got bboxes", bboxes.shape, bboxes)
    person_bboxes = bboxes[bboxes[:, 5] == 0] # get bboxes for class == person only
    person_bboxes = person_bboxes[:, :4]

    # Save bboxes, confidences, and class names for the first frame
    all_bboxes_info = []
    for i in range(bboxes.shape[0]):
        class_idx = int(bboxes[i, 5])
        class_name = class_names[class_idx]
        conf = float(bboxes[i, 4])
        coords = bboxes[i, :4].tolist()
        all_bboxes_info.append({
            "class": class_name,
            "confidence": conf,
            "bbox": coords
        })

    print("Here 4")
    # Draw bounding boxes on the first frame and save visualization
    import cv2
    vis_img = img0_np.copy()
    for info in all_bboxes_info:
        x1, y1, x2, y2 = map(int, info["bbox"])
        label_conf = f"{info['class']} {info['confidence']:.2f}"
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, label_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    vis_img_path = os.path.join(output_root, 
                                f"sample_{sample_idx:06d}_{label_a}_to_{label_b}_{yt_id_a}_{time_start_a}_{time_end_a}_bg_{yt_id_b}_{time_start_b}_{time_end_b}", 
                                "bbox_img", "bbox_image.jpg")
    
    os.makedirs(os.path.dirname(vis_img_path), exist_ok=True)
    cv2.imwrite(vis_img_path, vis_img)
    
    inference_state = predictor.init_state(video_path=temp_dir)

    #If a person was detected, then segment out the person
    if person_bboxes.shape[0] >= 1:
        hasPerson = True
        box = np.array(person_bboxes[0], dtype=np.float32)
        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box=box,
        )
        print("Here 5")

        #Go through each video frame, propogate the segmentation, calculate the binary mask
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            binarymask = (out_mask_logits.squeeze(0).squeeze(0)) > 0.0 #binary mask, where 1 is person, 0 isnot
            binarymask = binarymask.unsqueeze(2).to(device)

            img = torch.from_numpy(video_np[out_frame_idx]).to(device)
            seg_img = binarymask * img
            seg_img_np = seg_img.cpu().numpy()
            if seg_img_np.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                seg_img_np = np.transpose(seg_img_np, (1, 2, 0))

            segmented_frames.append(seg_img_np)
        print("Here 6")
    else:
        for frame in video_np:
            zero_frame = np.zeros_like(frame, dtype=np.uint8)
            segmented_frames.append(zero_frame)

    shutil.rmtree(temp_dir)
    return segmented_frames, hasPerson, all_bboxes_info, vis_img_path

global count
count = 0

def process_row(row):
    global count
    count += 1
    print(f"On row {count} of 2485" )
    segmented_frames, hasPerson, all_bboxes_info, vis_img_path = run_yolo_and_seg(row)

    path = row['action_swap_path']
    sample_name = os.path.basename(path)  # 'sample_000000_deadlifting_to_juggling soccer ball_...'
    sample_idx = int(sample_name.split('_')[1])

    label_a = row['label_A']
    label_b = row['label_B']
    yt_id_a = row['yt_id_A']
    yt_id_b = row['yt_id_B']
    time_start_a = row['time_start_A']
    time_end_a = row['time_end_A']
    time_start_b = row['time_start_B']
    time_end_b = row['time_end_B']

    sample_name = f"sample_{sample_idx:06d}_{label_a}_to_{label_b}_{yt_id_a}_{time_start_a}_{time_end_a}_bg_{yt_id_b}_{time_start_b}_{time_end_b}"
    video_dir = os.path.join(output_root, sample_name)

    os.makedirs(video_dir, exist_ok=True)

    for i, seg_frame in enumerate(segmented_frames):
        Image.fromarray(seg_frame).save(os.path.join(video_dir, f"{i:06d}.jpg"))

    new_row = row.copy()
    new_row["hasPerson"] = hasPerson
    new_row['segmented_path'] = os.path.join("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics", video_dir)
    new_row['bboxes_info'] = str(all_bboxes_info)
    new_row['bbox_vis_path'] = vis_img_path
    return new_row


total_row = len(df)

all_rows = []
for idx, row in df.iterrows():
    all_rows.append(process_row(row))

hasHuman_rows = [row for row in all_rows if row['hasPerson']]

new_df = pd.DataFrame(all_rows)
new_df.to_csv(final_csv_path, index=False)

new_df_hasHuman = pd.DataFrame(hasHuman_rows)
new_df_hasHuman.to_csv(final_csv_path_only_hasHuman, index=False)


print("============= DATASET CREATION IS COMPLETE ============")
print(f"A total of {len(new_df)} rows were written to {final_csv_path}")
print(f"A total of {len(new_df_hasHuman)} rows were written to {final_csv_path_only_hasHuman}")
