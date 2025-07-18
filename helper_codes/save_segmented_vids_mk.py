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

output_root_seg = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/new_segmented_minikinetics50_train"
output_root_binarymask = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/new_binarymasks_minikinetics50_train"

og_csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/minikinetics50_train_all.csv"

final_csv_path_all = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/new_minikinetics50_train_all.csv"
final_csv_path_only_hasHuman = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/new_minikinetics50_train_only_hasHuman.csv" #the csv where we will put ONLY the 
#samples where a human was found
final_csv_path_only_noHuman = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/minikinetics50/new_minikinetics50_train_only_noHuman.csv"

#also samples where a human was not found (in which case it would just be a black box)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device", device)


wandb.init(
    project="SlowFast_Kinetics",  # Change to your project name
    name="save_segmented_vids_for_MK50_validation",             # Optional: run name
    config={
        "og_csv_path": og_csv_path,
    }
)

#yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)
print("Loading YOLO model")
yolo = YOLO("yolov5su.pt")  # Let Ultralytics handle CUDA automatically
class_names = yolo.names  # dict: {0: 'person', 1: 'bicycle', ...}

sam2_checkpoint = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2/checkpoints/sam2.1_hiera_large.pt"

#model_cfg, expects a config path, not an absolute path
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2/sam2")
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")

df = pd.read_csv(og_csv_path)

# Compute indices for 8 and 32 evenly spaced frames
def sample_indices(n, total_frames):
    return [int(round(i * (total_frames - 1) / (n - 1) + 1)) for i in range(n)]

#Given a row in csv, sample indices, run yolo & video segmentation on it
#Returns
#   - segmented frames: a list of numpy arrays of shape [h, w, 3]
#   - hasPerson: a boolean
def run_yolo_and_seg(row):
    segmented_frames = [] #A list of PIL images containing all the segmented frames
    binarymask_frames = [] #A list of PIL images containing all the binarymask frames

    label = row['label']
    youtube_id = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']
    split = row['split']
    full_path = row['full_path']
    num_files = row['num_files']


    vid_name = f"{youtube_id}_{time_start:06d}_{time_end:06d}"
    seg_path = os.path.join(output_root_seg, label, vid_name)
    binarymask_path = os.path.join(output_root_binarymask, label, vid_name)

    indices = list(range(1, num_files + 1))
    hasPerson = False

    #Make a temporary directory storing video frames at certain indices
    #Make a numpy array "frames" to store the video frames at certain indices
    temp_dir = tempfile.mkdtemp()
    frames = []

    for counter, i in enumerate(indices):
        source = os.path.join(full_path, f"{i:06d}.jpg")
        dest = os.path.join(temp_dir, f"{counter:06d}.jpg")
        os.symlink(source, dest)

        img = Image.open(source).convert('RGB')  # Load as RGB
        img_np = np.array(img) #convert to np image
        frames.append(img_np)

    video_np = np.stack(frames, axis=0)

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

    # Draw bounding boxes on the first frame and save visualization
    import cv2
    vis_img = img0_np.copy()
    for info in all_bboxes_info:
        x1, y1, x2, y2 = map(int, info["bbox"])
        label_conf = f"{info['class']} {info['confidence']:.2f}"
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(vis_img, label_conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    vis_img_path = os.path.join(seg_path, "bbox_img", "bbox_image.jpg")
    
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

        #Go through each video frame, propogate the segmentation, calculate the binary mask
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(inference_state):
            binarymask = (out_mask_logits.squeeze(0).squeeze(0)) > 0.0 #binary mask, where 1 is person, 0 isnot
            binarymask = binarymask.unsqueeze(2).to(device)

            img = torch.from_numpy(video_np[out_frame_idx]).to(device)
            seg_img = binarymask * img

            binary_mask_np = binarymask.cpu().numpy()
            seg_img_np = seg_img.cpu().numpy()
            if seg_img_np.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                seg_img_np = np.transpose(seg_img_np, (1, 2, 0))
            segmented_frames.append(seg_img_np)
            img_bw = (binary_mask_np.squeeze() * 255).astype(np.uint8)  # shape [H, W]
            binarymask_frames.append(img_bw)
    else:
        for frame in video_np:
            zero_frame = np.zeros_like(frame, dtype=np.uint8)
            segmented_frames.append(zero_frame)
            gray_zero_frame = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
            binarymask_frames.append(gray_zero_frame)

    shutil.rmtree(temp_dir)
    return segmented_frames, binarymask_frames, hasPerson, all_bboxes_info, vis_img_path

global count
count = 0

def process_row(row):
    global count
    count += 1
    print(f"On row {count} of 6167" )
    segmented_frames, binarymask_frames, hasPerson, all_bboxes_info, vis_img_path = run_yolo_and_seg(row)

    label = row['label']
    youtube_id = row['youtube_id']
    time_start = row['time_start']
    time_end = row['time_end']
    split = row['split']
    full_path = row['full_path']
    num_files = row['num_files']

    sample_name = f"{youtube_id}_{time_start:06d}_{time_end:06d}"
    video_dir_seg = os.path.join(output_root_seg, label, sample_name)
    video_dir_binarymask = os.path.join(output_root_binarymask, label, sample_name)

    os.makedirs(video_dir_seg, exist_ok=True)
    os.makedirs(video_dir_binarymask, exist_ok=True)

    for i, seg_frame in enumerate(segmented_frames):
        Image.fromarray(seg_frame).save(os.path.join(video_dir_seg, f"{(i+1):06d}.jpg"))
    
    for i, binarymask_frame in enumerate(binarymask_frames):
        Image.fromarray(binarymask_frame.squeeze(), mode='L').save(os.path.join(video_dir_binarymask, f"{(i+1):06d}.jpg"))

    new_row = {
        'label':label,
        'youtube_id':youtube_id,
        'time_start':time_start,
        'time_end':time_end,
        'full_path':full_path,
        'num_files':num_files,
        'hasPerson':hasPerson,
        'segmented_path':video_dir_seg,
        'mask_path':video_dir_binarymask,
        'bboxes_info':str(all_bboxes_info),
        'bbox_vis_path':vis_img_path
    }

    return new_row


total_row = len(df)

all_rows = []
for idx, row in df.iterrows():
    all_rows.append(process_row(row))

hasHuman_rows = [row for row in all_rows if row['hasPerson']]
noHuman_rows = [row for row in all_rows if not row['hasPerson']]

new_df = pd.DataFrame(all_rows)
new_df.to_csv(final_csv_path_all, index=False)

new_df_hasHuman = pd.DataFrame(hasHuman_rows)
new_df_hasHuman.to_csv(final_csv_path_only_hasHuman, index=False)

new_df_noHuman = pd.DataFrame(noHuman_rows)
new_df_noHuman.to_csv(final_csv_path_only_noHuman, index=False)


print("============= DATASET CREATION IS COMPLETE ============")
print(f"A total of {len(new_df)} rows were written to {final_csv_path_all}")
print(f"A total of {len(new_df_hasHuman)} rows were written to {final_csv_path_only_hasHuman}")
print(f"A total of {len(new_df_noHuman)} rows were written to {final_csv_path_only_noHuman}")