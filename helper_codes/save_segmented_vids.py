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

os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2")
sys.path.insert(0, "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/sam2")
# print("current working direcotry", os.getcwd())
from sam2.build_sam import build_sam2_video_predictor
os.chdir("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics")

output_root = "dataset/segmented_minikinetics50/val"
og_csv_path = "dataset/HAT_minikinetics50_validation.csv"
final_csv_path = "dataset/segmented_minikinetics50/segmented_minikinetics50_validation.csv"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Using device", device)


wandb.init(
    project="SlowFast_Kinetics",  # Change to your project name
    name="save_segmented_vids_for_HAT_minikinetics50_validation",             # Optional: run name
    config={
        "og_csv_path": og_csv_path,
        "output_root": output_root,
        "final_csv_path": final_csv_path,
        "device": device
    }
)

yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to(device)

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

    video_path = row['full_path']
    num_files = row['num_files']
    indices = sample_indices(32, num_files)
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

    img0_np = video_np[0]
    results = yolo(img0_np) #get bbox for the first frame
    bboxes = results.xyxy[0]
    person_bboxes = bboxes[bboxes[:, 5] == 0] # get bboxes for class == person only
    person_bboxes = person_bboxes[:, :4].cpu().numpy()
    
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
            seg_img_np = seg_img.cpu().numpy()
            if seg_img_np.shape[0] == 3:  # (3, H, W) -> (H, W, 3)
                seg_img_np = np.transpose(seg_img_np, (1, 2, 0))

            segmented_frames.append(seg_img_np)

    shutil.rmtree(temp_dir)
    return segmented_frames, hasPerson 


#stores the rows with person & no_person, to be written into the csv
has_person_rows = []

total_row = len(df)

for idx, row in df.iterrows():
    wandb.log({"current_row": idx, "total_rows": total_row})
    print(f"Starting row {idx} / {total_row}")
    video_path = row['full_path']
    label = row['label']
    youtube_id = row['youtube_id']
    num_files = row['num_files']    
    time_start = row['time_start']
    time_end = row['time_end']
    

    segmented_frames, hasPerson = run_yolo_and_seg(row)

    if hasPerson == False:
        continue

    label_dir = os.path.join(output_root, str(label))
    video_dir = os.path.join(label_dir, f"{youtube_id}_{time_start:06d}_{time_end:06d}")
    os.makedirs(video_dir, exist_ok=True)

    #Save the frames into dataset/segmented_minikinetics50/train
    for i, seg_frame in enumerate(segmented_frames):
        Image.fromarray(seg_frame).save(os.path.join(video_dir, f"{i:06d}.jpg")) 

    #Writing the new row to the csv
    new_row = row.copy()

    #Add a new column which will map to the video that was segmented, not the original
    new_row['segmented_path'] = os.path.join("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics", video_dir)
    has_person_rows.append(new_row)

new_df = pd.DataFrame(has_person_rows)
new_df.to_csv(final_csv_path, index=False)