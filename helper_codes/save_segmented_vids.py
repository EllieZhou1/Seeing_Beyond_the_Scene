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

output_root = "dataset/segmented_mimetics"
og_csv_path = "dataset/mimetics.csv"
final_csv_path = "dataset/segmented_mimetics/segmented_mimetics.csv"
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
yolo = YOLO("yolov5s.pt")  # Load the YOLOv5 model

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
    label = row['label']
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

    vis_img_path = os.path.join("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/segmented_mimetics", 
                                label, f"{row['youtube_id']}_{row['time_start']:06d}_{row['time_end']:06d}",
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
    return segmented_frames, hasPerson, all_bboxes_info, vis_img_path


class_names = yolo.names  # dict: {0: 'person', 1: 'bicycle', ...}
# print("Class names:", class_names)
# hello

#stores the rows with person & no_person, to be written into the csv
has_person_rows = []

total_row = len(df)

for idx, row in df.iterrows():
    # wandb.log({"current_row": idx, "total_rows": total_row})
    print(f"Starting row {idx} / {total_row}")
    video_path = row['full_path']
    label = row['label']
    youtube_id = row['youtube_id']
    num_files = row['num_files']    
    time_start = row['time_start']
    time_end = row['time_end']
    # print("Full path:", video_path)
    
    segmented_frames, hasPerson, all_bboxes_info, vis_img_path = run_yolo_and_seg(row)

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
    new_row['bboxes_info'] = str(all_bboxes_info)

    new_row['bbox_vis_path'] = vis_img_path
    has_person_rows.append(new_row)

new_df = pd.DataFrame(has_person_rows)
new_df.to_csv(final_csv_path, index=False)