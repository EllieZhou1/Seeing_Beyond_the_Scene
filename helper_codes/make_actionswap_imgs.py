import os
import csv
import cv2
import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm

def create_action_swap_image(original_a_path, seg_a_path, inpaint_b_path, output_path):
    """
    Create action-swap image using the formula:
    original_human_classA * binarymask_classA + background_only_classB * (1-binarymask_classA)
    
    Args:
        original_a_path: Path to original image of human (class A)
        seg_a_path: Path to binary mask of human (class A)
        inpaint_b_path: Path to inpainted background (class B)
        output_path: Path where to save the result
    """
    try:
        # Read images
        original_a = cv2.imread(original_a_path)  # BGR format
        seg_a = cv2.imread(seg_a_path, cv2.IMREAD_GRAYSCALE)  # Binary mask
        inpaint_b = cv2.imread(inpaint_b_path)  # BGR format
        
        # Check if all images were loaded successfully
        if original_a is None:
            raise ValueError(f"Could not load original image: {original_a_path}")
        if seg_a is None:
            raise ValueError(f"Could not load segmentation mask: {seg_a_path}")
        if inpaint_b is None:
            raise ValueError(f"Could not load inpainted background: {inpaint_b_path}")
        
        # Ensure all images have the same dimensions
        h_a, w_a = original_a.shape[:2]
        h_seg, w_seg = seg_a.shape
        h_b, w_b = inpaint_b.shape[:2]
        
        if (h_a, w_a) != (h_seg, w_seg):
            # Resize segmentation mask to match original_a
            seg_a = cv2.resize(seg_a, (w_a, h_a), interpolation=cv2.INTER_NEAREST)
        
        if (h_a, w_a) != (h_b, w_b):
            # Resize inpaint_b to match original_a
            inpaint_b = cv2.resize(inpaint_b, (w_a, h_a), interpolation=cv2.INTER_LINEAR)
        
        # Normalize binary mask to [0, 1] range
        seg_a_norm = seg_a.astype(np.float32) / 255.0
        
        # Convert to 3-channel mask for RGB operations
        mask_3d = np.stack([seg_a_norm, seg_a_norm, seg_a_norm], axis=2)
        inverse_mask_3d = 1.0 - mask_3d
        
        # Convert images to float for computation
        original_a_float = original_a.astype(np.float32)
        inpaint_b_float = inpaint_b.astype(np.float32)
        
        # Apply the formula: original_human_classA * binarymask_classA + background_only_classB * (1-binarymask_classA)
        result = original_a_float * mask_3d + inpaint_b_float * inverse_mask_3d
        
        # Convert back to uint8
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save the result
        cv2.imwrite(output_path, result)
        return True
        
    except Exception as e:
        print(f"Error processing {output_path}: {str(e)}")
        return False

def process_video_sequence(row, output_base_dir, sample_idx):
    """
    Process a complete video sequence for one row in the CSV
    """
    # Extract information from the row
    path_seg_a = row['path_seg_A']
    path_original_a = row['path_original_A']
    path_inpaint_b = row['path_inpaint_B']
    num_files_a = int(row['num_files_A'])
    num_files_b = int(row['num_files_B'])
    label_a = row['label_A']
    label_b = row['label_B']
    yt_id_a = row['yt_id_A']
    time_start_a = row['time_start_A']
    time_end_a = row['time_end_A']
    yt_id_b = row['yt_id_B']
    time_start_b = row['time_start_B']
    time_end_b = row['time_end_B']
    
    # Create output directory for this sample
    sample_name = f"sample_{sample_idx:06d}_{label_a}_to_{label_b}_{yt_id_a}_{time_start_a}_{time_end_a}_bg_{yt_id_b}_{time_start_b}_{time_end_b}"
    sample_output_dir = os.path.join(output_base_dir, sample_name)
    os.makedirs(sample_output_dir, exist_ok=True)
    
    # Create metadata file for this sample
    metadata_path = os.path.join(sample_output_dir, "metadata.txt")
    with open(metadata_path, 'w') as f:
        f.write(f"Sample Index: {sample_idx}\n")
        f.write(f"Human Source (A): {label_a}\n")
        f.write(f"Background Source (B): {label_b}\n")
        f.write(f"Human Video: {yt_id_a}_{time_start_a}_{time_end_a}\n")
        f.write(f"Background Video: {yt_id_b}_{time_start_b}_{time_end_b}\n")
        f.write(f"Number of frames: {num_files_a}\n")
        f.write(f"Original A path: {path_original_a}\n")
        f.write(f"Seg A path: {path_seg_a}\n")
        f.write(f"Inpaint B path: {path_inpaint_b}\n")
    
    successful_frames = 0
    failed_frames = 0
    
    # Process each frame in the sequence
    # Always match the length of A (human source)
    for frame_idx in range(1, num_files_a + 1):
        jpg_frame_name = f"{frame_idx:06d}.jpg"
        png_frame_name = f"{frame_idx:06d}.png"
        
        # Input paths
        original_a_frame = os.path.join(path_original_a, jpg_frame_name)
        seg_a_frame = os.path.join(path_seg_a, png_frame_name)
        
        # For background, if B is shorter than A, cycle through B's frames
        # If B is longer than A, just use the corresponding frame
        b_frame_idx = ((frame_idx - 1) % num_files_b) + 1
        b_frame_name = f"{b_frame_idx:06d}.jpg"
        inpaint_b_frame = os.path.join(path_inpaint_b, b_frame_name)
        
        # Output path
        output_frame_path = os.path.join(sample_output_dir, jpg_frame_name)
        
        # Create the action-swap image
        if create_action_swap_image(original_a_frame, seg_a_frame, inpaint_b_frame, output_frame_path):
            successful_frames += 1
        else:
            failed_frames += 1
    
    return successful_frames, failed_frames, sample_output_dir

def generate_action_swap_dataset(csv_path, output_base_dir):
    """
    Generate the complete action-swap dataset from CSV file
    """
    # Create output directory
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Read CSV file
    print(f"Reading CSV file: {csv_path}")
    with open(csv_path, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        rows = list(reader)
    
    print(f"Found {len(rows)} samples to process")
    
    total_successful_frames = 0
    total_failed_frames = 0
    successful_samples = 0
    failed_samples = 0
    
    # Process each row with progress bar
    for sample_idx, row in enumerate(tqdm(rows, desc="Processing samples")):
        try:
            successful_frames, failed_frames, sample_dir = process_video_sequence(row, output_base_dir, sample_idx)
            
            if successful_frames > 0:
                successful_samples += 1
                total_successful_frames += successful_frames
                total_failed_frames += failed_frames
                
                if failed_frames > 0:
                    print(f"Sample {sample_idx}: {successful_frames} successful, {failed_frames} failed frames")
            else:
                failed_samples += 1
                print(f"Sample {sample_idx}: FAILED - no frames processed successfully")
                # Remove empty directory
                try:
                    shutil.rmtree(sample_dir)
                except:
                    pass
                    
        except Exception as e:
            failed_samples += 1
            print(f"Sample {sample_idx}: ERROR - {str(e)}")
    
    # Generate final summary
    summary_path = os.path.join(output_base_dir, "dataset_summary.txt")
    with open(summary_path, 'w') as f:
        f.write("Action-Swap Dataset Generation Summary\n")
        f.write("="*40 + "\n\n")
        f.write(f"Total samples in CSV: {len(rows)}\n")
        f.write(f"Successful samples: {successful_samples}\n")
        f.write(f"Failed samples: {failed_samples}\n")
        f.write(f"Total successful frames: {total_successful_frames}\n")
        f.write(f"Total failed frames: {total_failed_frames}\n")
        f.write(f"Output directory: {output_base_dir}\n")
    
    print("\n" + "="*50)
    print("DATASET GENERATION COMPLETE")
    print("="*50)
    print(f"Total samples processed: {len(rows)}")
    print(f"Successful samples: {successful_samples}")
    print(f"Failed samples: {failed_samples}")
    print(f"Total frames generated: {total_successful_frames}")
    print(f"Total failed frames: {total_failed_frames}")
    print(f"Output directory: {output_base_dir}")
    print(f"Summary saved to: {summary_path}")

def main():
    # Configuration
    csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_dataset.csv"  # Path to your generated CSV file
    output_base_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap"
    
    print("Action-Swap Dataset Generator")
    print("="*40)
    print(f"Input CSV: {csv_path}")
    print(f"Output directory: {output_base_dir}")
    
    # Check if CSV file exists
    if not os.path.exists(csv_path):
        print(f"ERROR: CSV file not found: {csv_path}")
        print("Please run the CSV generation script first.")
        return
    
    # Check if we have required packages
    try:
        import cv2
        import numpy as np
        from tqdm import tqdm
        print("All required packages available")
    except ImportError as e:
        print(f"ERROR: Missing required package: {e}")
        print("Please install: pip install opencv-python numpy tqdm")
        return
    
    # Generate the dataset
    generate_action_swap_dataset(csv_path, output_base_dir)

if __name__ == "__main__":
    main()