import os
import csv
import re
from pathlib import Path

def parse_video_directory_name(video_dir_name):
    """
    Parse video directory name to extract youtube_id, start_time, and end_time
    Format: <youtube_id>_<start_time>_<end_time>
    Example: ZZ1m75L6Aaw_000035_000045
    """
    parts = video_dir_name.rsplit('_', 2)
    youtube_id = parts[0]
    start_time = parts[1]
    end_time = parts[2]
    return youtube_id, start_time, end_time

def count_jpg_files(directory_path):
    """Count the number of .jpg files in a directory"""
    if not os.path.exists(directory_path):
        return 0
    return len([f for f in os.listdir(directory_path) if f.endswith('.jpg')])

def scan_directory_structure(base_path):
    """
    Scan the directory structure and return a dictionary of videos organized by label
    Returns: {label: [(video_dir_name, num_files), ...]}
    """
    videos_by_label = {}
    
    if not os.path.exists(base_path):
        print(f"Warning: Path {base_path} does not exist")
        return videos_by_label
    
    # Iterate through label directories
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if not os.path.isdir(label_path):
            continue
            
        videos_by_label[label] = []
        
        # Iterate through video directories within each label
        for video_dir in os.listdir(label_path):
            video_path = os.path.join(label_path, video_dir)
            if not os.path.isdir(video_path):
                continue
                
            # Count jpg files in this video directory
            num_files = count_jpg_files(video_path)
            videos_by_label[label].append((video_dir, num_files))
    
    return videos_by_label

def generate_action_swap_csv(original_path, inpaint_path, seg_path, output_csv_path, max_combinations=2485):
    """
    Generate CSV file with action-swap pairings
    """
    import random
    
    # Scan all three directories
    print("Scanning directory structures...")
    original_videos = scan_directory_structure(original_path)
    inpaint_videos = scan_directory_structure(inpaint_path)
    seg_videos = scan_directory_structure(seg_path)
    
    # Verify that all directories have the same structure
    if set(original_videos.keys()) != set(inpaint_videos.keys()) or set(original_videos.keys()) != set(seg_videos.keys()):
        print("Warning: Directory structures don't match between original, inpaint, and seg directories")
    
    # Create all possible combinations first
    print("Creating all possible combinations...")
    all_combinations = []
    
    for label_A in original_videos:
        for label_B in inpaint_videos:
            # Skip if same label (we want action swaps between different actions)
            if label_A == label_B:
                continue
                
            # Get videos for both labels
            videos_A = original_videos[label_A]
            videos_B = inpaint_videos[label_B]
            
            # Generate all combinations between videos of label_A and label_B
            for video_A, num_files_A in videos_A:
                for video_B, num_files_B in videos_B:
                    # Parse video directory names
                    yt_id_A, start_A, end_A = parse_video_directory_name(video_A)
                    yt_id_B, start_B, end_B = parse_video_directory_name(video_B)
                    
                    # Skip if parsing failed
                    if not all([yt_id_A, start_A, end_A, yt_id_B, start_B, end_B]):
                        continue
                    
                    # Create paths
                    path_seg_A = os.path.join(seg_path, label_A, video_A)
                    path_original_A = os.path.join(original_path, label_A, video_A)
                    path_inpaint_B = os.path.join(inpaint_path, label_B, video_B)
                    path_original_B = os.path.join(original_path, label_B, video_B)
                    
                    # Store combination data
                    combination = {
                        'path_seg_A': path_seg_A,
                        'path_original_A': path_original_A,
                        'path_inpaint_B': path_inpaint_B,
                        'path_original_B': path_original_B,
                        'num_files_A': num_files_A,
                        'num_files_B': num_files_B,
                        'label_A': label_A,
                        'label_B': label_B,
                        'yt_id_A': yt_id_A,
                        'time_start_A': start_A,
                        'time_end_A': end_A,
                        'yt_id_B': yt_id_B,
                        'time_start_B': start_B,
                        'time_end_B': end_B
                    }
                    
                    all_combinations.append(combination)
    
    print(f"Total possible combinations: {len(all_combinations)}")
    
    # Randomly sample the desired number of combinations
    if len(all_combinations) > max_combinations:
        print(f"Randomly sampling {max_combinations} combinations from {len(all_combinations)} possible combinations")
        selected_combinations = random.sample(all_combinations, max_combinations)
    else:
        print(f"Using all {len(all_combinations)} combinations (less than requested {max_combinations})")
        selected_combinations = all_combinations
    
    # Create CSV file
    with open(output_csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = [
            'path_seg_A', 'path_original_A', 'path_inpaint_B', 'path_original_B',
            'num_files_A', 'num_files_B', 'label_A', 'label_B',
            'yt_id_A', 'time_start_A', 'time_end_A',
            'yt_id_B', 'time_start_B', 'time_end_B'
        ]
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        # Write selected combinations
        for i, combination in enumerate(selected_combinations):
            writer.writerow(combination)
            
            if (i + 1) % 1000 == 0:
                print(f"Written {i + 1} combinations...")
    
    print(f"CSV file created: {output_csv_path}")
    print(f"Total combinations generated: {len(selected_combinations)}")
    
    # Print and save summary statistics
    print("\nSummary by label:")
    
    # Create dataset summary CSV
    dataset_summary_path = output_csv_path.replace('.csv', '_dataset_summary.csv')
    with open(dataset_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['label', 'total_videos'])
        writer.writeheader()
        
        for label in sorted(original_videos.keys()):
            num_videos = len(original_videos[label])
            print(f"  {label}: {num_videos} videos")
            writer.writerow({'label': label, 'total_videos': num_videos})
    
    print(f"Dataset summary saved to: {dataset_summary_path}")
    
    # Calculate label distribution in selected combinations
    print("\nLabel distribution in selected combinations:")
    label_a_counts = {}
    label_b_counts = {}
    for combo in selected_combinations:
        label_a_counts[combo['label_A']] = label_a_counts.get(combo['label_A'], 0) + 1
        label_b_counts[combo['label_B']] = label_b_counts.get(combo['label_B'], 0) + 1
    
    # Save human source (A) label distribution
    human_source_summary_path = output_csv_path.replace('.csv', '_human_source_distribution.csv')
    with open(human_source_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['label', 'count_as_human_source'])
        writer.writeheader()
        
        print("Human source (A) labels:")
        for label in sorted(label_a_counts.keys()):
            count = label_a_counts[label]
            print(f"  {label}: {count} times")
            writer.writerow({'label': label, 'count_as_human_source': count})
    
    print(f"Human source distribution saved to: {human_source_summary_path}")
    
    # Save background source (B) label distribution
    background_source_summary_path = output_csv_path.replace('.csv', '_background_source_distribution.csv')
    with open(background_source_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['label', 'count_as_background_source'])
        writer.writeheader()
        
        print("Background source (B) labels:")
        for label in sorted(label_b_counts.keys()):
            count = label_b_counts[label]
            print(f"  {label}: {count} times")
            writer.writerow({'label': label, 'count_as_background_source': count})
    
    print(f"Background source distribution saved to: {background_source_summary_path}")
    
    # Create combined distribution summary
    combined_summary_path = output_csv_path.replace('.csv', '_combined_distribution.csv')
    with open(combined_summary_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['label', 'total_videos', 'count_as_human_source', 'count_as_background_source'])
        writer.writeheader()
        
        all_labels = set(original_videos.keys()) | set(label_a_counts.keys()) | set(label_b_counts.keys())
        for label in sorted(all_labels):
            writer.writerow({
                'label': label,
                'total_videos': len(original_videos.get(label, [])),
                'count_as_human_source': label_a_counts.get(label, 0),
                'count_as_background_source': label_b_counts.get(label, 0)
            })
    
    print(f"Combined distribution summary saved to: {combined_summary_path}")

def main():
    # Define paths
    original_path = "/n/fs/visualai-scr/Data/HAT2/original"
    inpaint_path = "/n/fs/visualai-scr/Data/HAT2/inpaint"
    seg_path = "/n/fs/visualai-scr/Data/HAT2/seg"
    output_csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap_dataset.csv"
    
    print("Starting action-swap dataset CSV generation...")
    print(f"Original frames path: {original_path}")
    print(f"Inpaint frames path: {inpaint_path}")
    print(f"Segmentation frames path: {seg_path}")
    print(f"Output CSV path: {output_csv_path}")
    
    generate_action_swap_csv(original_path, inpaint_path, seg_path, output_csv_path, max_combinations=2485)
    
    print("Done!")

if __name__ == "__main__":
    main()