import os
import pandas as pd
from pathlib import Path
import re

def parse_video_dirname(dirname):
    """
    Parse directory name like 'youtube_id_000123_000456' 
    Returns: (youtube_id, time_start, time_end)
    """
    # Split by underscore and take last two parts as time_start and time_end
    parts = dirname.rsplit('_', 2)
    
    youtube_id = parts[0]
    time_start = parts[1]
    time_end = parts[2]
    
    # Convert padded time strings to integers
    try:
        time_start = int(time_start)
        time_end = int(time_end)
        return youtube_id, time_start, time_end
    except ValueError:
        return None, None, None

def create_hat2_csv(base_path='/n/fs/visualai-scr/Data/HAT2', output_file='/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/hat2_dataset.csv'):
    """
    Create CSV file from HAT2 directory structure
    
    Args:
        base_path: Path to HAT2 directory
        output_file: Output CSV filename
    """
    
    base_path = Path(base_path)
    
    # Check if base directory exists
    if not base_path.exists():
        print(f"Error: Base path {base_path} does not exist")
        return
    
    # Define subdirectories
    subdirs = {
        'original': base_path / 'original',
        'segmented': base_path / 'seg', 
        'background': base_path / 'inpaint'
    }
    
    # Check if all subdirectories exist
    for name, path in subdirs.items():
        if not path.exists():
            print(f"Warning: {name} directory {path} does not exist")
    
    rows = []
    
    # Get all labels from original directory (assuming it's the most complete)
    original_dir = subdirs['original']
    if not original_dir.exists():
        print("Error: Original directory not found")
        return
    
    # Iterate through label directories
    for label_dir in original_dir.iterdir():
        if not label_dir.is_dir():
            continue
            
        label = label_dir.name
        print(f"Processing label: {label}")
        
        # Get all video directories for this label
        for video_dir in label_dir.iterdir():
            if not video_dir.is_dir():
                continue
                
            video_dirname = video_dir.name
            youtube_id, time_start, time_end = parse_video_dirname(video_dirname)
            
            if youtube_id is None:
                print(f"  Warning: Could not parse {video_dirname}")
                continue
            
            # Build paths for all three versions
            original_path = subdirs['original'] / label / video_dirname
            segmented_path = subdirs['segmented'] / label / video_dirname  
            background_path = subdirs['background'] / label / video_dirname
            
            # Check which paths actually exist
            paths_exist = {
                'original': original_path.exists(),
                'segmented': segmented_path.exists(),
                'background': background_path.exists()
            }
            
            # Only add row if at least original exists
            if paths_exist['original']:
                row = {
                    'label': label,
                    'youtube_id': youtube_id,
                    'time_start': time_start,
                    'time_end': time_end,
                    'original_path': str(original_path) if paths_exist['original'] else None,
                    'segmented_path': str(segmented_path) if paths_exist['segmented'] else None,
                    'background_only_path': str(background_path) if paths_exist['background'] else None,
                    'has_original': paths_exist['original'],
                    'has_segmented': paths_exist['segmented'],
                    'has_background': paths_exist['background']
                }
                rows.append(row)
                
                # Print status for debugging
                status = []
                for path_type, exists in paths_exist.items():
                    status.append(f"{path_type}: {'✓' if exists else '✗'}")
                print(f"  {video_dirname}: {', '.join(status)}")
            else:
                print(f"  Skipping {video_dirname}: original not found")
    
    # Create DataFrame and save
    df = pd.DataFrame(rows)
    
    if len(df) == 0:
        print("No data found! Check your directory structure.")
        return
    
    # Sort by label, then youtube_id, then time_start
    df = df.sort_values(['label', 'youtube_id', 'time_start'])
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    # Print summary
    print(f"\n=== SUMMARY ===")
    print(f"Total videos: {len(df)}")
    print(f"Unique labels: {df['label'].nunique()}")
    print(f"Unique YouTube videos: {df['youtube_id'].nunique()}")
    print(f"Videos with all 3 versions: {df[['has_original', 'has_segmented', 'has_background']].all(axis=1).sum()}")
    print(f"Videos missing segmented: {(~df['has_segmented']).sum()}")
    print(f"Videos missing background: {(~df['has_background']).sum()}")
    print(f"\nSaved to: {output_file}")
    
    # Show sample of data
    print(f"\nSample data:")
    print(df[['label', 'youtube_id', 'time_start', 'time_end']].head())
    
    return df

if __name__ == "__main__":
    # Generate the CSV
    df = create_hat2_csv()
    
    # Optional: Show some analysis
    if df is not None:
        print(f"\nLabel distribution:")
        print(df['label'].value_counts().head(10))