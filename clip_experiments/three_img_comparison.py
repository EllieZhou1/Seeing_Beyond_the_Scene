def save_detailed_results_to_txt(result, label, youtube_id, time_start, time_end, 
                                frame_num, output_dir="clip_experiments/three_img_comparison_results/per_video_results"):
    """
    Save detailed results for one video to a txt file
    
    Args:
        result: Results dictionary from test_single_video_frame
        label: Action label
        youtube_id: YouTube video ID
        time_start, time_end: Time bounds
        frame_num: Frame number analyzed
        output_dir: Directory to save txt files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename: label_youtubeid_timestart_timeend.txt
    filename = f"{label}_{youtube_id}_{time_start:06d}_{time_end:06d}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"CLIP Bias Analysis Results\n")
        f.write(f"=" * 50 + "\n")
        f.write(f"Video: {youtube_id}\n")
        f.write(f"Action: {label}\n") 
        f.write(f"Time: {time_start}-{time_end}s\n")
        f.write(f"Frame: {frame_num}\n\n")
        
        for img_type in ['original', 'human_only', 'background_only']:
            f.write(f"{img_type.upper().replace('_', ' ')}:\n")
            f.write(f"  Correct action rank: {result[img_type]['correct_ranking']}/50 (score: {result[img_type]['correct_score']:.4f})\n")
            f.write(f"  Top 10 predictions:\n")
            
            for i, item in enumerate(result[img_type]['top10_predictions'][:10]):
                marker = "👉" if item['action'] == label else "  "
                f.write(f"    {marker} {item['rank']:2d}. {item['action']} ({item['score']:.4f})\n")
            f.write(f"\n")
        
        # Add bias assessment
        bg_rank = result['background_only']['correct_ranking']
        human_rank = result['human_only']['correct_ranking']
        orig_rank = result['original']['correct_ranking']
        
        f.write("BIAS ASSESSMENT:\n")
        if result['background_only']['top5']:
            f.write("⚠️  POTENTIAL BIAS: Background-only ranks in top 5!\n")
        else:
            f.write("✅ Good: Background-only doesn't rank highly\n")
            
        f.write(f"Ranking drop from original to background: {bg_rank - orig_rank}\n")
        f.write(f"Ranking drop from human to background: {bg_rank - human_rank}\n")
        
        if bg_rank < human_rank:
            f.write("🚨 WARNING: Background ranks higher than human-only!\n")
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import json

# Load CLIP model once at the start
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Loaded CLIP model")

def get_frame_path(base_path, frame_num=50):
    """Get path to specific frame number"""
    if base_path is None:
        return None
    return os.path.join(base_path, f"{frame_num:06d}.jpg") if "original" in base_path or "inpaint" in base_path else os.path.join(base_path, f"{frame_num:06d}.png")

def test_single_video_frame(original_path, seg_path, background_path, 
                           all_labels, correct_label, frame_num=50):
    """
    Test bias for one video frame with 3 image versions
    
    Args:
        original_path: base path to original video directory
        seg_path: base path to segmented video directory  
        background_path: base path to background video directory
        all_labels: list of all action labels
        correct_label: the correct label for this action
        frame_num: which frame to analyze (default 50)
    
    Returns:
        Dictionary with rankings and scores, or None if error
    """
    
    try:
        # Get frame paths
        orig_frame = get_frame_path(original_path, frame_num)
        seg_frame = get_frame_path(seg_path, frame_num)
        bg_frame = get_frame_path(background_path, frame_num)
        
        # Check if files exist
        if not all([orig_frame and os.path.exists(orig_frame),
                   seg_frame and os.path.exists(seg_frame), 
                   bg_frame and os.path.exists(bg_frame)]):
            return None
        
        # Load the 3 images
        original = np.array(Image.open(orig_frame).convert('RGB'))  
        seg = np.array(Image.open(seg_frame).convert('L'))  # Binary mask
        inpaint = np.array(Image.open(bg_frame).convert('RGB')) 

        # Create human-only image by applying mask
        seg_norm = seg.astype(np.float32) / 255.0
        mask_3d = np.stack([seg_norm, seg_norm, seg_norm], axis=2)
        human_only = (mask_3d * original).astype(np.uint8)

        images = [original, human_only, inpaint]
        
        # Process images and text together
        inputs = processor(text=all_labels, images=images, return_tensors="pt", 
                          padding=True, do_convert_rgb=False)
        
        # Get CLIP predictions
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (3, num_labels)
        probs = logits_per_image.softmax(dim=1)  # convert to probabilities
        
        # Find index of correct label
        correct_idx = all_labels.index(correct_label)
        
        results = {}
        image_types = ['original', 'human_only', 'background_only']
        
        for i, img_type in enumerate(image_types):
            # Get probability scores for this image
            scores = probs[i].detach().numpy()
            
            # Get full ranking order (indices sorted by score, descending)
            ranking_indices = np.argsort(scores)[::-1]
            
            # Create ranked list of actions with scores (only top 10 to save space)
            ranked_actions = []
            for rank, idx in enumerate(ranking_indices[:10]):  # Only save top 10
                ranked_actions.append({
                    'rank': rank + 1,
                    'action': all_labels[idx],
                    'score': float(scores[idx])
                })
            
            # Find where correct label ranks
            correct_ranking = np.where(ranking_indices == correct_idx)[0][0] + 1
            correct_score = scores[correct_idx].item()
            
            results[img_type] = {
                'correct_ranking': int(correct_ranking),
                'correct_score': float(correct_score),
                'top10_predictions': ranked_actions,
                'top5': correct_ranking <= 5,
                'top10': correct_ranking <= 10
            }
        
        return results
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        return None

def analyze_hat2_dataset(csv_path, output_csv, 
                        frame_num=50, sample_size=None):
    """
    Analyze entire HAT2 dataset for CLIP bias
    
    Args:
        csv_path: Path to HAT2 CSV file
        output_csv: Output CSV filename
        frame_num: Which frame to analyze (default 50)
        sample_size: If specified, randomly sample this many rows for testing
    """
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} entries")
    
    # Filter to only rows that have all three image types
    complete_df = df[(df['has_original'] == True) & 
                    (df['has_segmented'] == True) & 
                    (df['has_background'] == True)]
    print(f"Found {len(complete_df)} complete entries")
    
    # Sample if requested
    if sample_size and sample_size < len(complete_df):
        complete_df = complete_df.sample(n=sample_size, random_state=42)
        print(f"Sampling {sample_size} entries for analysis")
    
    # Your 50 action labels
    all_labels = ['playing guitar', 'bowling', 'playing saxophone', 'brushing teeth', 
                    'playing basketball', 'tying tie', 'skiing slalom', 'brushing hair', 
                    'punching person (boxing)', 'playing accordion', 'archery', 
                    'catching or throwing frisbee', 'drinking', 'reading book', 
                    'eating ice cream', 'flying kite', 'sweeping floor', 
                    'walking the dog', 'skipping rope', 'clean and jerk', 
                    'eating cake', 'catching or throwing baseball', 
                    'skiing (not slalom or crosscountry)', 'juggling soccer ball', 
                    'deadlifting', 'driving car', 'cleaning windows', 'shooting basketball', 
                    'canoeing or kayaking', 'surfing water', 'playing volleyball', 'opening bottle', 
                    'playing piano', 'writing', 'dribbling basketball', 'reading newspaper', 'playing violin', 
                    'juggling balls', 'playing trumpet', 'smoking', 'shooting goal (soccer)', 'hitting baseball', 
                    'sword fighting', 'climbing ladder', 'playing bass guitar', 'playing tennis', 'climbing a rope', 
                    'golf driving', 'hurdling', 'dunking basketball']
    
    results_rows = []
    
    # Process each video
    for idx, row in tqdm(complete_df.iterrows(), total=len(complete_df), desc="Analyzing videos"):
        
        result = test_single_video_frame( #Returns array of dictionaries - correct score, correct ranking, top 10 label predictions, correct label in top5?, correct label in top10?
            original_path=row['original_path'],
            seg_path=row['segmented_path'], 
            background_path=row['background_only_path'],
            all_labels=all_labels,
            correct_label=row['label'],
            frame_num=frame_num
        )
        
        if result is None:
            print(f"Skipping {row['youtube_id']} - {row['label']}: processing error")
            continue
        
        # Save detailed results to txt file
        save_detailed_results_to_txt(
            result=result,
            label=row['label'],
            youtube_id=row['youtube_id'], 
            time_start=row['time_start'],
            time_end=row['time_end'],
            frame_num=frame_num
        )
        
        # Create rows for CSV (one per image type)
        for img_type in ['original', 'human_only', 'background_only']:
            results_rows.append({
                'label': row['label'],
                'youtube_id': row['youtube_id'],
                'time_start': row['time_start'],
                'time_end': row['time_end'],
                'frame_num': frame_num,
                'image_type': img_type,
                'correct_ranking': result[img_type]['correct_ranking'],
                'correct_score': result[img_type]['correct_score'],
                'top5': result[img_type]['top5'],
                'top10': result[img_type]['top10'],
                'top5_predictions': str([p['action'] for p in result[img_type]['top10_predictions'][:5]])
            })
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results_df)} results to {output_csv}")
    
    return results_df

def analyze_bias_results(results_csv):
    """
    Analyze the bias results and print summary statistics
    """
    df = pd.read_csv(results_csv)
    
    print("=== HAT2 CLIP BIAS ANALYSIS ===\n")
    
    # Overall statistics
    bg_only = df[df['image_type'] == 'background_only']
    original = df[df['image_type'] == 'original'] 
    human_only = df[df['image_type'] == 'human_only']
    
    print("OVERALL PERFORMANCE METRICS:")
    print(f"  Original images - Top-1: {original['correct_ranking'].eq(1).mean():.1%}, Top-5: {original['top5'].mean():.1%}")
    print(f"  Human-only images - Top-1: {human_only['correct_ranking'].eq(1).mean():.1%}, Top-5: {human_only['top5'].mean():.1%}")
    print(f"  Background-only images - Top-1: {bg_only['correct_ranking'].eq(1).mean():.1%}, Top-5: {bg_only['top5'].mean():.1%}")
    

    print(f"\nHow Often Background-only in top-5: {bg_only['top5'].sum()}/{len(bg_only)} ({bg_only['top5'].mean():.1%})")
    print(f"  How Often Background-only in top-10: {bg_only['top10'].sum()}/{len(bg_only)} ({bg_only['top10'].mean():.1%})")

    print(f"\nMean background-only rank: {bg_only['correct_ranking'].mean():.1f}")
    print(f"  Mean human-only rank: {human_only['correct_ranking'].mean():.1f}")
    print(f"  Mean original rank: {original['correct_ranking'].mean():.1f}")

    print(f"\nMean background-only similarity score: {bg_only['correct_score'].mean():.1f}")
    print(f"  Mean human-only similarity score: {human_only['correct_score'].mean():.1f}")
    print(f"  Mean original similarity score: {original['correct_score'].mean():.1f}")
    
    # Most biased actions
    print(f"\nBIASED ACTIONS FROM MOST TO LEAST (actions w/ the highest % of predicting correct label in background only):")

    bg_only = bg_only.assign(is_rank_1 = bg_only['correct_ranking'] == 1) #add a new column (1 if correct_ranking=1, 0 if correct ranking is not 1)

    biased_actions = bg_only.groupby('label').agg({ 
        'is_rank_1': 'mean',
        'correct_ranking': ['count', 'mean'], 
        'correct_score': 'mean'
    }).round(3)
    biased_actions.columns = ['bg_only_top_1_rate','count', 'avg_rank', 'avg_score']
    #bg_only_top_1_rate: for each label (based on human action), # of times bg-only predicted the label correctly/total num vids with that label
    #count: total num of vids with that label
    #avg_rank: how high did the correct label rank when compared to similarity scores for the other labels
    #avg_score: the avg similarity score of the correct labels
    biased_actions = biased_actions.sort_values('bg_only_top_1_rate', ascending=False)
    pd.set_option('display.max_rows', 100)  # Set to a number > your row count
    print(biased_actions.to_string(float_format="{:.3f}".format))

    
    # # Least biased actions  
    # print(f"\nLEAST BIASED ACTIONS (actions w/ the least % of predicting correct label in background only):")
    # all_bg_by_action = bg_only.groupby('label').agg({
    #     'correct_ranking': ['count', 'mean'],
    #     'correct_score': 'mean'
    # }).round(3)
    # all_bg_by_action.columns = ['count', 'avg_rank', 'avg_score']
    # all_bg_by_action = all_bg_by_action.sort_values('avg_rank', ascending=False)
    # print(all_bg_by_action.head(10))

if __name__ == "__main__":
    # Analyze dataset - start with a small sample for testing
    csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/hat2_dataset.csv"  # Your CSV file path
    
    results_csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clip_experiments/three_img_comparison_results/three_img_comparison_results_2_and_6.csv"
    results_df = analyze_hat2_dataset(
        csv_path=csv_path,
        output_csv=results_csv_path, 
        sample_size=None  # Test with 50 videos first
    )
    
    # Analyze results
    # if len(results_df) > 0:
    analyze_bias_results(results_csv_path)
        
    print("\nTo run on full dataset, set sample_size=None")