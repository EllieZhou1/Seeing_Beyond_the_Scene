# CLIP on Action-Swap (non-MCQ version - we give all 50 labels)

import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm

# Load CLIP model once at the start
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Loaded CLIP model")

def save_detailed_results_to_txt(result, action_swap_path, label_A, label_B, 
                                output_dir="/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clip_experiments/action_swap_results_center_frame/per-video_results"):
    """
    Save detailed results for one action-swap image to a txt file
    
    Args:
        result: Results dictionary from test_action_swap_image
        action_swap_path: Path to action-swap image
        label_A: Human action label
        label_B: Background action label
        output_dir: Directory to save txt files
    """
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create filename from directory name (exactly as it appears)
    dir_name = os.path.basename(action_swap_path.rstrip('/'))
    filename = f"{dir_name}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w') as f:
        f.write(f"Action-Swap CLIP Bias Analysis Results\n")
        f.write(f"=" * 60 + "\n")
        f.write(f"Video Directory: {action_swap_path}\n")
        f.write(f"Frame: MIDDLE.jpg\n")
        f.write(f"Human Action Label (Correct): {label_A}\n") 
        f.write(f"Background Label (Incorrect): {label_B}\n\n")
        
        f.write(f"CLIP PREDICTIONS:\n")
        f.write(f"  Predicted Label: {result['predicted_label']} (score: {result['predicted_score']:.4f})\n\n")
        
        f.write(f"RANKING ANALYSIS:\n")
        f.write(f"  Human Action '{label_A}' ranked: #{result['human_action_rank']}/50 (score: {result['human_action_score']:.4f})\n")
        f.write(f"  Background Action '{label_B}' ranked: #{result['background_action_rank']}/50 (score: {result['background_action_score']:.4f})\n\n")
        
        f.write(f"Top 10 predictions:\n")
        for i, item in enumerate(result['top10_predictions']):
            marker = ""
            if item['action'] == label_A:
                marker = "👤 "
            elif item['action'] == label_B:
                marker = "🏞️ "
            else:
                marker = "   "
            f.write(f"  {marker}{item['rank']:2d}. {item['action']} ({item['score']:.4f})\n")
        
        # Add bias assessment
        f.write(f"\nBIAS ASSESSMENT:\n")
        if result['background_wins']:
            f.write("🚨 BIAS DETECTED: Background action ranks higher than human action!\n")
        else:
            f.write("✅ No bias: Human action ranks higher than background action\n")
            
        f.write(f"Ranking gap (bg_rank - human_rank): {result['ranking_gap']}\n")
        f.write(f"Score gap (bg_score - human_score): {result['score_gap']:.4f}\n")
        
        if result['background_top5']:
            f.write("⚠️  Background action in top 5!\n")
        if result['human_top5']:
            f.write("✅ Human action in top 5\n")

def test_action_swap_image(action_swap_dir, all_labels, label_A, label_B, frame_num):
    """
    Test bias for one action-swap video (using specific frame)
    
    Args:
        action_swap_dir: Path to action-swap video directory
        all_labels: List of all action labels
        label_A: Human action label (correct)
        label_B: Background action label (should be incorrect)
        frame_num: Which frame to analyze (default 50)
    
    Returns:
        Dictionary with bias metrics, or None if error
    """
    
    try:
        # Get path to specific frame
        # print("frame num", frame_num)
        frame_path = os.path.join(action_swap_dir, f"{frame_num:06d}.jpg")
        
        # Check if file exists
        if not os.path.exists(frame_path):
            print(f"Frame {frame_path} not found")
            return None
        
        # Load the action-swap image
        image = Image.open(frame_path).convert('RGB')
        
        # Process image and text together
        inputs = processor(text=all_labels, images=[image], return_tensors="pt", 
                          padding=True, do_convert_rgb=False)
        
        # Get CLIP predictions
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
        probs = logits_per_image.softmax(dim=1)  # convert to probabilities
        
        # Get probability scores for this image
        scores = probs[0].detach().numpy()
        
        # Get full ranking order (indices sorted by score, descending)
        ranking_indices = np.argsort(scores)[::-1]
        
        # Create ranked list of actions with scores (save top 10)
        ranked_actions = []
        for rank, idx in enumerate(ranking_indices[:10]):
            ranked_actions.append({
                'rank': rank + 1,
                'action': all_labels[idx],
                'score': float(scores[idx])
            })
        
        # Find indices of human and background labels
        human_idx = all_labels.index(label_A)
        background_idx = all_labels.index(label_B)
        
        # Find rankings
        human_rank = np.where(ranking_indices == human_idx)[0][0] + 1
        background_rank = np.where(ranking_indices == background_idx)[0][0] + 1
        
        # Get scores
        human_score = scores[human_idx].item()
        background_score = scores[background_idx].item()
        
        # Get top prediction
        predicted_label = all_labels[ranking_indices[0]]
        predicted_score = scores[ranking_indices[0]].item()
        
        result = {
            'predicted_label': predicted_label,
            'predicted_score': float(predicted_score),
            'human_action_rank': int(human_rank),
            'human_action_score': float(human_score),
            'background_action_rank': int(background_rank),
            'background_action_score': float(background_score),
            'top10_predictions': ranked_actions,
            'human_top1': human_rank == 1,
            'human_top5': human_rank <= 5,
            'background_top1': background_rank == 1,
            'background_top5': background_rank <= 5,
            'background_wins': background_rank < human_rank,
            'ranking_gap': background_rank - human_rank,
            'score_gap': background_score - human_score
        }
        
        return result
        
    except Exception as e:
        print(f"Error processing video directory {action_swap_dir}: {e}")
        return None

def analyze_action_swap_dataset(csv_path, output_csv, sample_size=None):
    """
    Analyze action-swap dataset for CLIP bias
    
    Args:
        csv_path: Path to action-swap CSV file
        output_csv: Output CSV filename
        sample_size: If specified, randomly sample this many rows for testing
    """
    
    # Load dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded action-swap dataset with {len(df)} entries")
    
    # Filter to only rows that have action_swap_path and it's a directory
    valid_df = df[df['action_swap_path'].notna()]
    # Additional check that paths exist and are directories
    valid_paths = []
    for path in valid_df['action_swap_path']:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if frame 50 exists
            frame_path = os.path.join(path, "000050.jpg")
            if os.path.exists(frame_path):
                valid_paths.append(True)
            else:
                valid_paths.append(False)
        else:
            valid_paths.append(False)
    
    valid_df = valid_df[valid_paths]
    print(f"Found {len(valid_df)} entries with valid action_swap directories and frame 50")
    
    # Sample if requested
    if sample_size and sample_size < len(valid_df):
        valid_df = valid_df.sample(n=sample_size, random_state=42)
        print(f"Sampling {sample_size} entries for analysis")
    
    # HAT action labels (same as your reference code)
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
    
    # Process each action-swap image
    for idx, row in tqdm(valid_df.iterrows(), total=len(valid_df), desc="Analyzing action-swap images"):
        
        jpg_count = len([f for f in os.listdir(row['action_swap_path']) if f.lower().endswith('.jpg')])


    
        result = test_action_swap_image(
            action_swap_dir=row['action_swap_path'],
            all_labels=all_labels,
            label_A=row['label_A'],  # Human action
            label_B=row['label_B'],  # Background action
            frame_num=int(jpg_count/2)
        )
        
        if result is None:
            print(f"Skipping {row['action_swap_path']}: processing error")
            continue
        
        # Save detailed results to txt file
        save_detailed_results_to_txt(
            result=result,
            action_swap_path=row['action_swap_path'],
            label_A=row['label_A'],
            label_B=row['label_B']
        )
        
        # Create row for CSV
        results_rows.append({
            'action_swap_path': row['action_swap_path'],
            'frame_num': jpg_count/2,
            'label_A': row['label_A'],  # Human action
            'label_B': row['label_B'],  # Background action
            'predicted_label': result['predicted_label'],
            'predicted_score': result['predicted_score'],
            'human_action_rank': result['human_action_rank'],
            'human_action_score': result['human_action_score'],
            'background_action_rank': result['background_action_rank'],
            'background_action_score': result['background_action_score'],
            'human_top1': result['human_top1'],
            'human_top5': result['human_top5'],
            'background_top1': result['background_top1'],
            'background_top5': result['background_top5'],
            'background_wins': result['background_wins'],
            'ranking_gap': result['ranking_gap'],
            'score_gap': result['score_gap'],
            'top10_predictions': str([p['action'] for p in result['top10_predictions']])
        })
    
    # Save results
    results_df = pd.DataFrame(results_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(results_df)} results to {output_csv}")
    
    return results_df

def analyze_action_swap_bias_results(results_csv):
    """
    Analyze the action-swap bias results and print summary statistics
    """
    df = pd.read_csv(results_csv)
    
    print("=== ACTION-SWAP CLIP BIAS ANALYSIS ===\n")
    
    # Overall statistics
    print("OVERALL PERFORMANCE METRICS:")
    print(f"  Human label top-1: {df['human_top1'].mean():.3%}")
    print(f"  Human label top-5: {df['human_top5'].mean():.3%}")
    print(f"  Background label top-1: {df['background_top1'].mean():.1%}")
    print(f"  Background label top-5: {df['background_top5'].mean():.1%}")
    print(f"  How Often Background wins (ranks higher): {df['background_wins'].mean():.1%}")
    
    print(f"\nRANKING STATISTICS:")
    print(f"  Mean human action rank: {df['human_action_rank'].mean():.1f}")
    print(f"  Mean background action rank: {df['background_action_rank'].mean():.1f}")
    print(f"  Mean ranking gap (bg - human): {df['ranking_gap'].mean():.1f}")
    
    print(f"\nSCORE STATISTICS:")
    print(f"  Mean human action score: {df['human_action_score'].mean():.3f}")
    print(f"  Mean background action score: {df['background_action_score'].mean():.3f}")
    print(f"  Mean score gap (bg - human): {df['score_gap'].mean():.3f}")
    
    # Metrics grouped by human action (label_A)
    human_bias = df.groupby('label_A').agg({
        'background_top1': 'mean',
        'human_top1': 'mean',
        'background_wins': 'mean',
        'human_action_rank': ['count', 'mean'],
        'background_action_rank': 'mean'
    }).round(3)
    human_bias.columns = ['bg_top1_rate', 'human_top1_rate', 'bg_wins_rate', 'count', 'human_avg_rank', 'bg_avg_rank']
    human_bias_sortbg = human_bias.sort_values('bg_top1_rate', ascending=False)
    human_bias_sorthuman = human_bias.sort_values('human_top1_rate', ascending=False)
    
    pd.set_option('display.max_rows', 100)
    print(f"\nBIAS BY HUMAN ACTION LABEL (Highest bg accuracy):")
    print(human_bias_sortbg.to_string(float_format="{:.3f}".format))

    print(f"\nBIAS BY HUMAN ACTION LABEL (Highest human accuracy):")
    print(human_bias_sorthuman.to_string(float_format="{:.3f}".format))

    
    # Metrics grouped by background action (label_B)
    bg_bias = df.groupby('label_B').agg({
        'background_top1': 'mean',
        'human_top1': 'mean', 
        'background_wins': 'mean',
        'background_action_rank': ['count', 'mean'],
        'human_action_rank': 'mean'
    }).round(3)
    bg_bias.columns = ['bg_top1_rate', 'human_top1_rate', 'bg_wins_rate', 'count', 'bg_avg_rank', 'human_avg_rank']
    bg_bias_sortbg = bg_bias.sort_values('bg_top1_rate', ascending=False)
    bg_bias_sorthuman = bg_bias.sort_values('human_top1_rate', ascending=False)

    print(f"\nBIAS GROUPED BY BACKGROUND LABEL (Highest bg accuracy)")
    print(bg_bias_sortbg.to_string(float_format="{:.3f}".format))

    print(f"\nBIAS GROUPED BY BACKGROUND LABEL (Highest human accuracy)")
    print(bg_bias_sorthuman.to_string(float_format="{:.3f}".format))

if __name__ == "__main__":
    # Paths
    csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/new_action_swap_all.csv"
    output_csv = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clip_experiments/action_swap_results_center_frame/action_swap_results.csv"
    
    # Create output directory if it doesn't exist
    output_dir = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clip_experiments/action_swap_results_center_frame"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "per-video_results"), exist_ok=True)
    
    # # Test with small sample first
    # print("Testing with small sample...")
    # sample_output = output_csv.replace('.csv', '_sample.csv')
    # results_df = analyze_action_swap_dataset(
    #     csv_path=csv_path,
    #     output_csv=sample_output,
    #     sample_size=50  # Test with 50 videos first
    # )
    
    # # Analyze results
    # if len(results_df) > 0:
    #     analyze_action_swap_bias_results(sample_output)
        
    # print("\nTo run on full dataset, set sample_size=None")
    
    # Uncomment to run on full dataset:
    results_df = analyze_action_swap_dataset(csv_path, output_csv, sample_size=None)
    analyze_action_swap_bias_results(output_csv)