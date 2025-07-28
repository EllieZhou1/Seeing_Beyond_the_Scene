import os
import pandas as pd
import numpy as np
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from collections import defaultdict
import torch

def evaluate_clip_on_mimetics():
    # Load the CSV file
    csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq.csv"
    df = pd.read_csv(csv_path)
    
    # Initialize CLIP model and processor
    print("Loading CLIP model...")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    
    # Results storage
    results = []
    correct_predictions = 0
    top5_correct = 0
    total_samples = 0
    
    # For per-label accuracy
    label_correct = defaultdict(int)
    label_total = defaultdict(int)
    
    print(f"Processing {len(df)} samples...")
    
    for idx, row in df.iterrows():
        try:
            # Get the frame path (50th frame)

            jpg_count = sum(1 for f in os.listdir(row['full_path']) if f.lower().endswith('.jpg'))
            frame_path = os.path.join(row['full_path'], f"{int(jpg_count / 2):06d}.jpg")
            
            # Check if file exists
            if not os.path.exists(frame_path):
                print(f"Frame {frame_path} not found, skipping...")
                continue
            
            # Load the image
            image = Image.open(frame_path).convert('RGB')
            
            # Prepare the choice labels
            choices = [row['choice_1'], row['choice_2'], row['choice_3'], 
                      row['choice_4'], row['choice_5']]
            
            # Process image and text together
            inputs = processor(text=choices, images=[image], return_tensors="pt",
                             padding=True, do_convert_rgb=False)
            
            # Get CLIP predictions
            with torch.no_grad():
                outputs = model(**inputs)
                logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
                probs = logits_per_image.softmax(dim=1)  # convert to probabilities
            
            # Get probability scores for this image
            scores = probs[0].detach().numpy()
            
            # Get CLIP's top choice (1-indexed to match human_choice format)
            clip_choice = np.argmax(scores) + 1
            
            # Get human choice
            human_choice = int(row['human_choice'])
            
            # Calculate accuracies
            total_samples += 1
            
            # Top-1 accuracy
            if clip_choice == human_choice:
                correct_predictions += 1
            
            # Top-5 accuracy (since we only have 5 choices, this is always 1)
            # But we'll calculate it properly for completeness
            sorted_indices = np.argsort(scores)[::-1]  # descending order
            top5_indices = sorted_indices[:5] + 1  # convert to 1-indexed
            if human_choice in top5_indices:
                top5_correct += 1
            
            # Per-label accuracy
            true_label = choices[human_choice - 1]  # Get the actual label text
            label_total[true_label] += 1
            if clip_choice == human_choice:
                label_correct[true_label] += 1
            
            # Store result for CSV output
            result_row = {
                'full_path': row['full_path'],
                'label': row['label'],
                'choice_1': row['choice_1'],
                'choice_2': row['choice_2'],
                'choice_3': row['choice_3'],
                'choice_4': row['choice_4'],
                'choice_5': row['choice_5'],
                'human_choice': human_choice,
                'clip_choice': clip_choice
            }
            results.append(result_row)
            
            if (idx + 1) % 50 == 0:
                print(f"Processed {idx + 1}/{len(df)} samples...")
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate final accuracies
    top1_accuracy = correct_predictions / total_samples if total_samples > 0 else 0
    top5_accuracy = top5_correct / total_samples if total_samples > 0 else 0
    
    # Calculate per-label accuracies
    per_label_accuracies = {}
    for label in label_total:
        per_label_accuracies[label] = label_correct[label] / label_total[label] if label_total[label] > 0 else 0
    
    # Write results to text file

    mimetic_results_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/clip_experiments/mimetics_mcq_results"

    with open(os.path.join(mimetic_results_path, "mimetics_mcq_results.txt"), 'w') as f:
        f.write("CLIP Evaluation Results on Mimetics MCQ Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples processed: {total_samples}\n")
        f.write(f"Top-1 Accuracy: {top1_accuracy:.4f} ({correct_predictions}/{total_samples})\n")
        f.write(f"Top-5 Accuracy: {top5_accuracy:.4f} ({top5_correct}/{total_samples})\n\n")
        
        f.write("Per-Label Top-1 Accuracies:\n")
        f.write("-" * 30 + "\n")
        for label, accuracy in sorted(per_label_accuracies.items()):
            f.write(f"{label}: {accuracy:.4f} ({label_correct[label]}/{label_total[label]})\n")
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(mimetic_results_path, "mimetics_mcq_results.csv"), index=False)
    
    print(f"\nEvaluation complete!")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-5 Accuracy: {top5_accuracy:.4f}")
    print(f"Results saved to 'mimetics_mcq_results.txt' and 'mimetics_mcq_results.csv'")
    
    return top1_accuracy, top5_accuracy, per_label_accuracies

if __name__ == "__main__":
    evaluate_clip_on_mimetics()