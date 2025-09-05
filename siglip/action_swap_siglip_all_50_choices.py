#SIGLIP on Action-Swap (MCQ version - we give all 50 labels choices)

import os
import pandas as pd
import numpy as np
from PIL import Image
from collections import defaultdict
import torch
import requests
from PIL import Image
from transformers import AutoProcessor, AutoModel
from tqdm.auto import tqdm

def evaluate_siglip_on_actionswap():
    # Load the CSV file
    csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq_all_fields.csv"
    df = pd.read_csv(csv_path)
    
    # Initialize CLIP model and processor
    print("Loading SIGLIP model...")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    

    print("Using device", model.device)

    # Results storage
    results = []
    human_correct = 0
    background_correct = 0
    total_samples = 0
    
    print(f"Processing {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows()):
        try:
            # Get the frame path (middle frame)
            frame_path = os.path.join(row['action_swap_path'], f"{int(row['num_files_A'] / 2):06d}.jpg")
            
            # Check if file exists
            if not os.path.exists(frame_path):
                print(f"Frame {frame_path} not found, skipping...")
                continue
            
            # Load the image
            image = Image.open(frame_path).convert('RGB')
            
            # Prepare the choice labels
            choices = [key for key in data]
            
            # Process image and text together
            inputs = processor(text=choices, images=[image], return_tensors="pt",
                             padding="max_length", max_length=64)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Get CLIP predictions
            with torch.no_grad():
                outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image  # shape: (1, num_labels)
            probs = torch.sigmoid(logits_per_image)  # convert to probabilities
            
            # Get probability scores for this image
            scores = probs[0].detach().cpu().numpy()
            
            # Get CLIP's top choice (1-indexed to match human_choice format)
            siglip_choice = np.argmax(scores)
            
            # Get human choice
            human_idx = data[row['label_A']]
            background_idx = data[row['label_B']]
            
            total_samples += 1
            
            # Top-1 accuracy
            if siglip_choice == human_idx:
                human_correct += 1

            # Background Top-1 accuracy
            if siglip_choice == background_idx:
                background_correct += 1
            
            # Store result for CSV output
            result_row = {
                'action_swap_path': row['action_swap_path'],
                'label_A': row['label_A'],
                'label_B': row['label_B'],
                'siglip_choice': idx_to_label[siglip_choice]
            }
            results.append(result_row)
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate final accuracies
    human_accuracy = float(human_correct) / total_samples if total_samples > 0 else 0

    # Calculate background accuracies
    background_accuracy = float(background_correct) / total_samples if total_samples > 0 else 0
    
    # Write results to text file
    actionswap_results_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/siglip/actionswap_mcq_results_all_50_choices"

    os.makedirs(actionswap_results_path, exist_ok=True)

    with open(os.path.join(actionswap_results_path, "actionswap_mcq_results.txt"), 'w') as f:
        f.write("SIGLIP Evaluation Results on Action-Swap Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples processed: {total_samples}\n")

        f.write(f"Human Accuracy: {human_accuracy:.4f}")

        f.write("\n")
        f.write(f"Background Accuracy: {background_accuracy:.4f}")

    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(actionswap_results_path, "actionswap_mcq_results.csv"), index=False)
    
    print(f"\nEvaluation complete!")
    print(f"Human Accuracy: {human_accuracy:.4f}")
    print(f"Background Accuracy: {background_accuracy:.4f}")
    print(f"Results saved to 'actionswap_mcq_results.txt' and 'actionswap_mcq_results.csv'")
    
    return

if __name__ == "__main__":
    import json

    # Load JSON file
    with open('/n/fs/visualai-scr/temp_LLP/ellie/50class_to_label.json', 'r') as f:
        data = json.load(f)

    idx_to_label = {v: k for k, v in data.items()}

    evaluate_siglip_on_actionswap()
