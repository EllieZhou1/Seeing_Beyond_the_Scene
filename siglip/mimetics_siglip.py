#CLIP on Action-Swap (MCQ version - we only give 5 label choices)

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

def evaluate_siglip_on_mimetics():
    # Load the CSV file
    csv_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/mimetics/mimetics_mcq_all_fields.csv"
    df = pd.read_csv(csv_path)
    
    # Initialize CLIP model and processor
    print("Loading SIGLIP model...")
    model = AutoModel.from_pretrained("google/siglip2-base-patch16-224", torch_dtype=torch.float16, device_map="auto", attn_implementation="sdpa")
    processor = AutoProcessor.from_pretrained("google/siglip2-base-patch16-224")
    

    print("Using device", model.device)

    # Results storage
    results = []
    human_correct = 0
    total_samples = 0
    
    print(f"Processing {len(df)} samples...")
    
    for idx, row in tqdm(df.iterrows()):
        try:
            # Get the frame path (middle frame)
            frame_path = os.path.join(row['full_path'], f"{int(row['num_files'] / 2):06d}.jpg")
            
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
            siglip_choice = np.argmax(scores) + 1
            
            # Get human choice
            human_choice = int(row['human_choice'])
            total_samples += 1
            
            # Top-1 accuracy
            if siglip_choice == human_choice:
                human_correct += 1
            
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
                'siglip_choice': siglip_choice
            }
            results.append(result_row)
                
        except Exception as e:
            print(f"Error processing sample {idx}: {str(e)}")
            continue
    
    # Calculate final accuracies
    human_accuracy = float(human_correct) / total_samples if total_samples > 0 else 0
    
    # Write results to text file
    mimetics_results_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/siglip/mimetics_mcq_results"

    os.makedirs(mimetics_results_path, exist_ok=True)

    with open(os.path.join(mimetics_results_path, "mimetics_mcq_results.txt"), 'w') as f:
        f.write("SIGLIP Evaluation Results on Mimetics MCQ Dataset\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total samples processed: {total_samples}\n")

        f.write(f"Human Accuracy: {human_accuracy:.4f}")

    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(mimetics_results_path, "mimetics_mcq_results.csv"), index=False)
    
    print(f"\nEvaluation complete!")
    print(f"Human Accuracy: {human_accuracy:.4f}")
    print(f"Results saved to 'mimetics_mcq_results.txt' and 'mimetics_mcq_results.csv'")
    
    return

if __name__ == "__main__":
    evaluate_siglip_on_mimetics()