import pandas as pd
import random
from typing import List

def create_mcq_csv(input_path: str, output_path: str):
    """
    Transform action-swap CSV into multiple choice question format.
    
    Args:
        input_path: Path to the original CSV file
        output_path: Path where the new CSV will be saved
    """
    
    # List of possible action labels
    all_labels = [
        'playing guitar', 'bowling', 'playing saxophone', 'brushing teeth', 
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
        'golf driving', 'hurdling', 'dunking basketball'
    ]
    
    # Read the original CSV
    df = pd.read_csv(input_path)
    
    # Initialize lists to store the new data
    new_data = []
    
    for _, row in df.iterrows():
        action_swap_path = row['action_swap_path']
        label_A = row['label_A']
        label_B = row['label_B']
        
        # Create a list of available labels (excluding label_A and label_B)
        available_labels = [label for label in all_labels if label not in [label_A, label_B]]
        
        # Randomly select 3 distractor labels
        distractors = random.sample(available_labels, 3)
        
        # Create the list of 5 choices (label_A, label_B, and 3 distractors)
        choices = [label_A, label_B] + distractors
        
        # Randomly shuffle the choices so label_A and label_B don't always appear in the same positions
        random.shuffle(choices)
        
        # Find which choice positions contain label_A and label_B
        human_choice_num = choices.index(label_A) + 1  # +1 because choices are 1-indexed
        background_choice_num = choices.index(label_B) + 1  # +1 because choices are 1-indexed
        
        # Create the new row
        new_row = {
            'action_swap_path': action_swap_path,
            'label_A': label_A,
            'label_B': label_B,
            'choice_1': choices[0],
            'choice_2': choices[1],
            'choice_3': choices[2],
            'choice_4': choices[3],
            'choice_5': choices[4],
            'human_choice': human_choice_num,
            'background_choice': background_choice_num
        }
        
        new_data.append(new_row)
    
    # Create new DataFrame
    new_df = pd.DataFrame(new_data)
    
    # Save to CSV
    new_df.to_csv(output_path, index=False)
    print(f"Successfully created MCQ CSV with {len(new_df)} rows at: {output_path}")
    
    # Display first few rows as preview
    print("\nPreview of the generated CSV:")
    print(new_df.head())

if __name__ == "__main__":
    # Set random seed for reproducibility (optional)
    random.seed(42)
    
    # File paths
    input_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_all.csv"
    output_path = "/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/dataset/action_swap/action_swap_mcq.csv"
    
    # Create the MCQ CSV
    create_mcq_csv(input_path, output_path)