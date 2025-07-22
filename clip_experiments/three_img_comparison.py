import os
import PIL.Image as Image
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt

base_dir = "/n/fs/visualai-scr/Data/HAT2"
# original_path = os.path.join(base_dir, "original", "archery", "0S-P4lr_c7s_000022_000032", "000007.jpg")
# seg_path = os.path.join(base_dir, "seg", "archery", "0S-P4lr_c7s_000022_000032", "000007.png")
# bg_path = os.path.join(base_dir, "inpaint", "archery", "0S-P4lr_c7s_000022_000032", "000007.jpg")


correct_label = "juggling balls"
youtube_id = "C-AOi2aAPvc_000041_000051"
frame_num = 50

original_path = os.path.join(base_dir, "original", correct_label, youtube_id, f"{frame_num:06d}.jpg")
seg_path = os.path.join(base_dir, "seg", correct_label, youtube_id, f"{frame_num:06d}.png")
original_path = os.path.join(base_dir, "inpaint", correct_label, youtube_id, f"{frame_num:06d}.jpg")

# Load CLIP model
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("Loaded CLIP model")

def test_single_action(original_path, seg_path, background_only_path, 
                      all_labels, correct_label):
    """
    Test bias for one action with 3 image versions
    
    Args:
        original_path: path to original image
        human_only_path: path to human-only image  
        background_only_path: path to background-only image
        all_labels: list of all 50 action labels
        correct_label: the correct label for this action
    
    Returns:
        Dictionary with rankings and scores
    """
    
    # Load the 3 images
    images = []

    original = np.array(Image.open(original_path).convert('RGB'))  
    seg = np.array(Image.open(seg_path).convert('L'))  # Binary mask
    inpaint = np.array(Image.open(background_only_path).convert('RGB')) 

    seg_norm = seg.astype(np.float32) / 255.0
    # print("seg min and max", seg_norm.min(), seg_norm.max())

    mask_3d = np.stack([seg_norm, seg_norm, seg_norm], axis=2)
    human_only = (mask_3d * original).astype(np.uint8)

    # print("original min and max", original.min(), original.max())
    # print("inpaint min and max", inpaint.min(), inpaint.max())
    # print("seg_norm min and max", seg_norm.min(), seg_norm.max())
    # print("human only min and max", human_only.min(), human_only.max())

    # plt.imshow(human_only)
    # plt.axis('off')  # hides axis
    # plt.show()

    images.append(original)
    images.append(human_only)
    images.append(inpaint)
    
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
        
        # Create ranked list of actions with scores
        ranked_actions = []
        for rank, idx in enumerate(ranking_indices):
            ranked_actions.append({
                'rank': rank + 1,
                'action': all_labels[idx],
                'score': float(scores[idx])
            })
        
        # Find where correct label ranks
        correct_ranking = np.where(ranking_indices == correct_idx)[0][0] + 1
        correct_score = scores[correct_idx].item()
        
        results[img_type] = {
            'correct_ranking': int(correct_ranking), #The ranking of the correct text label among all labels
            'correct_score': float(correct_score), #Similarity score of the img & the correct text label
            'full_rankings': ranked_actions, #Full ranking of all 50 labels, ranked by similarity
            'top5': correct_ranking <= 5 #Whether or not the correct ranking was in the top 5 or not
        }
    
    return results

# Example usage:
if __name__ == "__main__":
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
    
    # Test one action
    result = test_single_action(
        original_path=original_path,
        seg_path=seg_path, 
        background_only_path=bg_path,
        all_labels=all_labels,
        correct_label=correct_label
    )
    
    # Print results
    print(f"Results for {correct_label}:")
    
    for img_type in ['original', 'human_only', 'background_only']:
        print(f"\n{img_type.upper()}:")
        print(f"  Correct action rank: {result[img_type]['correct_ranking']}/50 (score: {result[img_type]['correct_score']:.3f})")
        print(f"  Top 5 predictions:")
        for item in result[img_type]['full_rankings'][:5]:
            marker = "👉" if item['action'] == "playing tennis" else "  "
            print(f"    {marker} {item['rank']:2d}. {item['action']} ({item['score']:.3f})")
    
    # Check for bias
    if result['background_only']['top5']:
        print("\n⚠️  POTENTIAL BIAS: Background-only image ranks in top 5!")
    else:
        print("\n✅ Good: Background-only image doesn't rank highly")

