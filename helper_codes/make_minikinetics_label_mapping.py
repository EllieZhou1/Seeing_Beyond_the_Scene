import json

labels = [
    'playing guitar', 'bowling', 'playing saxophone', 'brushing teeth', 'playing basketball',
    'tying tie', 'skiing slalom', 'brushing hair', 'punching person (boxing)', 'playing accordion',
    'archery', 'catching or throwing frisbee', 'drinking', 'reading book', 'eating ice cream',
    'flying kite', 'sweeping floor', 'walking the dog', 'skipping rope', 'clean and jerk',
    'eating cake', 'catching or throwing baseball', 'skiing (not slalom or crosscountry)',
    'juggling soccer ball', 'deadlifting', 'driving car', 'cleaning windows', 'shooting basketball',
    'canoeing or kayaking', 'surfing water', 'playing volleyball', 'opening bottle', 'playing piano',
    'writing', 'dribbling basketball', 'reading newspaper', 'playing violin', 'juggling balls',
    'playing trumpet', 'smoking', 'shooting goal (soccer)', 'hitting baseball', 'sword fighting',
    'climbing ladder', 'playing bass guitar', 'playing tennis', 'climbing a rope',
    'golf driving', 'hurdling', 'dunking basketball'
]

# Map each label to a unique number from 0 to 49
label_map = {label: idx for idx, label in enumerate(labels)}

# Save to a JSON file
with open("/n/fs/visualai-scr/temp_LLP/ellie/slowfast_kinetics/class50_to_label.json", "w") as f:
    json.dump(label_map, f, indent=4)

print("finished")