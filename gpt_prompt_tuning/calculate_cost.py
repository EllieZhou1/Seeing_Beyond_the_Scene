import json

with open("4o-mini_manual_prompt_1_75_mini_actionswap/filtered_prompts_results.json", "r") as f:
    data = json.load(f)

prompt_tokens = 0
completion_tokens = 0

for key, entry in data.items():   # key = filepath string, entry = dict
    print("key:", key)
    # if entry["flagged"] == False:
    prompt_tokens += entry["prompt_tokens"]
    completion_tokens += entry["completion_tokens"]

print("Prompt tokens:", prompt_tokens)
print("Completion tokens:", completion_tokens)
