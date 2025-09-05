import pandas as pd

# Load the CSV file
df = pd.read_csv("dataset/action_swap/action_swap_mcq_all_fields.csv")

# Randomly sample 25%
sample_df = df.sample(frac=0.25, random_state=42)  # random_state for reproducibility

# Get the remaining 75%
remaining_df = df.drop(sample_df.index)

# Save to new CSV files
sample_df.to_csv("dataset/action_swap/mini_action_swap_mcq_25.csv", index=False)
remaining_df.to_csv("dataset/action_swap/mini_action_swap_75.csv", index=False)

print(f"Sample size: {len(sample_df)} rows")
print(f"Remaining size: {len(remaining_df)} rows")