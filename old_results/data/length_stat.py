import json
import os
import csv

# Input file
input_file = "deepseek-v3_outputs.jsonl"

# Extract model name
base = os.path.basename(input_file)
model_name = base.replace("_outputs.jsonl", "")

# Output file
output_file = f"{model_name}_Prompt_Avg_Length.csv"

# Storage
prompt_lengths = {}

# Read jsonl file
with open(input_file, "r", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        prompt_id = int(record["prompt_id"])
        output_text = record["text"]

        # Count words
        word_count = len(output_text.split())

        # Save by prompt
        if prompt_id not in prompt_lengths:
            prompt_lengths[prompt_id] = []
        prompt_lengths[prompt_id].append(word_count)

# Compute averages (rounded to integer)
prompt_avg = {pid: round(sum(lengths)/len(lengths)) for pid, lengths in prompt_lengths.items()}

# Write CSV
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Group", "Prompt ID", "Average Word Count"])

    for group_idx in range(10):   # 10 groups of 16 prompts
        start = group_idx * 16 + 1
        end = start + 15
        group_name = f"Group {group_idx+1}"

        # Group header
        writer.writerow([group_name, "", ""])

        # Prompts in this group
        for pid in range(start, end+1):
            avg_len = prompt_avg.get(pid, 0)
            writer.writerow([group_name, f"Prompt {pid}", avg_len])

print(f"Saved to: {output_file}")

