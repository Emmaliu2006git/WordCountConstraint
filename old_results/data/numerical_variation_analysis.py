import json
import numpy as np
import os
import csv
from collections import Counter

# ===== Input =====
input_file = "deepseek-v3_evaluation_result.json"

# ===== Extract model name =====
base = os.path.basename(input_file)
model_name = base.replace("_evaluation_result.json", "")

# ===== Output file =====
output_file = f"{model_name}_Numerical_Satisfaction_Variation.csv"

# ===== Load JSON =====
with open(input_file, "r") as f:
    data = json.load(f)

detail = data["detail"]

# ===== Extract rates =====
numerical_rates = [detail[str(i)]["numerical_satisfaction_rate"] for i in range(1, 161)]

std_list = []
max_positions = []
min_positions = []

# ===== Write CSV =====
with open(output_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["Prompt Group", "Metric", "Value", "Position (if applicable)"])

    # --- Each group ---
    for group_idx in range(10):
        start = group_idx * 16
        end = start + 16
        group = numerical_rates[start:end]

        arr = np.array(group)
        std = arr.std(ddof=0)

        max_val = arr.max()
        min_val = arr.min()
        max_pos = (arr.argmax() % 16) + 1
        min_pos = (arr.argmin() % 16) + 1

        group_name = f"Prompt {group_idx+1}"

        # Save for summary
        std_list.append((group_name, std))
        max_positions.append(max_pos)
        min_positions.append(min_pos)

        # Write group results
        writer.writerow([group_name, "STD", round(std, 4), ""])
        writer.writerow([group_name, "Max Value", round(max_val, 4), max_pos])
        writer.writerow([group_name, "Min Value", round(min_val, 4), min_pos])
        writer.writerow([])

    # --- Summary ---
    writer.writerow(["Summary"])

    # Top 3 most frequent max/min positions
    max_common = Counter(max_positions).most_common(3)
    min_common = Counter(min_positions).most_common(3)

    writer.writerow(["Top 3 Max Positions"])
    for pos, count in max_common:
        writer.writerow([f"Position {pos}", f"{count} times"])

    writer.writerow([])
    writer.writerow(["Top 3 Min Positions"])
    for pos, count in min_common:
        writer.writerow([f"Position {pos}", f"{count} times"])

    writer.writerow([])
    writer.writerow(["Top 3 STD Groups"])
    top3_std = sorted(std_list, key=lambda x: x[1], reverse=True)[:3]
    for group_name, val in top3_std:
        writer.writerow([group_name, "STD", round(val, 4)])

print(f"Saved to: {output_file}")
