import json
import os
from collections import defaultdict

root = os.getcwd()

model_files = {
    "gpt-4.1": "gpt-4.1_output_eval.json",
    "llama": "llama4scout_output_eval.json",
    "deepseek": "deepseek-v3_output_eval.json",
}

for model, filename in model_files.items():
    input_path = os.path.join(root, "eval", filename)
    output_dir = os.path.join(root, "sequential_data")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{model}_sequential_eval.json")

    with open(input_path, "r") as f:
        data = json.load(f)

    records = data["per_output_records"]

    prompt_parts_scores = defaultdict(lambda: defaultdict(list))

    for rec in records:
        pid = rec["prompt_id"]
        parts = rec["part_results"]
        for part_id, part_info in parts.items():
            prompt_parts_scores[pid][int(part_id)].append(part_info["scores"]["hard"])

    prompt_averages = {}
    max_part_index = 0

    for pid, parts in prompt_parts_scores.items():
        part_avg = {}
        for part_id, scores in parts.items():
            part_avg[part_id] = sum(scores) / len(scores)
            if part_id > max_part_index:
                max_part_index = part_id
        prompt_averages[pid] = part_avg

    final_avg = {}
    for part_idx in range(1, max_part_index + 1):
        collected = []
        for pid in prompt_averages:
            if part_idx in prompt_averages[pid]:
                collected.append(prompt_averages[pid][part_idx])
        if collected:
            final_avg[part_idx] = sum(collected) / len(collected)

    output = {
        "prompt_level_part_accuracy": prompt_averages,
        "final_sequential_accuracy": final_avg
    }

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
