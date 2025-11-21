import json
import os

root = os.getcwd()
models = {
    "gpt": "gpt-4.1_sequential_eval.json",
    "llama": "llama_sequential_eval.json",
    "deepseek": "deepseek_sequential_eval.json"
}

# length gte 3, check alternating
def is_alternating(seq):
    if len(seq) < 3:
        return False
    if seq[1] == seq[0]:
        return False
    up = seq[1] > seq[0]
    for i in range(2, len(seq)):
        if seq[i] == seq[i-1]:
            return False
        if up and seq[i] <= seq[i-1]:
            return False
        if not up and seq[i] >= seq[i-1]:
            return False
        up = not up
    return True


def is_monotonic_inc(seq):
    return all(seq[i] >= seq[i-1] for i in range(1, len(seq)))

def is_monotonic_dec(seq):
    return all(seq[i] <= seq[i-1] for i in range(1, len(seq)))

def is_flat(seq):
    return len(set(seq)) == 1

final_output = {}

for model_name, filename in models.items():
    path = os.path.join(root, "sequential_data", filename)
    with open(path, "r") as f:
        data = json.load(f)

    prompt_data = data["prompt_level_part_accuracy"]

    result = {
        "alternating_prompts": [],
        "monotonic_increasing_prompts": [],
        "monotonic_decreasing_prompts": [],
        "flat_prompts": [],
        "other_prompts": []
    }

    for pid, parts in prompt_data.items():
        seq = [parts[str(i)] for i in sorted(map(int, parts.keys()))]
        if is_flat(seq):
            result["flat_prompts"].append(pid)
        elif is_alternating(seq):
            result["alternating_prompts"].append(pid)
        elif is_monotonic_inc(seq):
            result["monotonic_increasing_prompts"].append(pid)
        elif is_monotonic_dec(seq):
            result["monotonic_decreasing_prompts"].append(pid)
        else:
            result["other_prompts"].append(pid)

    result["counts"] = {
        "alternating": len(result["alternating_prompts"]),
        "monotonic_increasing": len(result["monotonic_increasing_prompts"]),
        "monotonic_decreasing": len(result["monotonic_decreasing_prompts"]),
        "flat": len(result["flat_prompts"]),
        "other": len(result["other_prompts"])
    }

    final_output[model_name] = result

    per_model_path = os.path.join(root, "sequential_data", f"{model_name}_pattern.json")
    with open(per_model_path, "w") as f:
        json.dump(result, f, indent=2)

combined_path = os.path.join(root, "sequential_data", "pattern_analysis.json")
with open(combined_path, "w") as f:
    json.dump(final_output, f, indent=2)

print(json.dumps(final_output, indent=2))
