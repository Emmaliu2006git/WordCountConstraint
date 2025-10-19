import json
import os
from collections import defaultdict
from helper import *

def ok(count: int, v: dict) -> bool:
    rel = v["relation"]
    if rel == "gte":
        return count >= v["target"]
    if rel == "lte":
        return count <= v["target"]
    if rel == "approx":
        return abs(count - v["target"]) <= 0.15 * v["target"]
    if rel == "range":
        return v["lower"] <= count <= v["upper"]
    raise ValueError(f"Unknown relation: {rel}")

def check_word_count(text: str, verification: dict) -> bool:
    words = text.strip().split()
    count = len(words)
    return ok(count, verification)

def constraint_to_func(constraint_type: str) -> str:
    second = constraint_type.split("+")[-1].strip().lower()
    second = second.replace(" ", "_").replace("-", "_")
    return "check_" + second

def evaluate_sample(text: str, verification: list, constraint_type: str):
    num_check = check_word_count(text, verification[0])
    func_name = constraint_to_func(constraint_type)
    if func_name not in globals():
        raise ValueError(f"No function found for {constraint_type}, expected {func_name}")
    try:
        add_check = globals()[func_name](text, verification[1])
    except Exception as e:
        print("ERROR at constraint:", constraint_type)
        print("BAD verification[1]:", verification[1])
        raise
    score = (0.5 if num_check else 0) + (0.5 if add_check else 0)
    return num_check, add_check, score

def evaluate_model(input_file, output_file, mkey):
    verification_map = {}
    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            row = json.loads(line)
            verification_map[idx] = row["verification"]

    prompt_results = defaultdict(list)
    constraint_map = {}

    with open(output_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            line = line.lstrip("\ufeff").strip()
            if not line:
                continue
            data = json.loads(line)
            raw_pid = data.get("prompt_id")
            pid = raw_pid if isinstance(raw_pid, int) and raw_pid >= 1 else ((idx - 1) // 8) + 1
            text = data["text"]
            constraint_type = data["constraint"]
            verification = verification_map[pid]
            num_check, add_check, score = evaluate_sample(text, verification, constraint_type)
            prompt_results[pid].append({
                "numerical": num_check,
                "additional": add_check,
                "score": score
            })
            constraint_map[pid] = constraint_type

    prompt_summary = {}
    for pid, samples in prompt_results.items():
        n = len(samples)
        dual = sum(1 for s in samples if s["numerical"] and s["additional"]) / n
        numerical = sum(1 for s in samples if s["numerical"]) / n
        additional = sum(1 for s in samples if s["additional"]) / n
        acc = sum(s["score"] for s in samples) / n
        prompt_summary[pid] = {
            "dual_satisfaction_rate": dual,
            "numerical_satisfaction_rate": numerical,
            "additional_satisfaction_rate": additional,
            "per_prompt_accuracy": acc,
            "constraint_type": constraint_map[pid]
        }

    type_summary = defaultdict(list)
    for pid, vals in prompt_summary.items():
        type_summary[vals["constraint_type"]].append(vals)

    constraint_summary = {}
    for t, items in type_summary.items():
        constraint_summary[t] = {
            "dual_satisfaction_rate": sum(i["dual_satisfaction_rate"] for i in items) / len(items),
            "numerical_satisfaction_rate": sum(i["numerical_satisfaction_rate"] for i in items) / len(items),
            "additional_satisfaction_rate": sum(i["additional_satisfaction_rate"] for i in items) / len(items),
            "performance_accuracy": sum(i["per_prompt_accuracy"] for i in items) / len(items)
        }

    model_level = {
        "dual_satisfaction_rate": sum(v["dual_satisfaction_rate"] for v in prompt_summary.values()) / len(prompt_summary),
        "numerical_satisfaction_rate": sum(v["numerical_satisfaction_rate"] for v in prompt_summary.values()) / len(prompt_summary),
        "additional_satisfaction_rate": sum(v["additional_satisfaction_rate"] for v in prompt_summary.values()) / len(prompt_summary),
        "overall_accuracy": sum(v["per_prompt_accuracy"] for v in prompt_summary.values()) / len(prompt_summary),
    }

    result_file = f"{mkey}_evaluation_result.json"
    output = {
        "model_level": model_level,
        "constraint_type": constraint_summary,
        "detail": {
            pid: {
                "dual_satisfaction_rate": v["dual_satisfaction_rate"],
                "numerical_satisfaction_rate": v["numerical_satisfaction_rate"],
                "additional_satisfaction_rate": v["additional_satisfaction_rate"],
                "per_prompt_accuracy": v["per_prompt_accuracy"]
            }
            for pid, v in prompt_summary.items()
        }
    }

    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Saved result for {mkey} -> {result_file}")

def main():
    input_file = "prompts_complete.jsonl"
    models = ["llama4scout", "deepseek-v3", "gpt-4.1"]
    for mkey in models:
        output_file = f"{mkey}_outputs.jsonl"
        if not os.path.exists(output_file):
            print(f"Skip {mkey}, file {output_file} not found")
            continue
        evaluate_model(input_file, output_file, mkey)

if __name__ == "__main__":
    main()
