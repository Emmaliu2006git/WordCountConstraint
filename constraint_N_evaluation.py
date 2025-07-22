import json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data/position_test_1"   # your data directory
PROMPTS_FILE = DATA / "prompts.jsonl"
WORD_RE = re.compile(r"\b\w+\b")

# ====== Set the number of tasks for your dataset ======
task_N = 3   

# Load prompt verifications
with PROMPTS_FILE.open() as f:
    VERIF = {p["prompt_id"]: p["verification"] for p in map(json.loads, f)}

def count_word(txt: str) -> int:
    """Count number of words in a text."""
    return len(WORD_RE.findall(txt))

def ok(count: int, v: dict) -> bool:
    """Judge if count satisfies relation defined in v."""
    rel = v["relation"]
    if rel == "gte":
        return count >= v["target"]
    if rel == "lte":
        return count <= v["target"]
    if rel == "approx":
        return abs(count - v["target"]) <= 0.15 * v["target"]
    if rel == "range":
        return v["lower"] <= count <= v["upper"]
    raise ValueError(rel)

def evaluate(model_key: str):
    f_out = DATA / f"{model_key}_outputs.jsonl"
    if not f_out.exists():
        print(f"[skip] {f_out} missing")
        return

    # Prepare containers
    prompt_ids = sorted(VERIF)
    per_prompt = []
    per_prompt_skip = []
    # per_task_rows: each row is [t1_acc, t2_acc, t3_acc] for that prompt
    per_task_rows = []
    per_task_accumulators = [[0, 0] for _ in range(task_N)]  # [hit, total] for each task (overall)
    per_relation_hits = {}     # relation -> hit count
    per_relation_total = {}    # relation -> total task count

    with f_out.open() as f:
        # Each prompt's 8 samples (rows), group by prompt_id
        outputs_by_prompt = {pid: [] for pid in prompt_ids}
        for row in map(json.loads, f):
            outputs_by_prompt[row["prompt_id"]].append(row["text"])

    for pid in prompt_ids:
        samples = outputs_by_prompt[pid]
        skip_cnt = 0
        sample_acc_list = []
        # For per task: [correct_cnt, valid_cnt]
        task_hits = [[0, 0] for _ in range(task_N)]
        # Get task verification info
        verif_list = VERIF[pid]
        # Loop over 8 samples for this prompt
        for text in samples:
            # split by exact "**********", strip spaces
            splits = [seg.strip() for seg in text.split("**********")]
            if len(splits) != task_N:
                skip_cnt += 1
                continue  # skip abnormal sample
            task_score = 0  # for this sample: #correct/total
            for i in range(task_N):
                # this task's check info
                v = verif_list[i]
                rel = v["relation"]
                word_count = count_word(splits[i])
                is_ok = ok(word_count, v)
                if is_ok:
                    task_hits[i][0] += 1
                    task_score += 1
                    per_relation_hits[rel] = per_relation_hits.get(rel, 0) + 1
                per_relation_total[rel] = per_relation_total.get(rel, 0) + 1
                task_hits[i][1] += 1  # valid count
                per_task_accumulators[i][1] += 1  # overall valid
                if is_ok:
                    per_task_accumulators[i][0] += 1  # overall hit
            sample_acc_list.append(task_score / task_N)
        # Output per_prompt accuracy (one value for this prompt)
        if sample_acc_list:
            per_prompt.append(round(sum(sample_acc_list) / len(sample_acc_list), 2))
        else:
            per_prompt.append("error")  # all skipped
        # Output per_task (three values for this prompt, or 'error' if全skip)
        if any(h[1] > 0 for h in task_hits):
            row = []
            for i in range(task_N):
                if task_hits[i][1] == 0:
                    row.append("error")
                else:
                    row.append(round(task_hits[i][0] / task_hits[i][1], 2))
            per_task_rows.append(row)
        else:
            per_task_rows.append(["error"] * task_N)
        per_prompt_skip.append(round(skip_cnt / 8, 3))  # skip ratio, always/8

    # Overall accuracy (average of all per_prompt, skip error)
    valid_per_prompt = [x for x in per_prompt if isinstance(x, float)]
    if valid_per_prompt:
        overall_accuracy = round(sum(valid_per_prompt) / len(valid_per_prompt), 2)
    else:
        overall_accuracy = "error"

    # Overall per_task (average over 20 prompt, skip error)
    overall_per_task = []
    for i in range(task_N):
        valid = [row[i] for row in per_task_rows if isinstance(row[i], float)]
        if valid:
            overall_per_task.append(round(sum(valid) / len(valid), 2))
        else:
            overall_per_task.append("error")

    # Per relation
    per_relation = {
        rel: round(per_relation_hits.get(rel, 0) / per_relation_total[rel], 2)
        for rel in per_relation_total
    }

    # ===== Output: per your requirement (vertical, plain text style) =====
    print("\nper_prompt:")
    for x in per_prompt:
        print(x)
    print("\noverall_accuracy:")
    print(overall_accuracy)
    print("\nper_task:")
    for row in per_task_rows:
        print(" ".join(str(x) for x in row))
    print("\noverall_per_task:")
    print(" ".join(str(x) for x in overall_per_task))
    print("\nper_relation:")
    for rel in per_relation:
        print(f"{rel}: {per_relation[rel]}")
    print("\nper_prompt_skip:")
    for x in per_prompt_skip:
        print(x)

    # Optionally, also dump to json file for record
    res = {
        "per_prompt": per_prompt,
        "overall_accuracy": overall_accuracy,
        "per_task": per_task_rows,
        "overall_per_task": overall_per_task,
        "per_relation": per_relation,
        "per_prompt_skip": per_prompt_skip,
    }
    with (DATA / f"{model_key}_eval.json").open("w") as f:
        json.dump(res, f, indent=2)

    print(f"\n[done] {model_key}")

if __name__ == "__main__":
    models = sys.argv[1:] or ["llama4scout", "deepseek-v3", "gpt-4.1"]
    for m in models:
        evaluate(m)
