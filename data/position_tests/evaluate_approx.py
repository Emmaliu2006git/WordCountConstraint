import json, re, sys, pathlib

# ---------------------------------------------
# Setup paths and constants
# ---------------------------------------------
ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data/format_test_1"
PROMPTS_FILE = DATA / "prompts.jsonl"
WORD_RE = re.compile(r"\b\w+\b")

# Load verification conditions
with PROMPTS_FILE.open() as f:
    VERIF = {p["prompt_id"]: p["verification"] for p in map(json.loads, f)}

# ---------------------------------------------
# Word count utility
# ---------------------------------------------
def count_word(txt: str) -> int:
    return len(WORD_RE.findall(txt))

# ---------------------------------------------
# Check if a word count satisfies a constraint
# ---------------------------------------------
def ok(count: int, v: dict) -> bool:
    rel = v["relation"]
    if rel == "gte":     return count >= v["target"]
    if rel == "lte":     return count <= v["target"]
    if rel == "approx":  return abs(count - v["target"]) <= 0.1 * v["target"]
    if rel == "range":   return v["lower"] <= count <= v["upper"]
    raise ValueError(rel)

# ---------------------------------------------
# Main evaluation per model
# ---------------------------------------------
def evaluate(model_key: str):
    f_out = DATA / f"{model_key}_outputs.jsonl"
    if not f_out.exists():
        print(f"[skip] {f_out} missing")
        return

    # Init data containers
    hits = {prompt_id: 0 for prompt_id in VERIF}
    relation_hits = {}
    relation_total = {}
    details = {prompt_id: [] for prompt_id in VERIF}
    undergen_total = 0  # count of under-target generations
    total_samples = 0   # total number of evaluated samples

    # ---------------------------------------------
    # Evaluate all generations from the model
    # ---------------------------------------------
    with f_out.open() as f:
        for row in map(json.loads, f):
            prompt_id = row["prompt_id"]
            c = count_word(row["text"])
            v = VERIF[prompt_id]
            rel = v["relation"]
            target = v.get("target", 0)

            total_samples += 1
            relation_total[rel] = relation_total.get(rel, 0) + 1

            passed = ok(c, v)
            if passed:
                hits[prompt_id] += 1
                relation_hits[rel] = relation_hits.get(rel, 0) + 1

            diff_pct = round((c - target) / target * 100, 2) if target else 0.0
            under = int(c < target)
            undergen_total += under

            details[prompt_id].append({
                "actual": c,
                "target": target,
                "diff_pct": diff_pct,
                "pass": passed,
                "under": bool(under)
            })

    # ---------------------------------------------
    # Compute final statistics
    # ---------------------------------------------
    # Accuracy per prompt
    per_prompt = [hits[p] / len(details[p]) for p in sorted(hits)]
    overall = sum(per_prompt) / len(per_prompt)

    # Accuracy per relation
    per_relation = {
        r: round(relation_hits.get(r, 0) / relation_total[r], 2)
        for r in relation_total
    }

    # Avg absolute diff % and undergen rate per prompt
    per_prompt_avg_abs_diff_pct = {}
    per_prompt_undergen_pct = {}
    for p, records in details.items():
        abs_diffs = [abs(x["diff_pct"]) for x in records]
        unders = [x["under"] for x in records]
        per_prompt_avg_abs_diff_pct[p] = round(sum(abs_diffs) / len(abs_diffs), 2)
        per_prompt_undergen_pct[p] = round(sum(unders) / len(unders), 2)

    # Overall under-generation percent
    overall_undergen_pct = round(undergen_total / total_samples, 2)

    # ---------------------------------------------
    # Write evaluation results
    # ---------------------------------------------
    res = {
        "overall_rate": round(overall, 2),
        "per_prompt_rate": {p: round(hits[p] / len(details[p]), 2) for p in sorted(hits)},
        "per_relation": per_relation,
        "per_prompt_avg_abs_diff_pct": per_prompt_avg_abs_diff_pct,
        "per_prompt_undergen_pct": per_prompt_undergen_pct,
        "overall_undergen_pct": overall_undergen_pct,
        "details": details
    }

    output_file = DATA / f"{model_key}_eval.json"
    with output_file.open("w") as f:
        json.dump(res, f, indent=2)

    print(f"[done] {model_key}: {overall:.2%} → saved to {output_file.name}")

# ---------------------------------------------
# Run for all models
# ---------------------------------------------
if __name__ == "__main__":
    models = sys.argv[1:] or ["llama4scout", "deepseek-v3", "gpt-4.1"]
    for m in models:
        evaluate(m)
