import json, pathlib, re, statistics, os
from collections import defaultdict
from helpers.counting import word_count, paragraph_count, line_count
from helpers.metrics import parse_target, hard_metric
# from helpers.metrics import soft_metric_basic, soft_metric_advanced  

#constant declaration
DATA_DIR = pathlib.Path("data")
MODEL_FILES = [
    DATA_DIR / "deepseek-v3_output.txt",
    DATA_DIR / "gpt-4.1_output.txt",
    DATA_DIR / "llama4scout_output.txt",
]
VERIFICATION_PATH = DATA_DIR / "segmented.jsonl"
OUTDIR = DATA_DIR
#to split into prompts
PROMPT_SPLIT_RE = re.compile(r"^##\s*prompt_id:\s*(\d+)\s*,\s*type:\s*segmented\s*$", re.MULTILINE)
#to split into parts
PART_HEADER_RE = re.compile(r"(?mi)^#part\s*(\d+)\s*$")

LEVEL_COUNTERS = {
    "word": word_count,
    "paragraph": paragraph_count,
    "line": line_count,
}

# store all verification ruls into verifs
def load_verifications(segmented_jsonl_path: pathlib.Path):
    verifs = {}
    with segmented_jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            if obj.get("prompt_type") != "segmented":
                continue
            verifs[int(obj["prompt_id"])] = obj["verification"]
    return verifs

#parse prompts
def parse_model_outputs(path: pathlib.Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(PROMPT_SPLIT_RE.finditer(txt))
    per_prompt = defaultdict(list)
    for i, m in enumerate(matches):
        pid = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(txt)
        content = txt[start:end].strip()
        per_prompt[pid].append((i, content))
    for pid in per_prompt:
        per_prompt[pid].sort(key=lambda t: t[0])
    return per_prompt

#parse parts
def slice_parts(content: str):
    headers = list(PART_HEADER_RE.finditer(content))
    if not headers:
        return {}
    parts = {}
    for j, hdr in enumerate(headers):
        n = int(hdr.group(1))
        s = hdr.end()
        e = headers[j + 1].start() if j + 1 < len(headers) else len(content)
        parts[n] = content[s:e].strip()
    return parts

# get actual count
def measure(level: str, text: str) -> int:
    counter = LEVEL_COUNTERS.get(level)
    return counter(text) if counter else 0

# multiple metric structure, evaluate one part
def part_scores(relation: str, actual: int, target):
    """Return a dict of {metric_name: score ∈ [0,1]} for all metrics."""
    hard = float(hard_metric(relation, actual, target))
    # soft_basic = soft_metric_basic(relation, actual, target)
    # soft_advanced = soft_metric_advanced(relation, actual, target)
    return {
        "hard": hard,
        # "soft_basic": soft_basic,
        # "soft_advanced": soft_advanced
    }

# evaluate a single output/sample
def verify_output(prompt_id: int, content: str, vtag: dict):
    exp_n = int(vtag["part_number"])
    #slice_parts(content: str), return parts
    parts = slice_parts(content)
    if not parts:
        return {
            "format_valid": False,
            "parts_found": 0,
            "missing_parts": list(range(1, exp_n + 1)),
            "part_results": {},
            "output_pass": False,
            "sample_scores": {"hard": 0.0}
        }

    missing = [k for k in range(1, exp_n + 1) if k not in parts]
    part_results = {}
    metric_bins = {"hard": []}  # currently only hard metric
    # metric_bins = {"hard": [], "soft_basic": [], "soft_advanced": []}  # reserved for later use

    for k in range(1, exp_n + 1):
        spec = vtag[str(k)]
        level = spec["level"]
        relation = spec["relation"]
        target_raw = spec["target"]
        actual = measure(level, parts.get(k, ""))
        target = parse_target(relation, target_raw)
        scores_dict = part_scores(relation, actual, target)
        for key, val in scores_dict.items():
            metric_bins[key].append(val)
        part_results[k] = {
            "level": level,
            "relation": relation,
            "target": target_raw,
            "measured": actual,
            "scores": scores_dict
        }

    # average each metric across all parts
    sample_scores = {
        key: statistics.mean(vals) if vals else 0.0
        for key, vals in metric_bins.items()
    }
    output_pass = sample_scores["hard"] == 1.0
    return {
        "format_valid": True,
        "parts_found": len(parts),
        "missing_parts": missing,
        "part_results": part_results,
        "output_pass": output_pass,
        "sample_scores": sample_scores
    }

# evaluate one model file
def evaluate_model_file(model_file: pathlib.Path, vtags: dict):
    model_name = model_file.stem
    #def parse_model_outputs(path: pathlib.Path). dict that contains all all prompt_id-sample pair
    per_prompt = parse_model_outputs(model_file)
    prompt_acc, invalid, mismatches, per_output = {}, [], [], []
    #for every prompt
    for pid, outs in per_prompt.items():
        if pid not in vtags:
            continue
        expected_parts = int(vtags[pid]["part_number"])
        metric_sums = {"hard": []}  # extendable to multiple metrics

        #every output
        for j, (_ord, content) in enumerate(outs, start=1):
            # def verify_output(prompt_id: int, content: str, vtag: dict), result for one sample
            res = verify_output(pid, content, vtags[pid])
            #details for each output
            per_output.append({
                "prompt_id": pid,
                "output_idx": j,
                "pass": int(bool(res["output_pass"])),  # pass/fail for single output
                "sample_scores": res["sample_scores"], #now is binary, will be 0-1 using different metrics
                "format_valid": res["format_valid"],    #has part n tag
                "parts_found": res["parts_found"],  
                "expected_parts": expected_parts,
                "missing_parts": res["missing_parts"],
                "part_results": res["part_results"] #level, relation, target, actual, scores(hard, soft...)
            })
            # store scores for per prompt average
            for metric_name, score_val in res["sample_scores"].items():
                # If this metric does not exist in metric_sums yet, create an empty list for it
                if metric_name not in metric_sums:
                    metric_sums[metric_name] = []
                metric_sums[metric_name].append(score_val)
            #illegal output format, update invalid
            if not res["format_valid"]:
                invalid.append({
                    "prompt_id": pid,
                    "output_idx": j,
                    "reason": "missing part n tag",
                })
            #part number mismatch
            elif res["parts_found"] != expected_parts:
                mismatches.append({
                    "prompt_id": pid,
                    "output_idx": j,
                    "expected_parts": expected_parts,
                    "actual_parts": res["parts_found"]
                })
        # prompt level accuracy for each metric
        prompt_acc[pid] = {}
        for metric, vals in metric_sums.items():
            if vals:
                prompt_acc[pid][metric] = statistics.mean(vals)
            else:
                prompt_acc[pid][metric] = 0.0


    # model-level accuracy for each metric
    model_accuracy = {
        metric: statistics.mean([v[metric] for v in prompt_acc.values()])
        if prompt_acc else 0.0
        for metric in metric_sums
    }

    return {
        "model": model_name,
        "prompt_accuracy": prompt_acc,
        "model_accuracy": model_accuracy,
        "invalid_outputs": invalid,
        "part_count_mismatches": mismatches,
        "per_output_records": per_output
    }

def main():
    #get verification tags
    vtags = load_verifications(VERIFICATION_PATH)
    # entire results for 3 models
    results = []
    for mf in MODEL_FILES:
        if not mf.exists():
            print(f"[skip] {mf} not found (cwd={os.getcwd()})")
            continue

        model_name = mf.stem
        print(model_name)

        #def evaluate_model_file(model_file: pathlib.Path, vtags: dict), result of one model
        res = evaluate_model_file(mf, vtags)
        results.append(res)
        out_path = OUTDIR / f"{res['model']}_eval.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    # no models being processed
    if not results:
        print("No model files processed.")
        return

    # print overall accuracy of each model
    for r in results:
        for metric_name, acc in r["model_accuracy"].items():
            print(f"{r['model']} ({metric_name}) overall accuracy: {acc:.3f}")

if __name__ == "__main__":
    main()
