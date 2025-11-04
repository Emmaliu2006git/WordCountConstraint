import json, pathlib, statistics, os
from collections import defaultdict
# import counters for actual count
from helpers.counting import word_count, paragraph_count, line_count
# import different metric functions
from helpers.metrics import parse_target
from helpers.metrics import hard_metric # from helpers.metrics import soft_metric_basic, soft_metric_advanced  
import re

# constants declaration
DATA_DIR = pathlib.Path("data")
OUTDIR = pathlib.Path("eval")

MODEL_FILES = [
    DATA_DIR / "deepseek-v3_output.jsonl",
    DATA_DIR / "gpt-4.1_output.jsonl",
    DATA_DIR / "llama4scout_output.jsonl",
]


LEVEL_COUNTERS = {
    "word": word_count,
    "paragraph": paragraph_count,
    "line": line_count,
}

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
# pass in the output file
def load_model_outputs(jsonl_path: pathlib.Path):
    # create a dictionary, store all the output under this single pid
    per_prompt = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = int(obj["prompt_id"])
            per_prompt[pid].append(obj)
    return per_prompt

"""Split content into numbered parts using '#part n' headers."""
def slice_parts(content: str):
    # get the number after part tag
    part_header_re = re.compile(r"(?mi)^#part\s*(\d+)\s*$")
    headers = list(part_header_re.finditer(content))
    if not headers:
        return {}
    parts = {}
    # part and content
    for j, hdr in enumerate(headers):
        n = int(hdr.group(1))   # get the number
        s = hdr.end()   # part 
        e = headers[j + 1].start() if j + 1 < len(headers) else len(content)
        parts[n] = content[s:e].strip()
    return parts


def measure(level: str, text: str) -> int:
    counter = LEVEL_COUNTERS.get(level)
    return counter(text) if counter else 0

# input relation and actual and target
def part_scores(relation: str, actual: int, target):
    """Return per-part metric dictionary."""
    hard = float(hard_metric(relation, actual, target))
    # soft_basic = soft_metric_basic(relation, actual, target)
    # soft_advanced = soft_metric_advanced(relation, actual, target)
    return {
        "hard": hard,
        # "soft_basic": soft_basic,
        # "soft_advanced": soft_advanced
    }


# for one single output
def verify_output(content: str, vtag: dict):
    exp_n = int(vtag["part_number"])
    parts = slice_parts(content)
    part_results = {}
    metric_bins = {"hard": []}
    # iterate through the parts
    for k in range(1, exp_n + 1):
        spec = vtag[str(k)]
        level = spec["level"]
        relation = spec["relation"]
        target_raw = spec["target"]
        actual = measure(level, parts[k])
        target = parse_target(relation, target_raw)
        # score of one part
        scores = part_scores(relation, actual, target)
        for key, val in scores.items():
            metric_bins[key].append(val)
        part_results[k] = {
            "level": level,
            "relation": relation,
            "target": target_raw,
            "measured": actual,
            "scores": scores
        }

    sample_scores = {k: statistics.mean(v) if v else 0.0 for k, v in metric_bins.items()}   # output score
    output_pass = sample_scores["hard"] == 1.0  # output pass

    return {
        "part_results": part_results,
        "output_pass": output_pass,
        "sample_scores": sample_scores
    }


def evaluate_model_file(model_jsonl: pathlib.Path):
    model_name = model_jsonl.stem
    per_prompt = load_model_outputs(model_jsonl)
    prompt_pass_rate = {}
    prompt_accuracy = {}
    per_output = []
    # for a single prompt
    for pid, records in per_prompt.items():
        metric_sums = {"hard": []}
        vtag = records[0]["verification"]

        total_outputs = len(records)
        pass_outputs = 0
        # for a single output
        for j, obj in enumerate(records, start=1):
            content = obj["output"]
            res = verify_output(content, vtag)

            if res["output_pass"]:
                pass_outputs += 1

            per_output.append({
                "prompt_id": pid,
                "output_idx": j,
                "pass": int(res["output_pass"]),
                "sample_scores": res["sample_scores"],
                "part_results": res["part_results"]
            })

            for metric, val in res["sample_scores"].items():
                metric_sums.setdefault(metric, []).append(val)

        prompt_pass_rate[pid] = pass_outputs / total_outputs if total_outputs else 0.0
        prompt_accuracy[pid] = {
            m: statistics.mean(v) if v else 0.0 for m, v in metric_sums.items()
        }

    model_accuracy = {
        metric: statistics.mean([v[metric] for v in prompt_accuracy.values()])
        if prompt_accuracy else 0.0
        for metric in metric_sums
    }
    model_accuracy["prompt_pass_rate"] = statistics.mean(
        [v for v in prompt_pass_rate.values()]
    ) if prompt_pass_rate else 0.0

    return {
        "model": model_name,
        "prompt_pass_rate": prompt_pass_rate,
        "prompt_accuracy": prompt_accuracy,
        "model_accuracy": model_accuracy,
        "per_output_records": per_output
    }




# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    results = []
    for mf in MODEL_FILES:
        if not mf.exists():
            print(f"[skip] {mf} not found (cwd={os.getcwd()})")
            continue
        print(mf.stem)
        res = evaluate_model_file(mf)
        results.append(res)
        out_path = OUTDIR / f"{res['model']}_eval.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    if not results:
        print("No model files processed.")
        return

    for r in results:
        for metric_name, acc in r["model_accuracy"].items():
            print(f"{r['model']} ({metric_name}) overall accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
