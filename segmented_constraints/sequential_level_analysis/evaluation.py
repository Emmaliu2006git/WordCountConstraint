import json, pathlib, statistics, os
from collections import defaultdict
from helpers.counting import word_count, paragraph_count, line_count
from helpers.metrics import parse_target
from helpers.metrics import hard_metric
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
OUTDIR = ROOT/"eval"

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

def load_model_outputs(jsonl_path: pathlib.Path):
    per_prompt = defaultdict(list)
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pid = int(obj["prompt_id"])
            per_prompt[pid].append(obj)
    return per_prompt

def slice_parts(content: str):
    part_header_re = re.compile(r"(?mi)^#part\s*(\d+)\s*$")
    headers = list(part_header_re.finditer(content))
    if not headers:
        return {}
    parts = {}
    for j, hdr in enumerate(headers):
        n = int(hdr.group(1))
        s = hdr.end()
        e = headers[j+1].start() if j+1 < len(headers) else len(content)
        parts[n] = content[s:e].strip()
    return parts

def measure(level: str, text: str) -> int:
    counter = LEVEL_COUNTERS.get(level)
    return counter(text) if counter else 0

def score_last_part(content: str, vtag: dict):
    last = int(vtag["part_number"])
    parts = slice_parts(content)
    spec = vtag[str(last)]
    level = spec["level"]
    relation = spec["relation"]
    target_raw = spec["target"]
    actual = measure(level, parts.get(last, ""))
    target = parse_target(relation, target_raw)
    score = float(hard_metric(relation, actual, target))
    return score, actual, target_raw, level, relation

def evaluate_model_file(model_jsonl: pathlib.Path):
    model_name = model_jsonl.stem
    per_prompt = load_model_outputs(model_jsonl)

    prompt_avg_scores = {}
    prompt_details = {}
    per_output = []

    for pid, records in per_prompt.items():
        vtag = records[0]["verification"]

        sample_scores = []
        sample_records = []

        for j, obj in enumerate(records, start=1):
            content = obj["output"]
            score, actual, target_raw, level, relation = score_last_part(content, vtag)

            sample_scores.append(score)
            sample_records.append({
                "output_idx": j,
                "score": score,
                "measured": actual,
                "target": target_raw,
                "level": level,
                "relation": relation
            })

            per_output.append({
                "prompt_id": pid,
                "output_idx": j,
                "score": score,
                "measured": actual,
                "target": target_raw
            })

        avg_score = statistics.mean(sample_scores) if sample_scores else 0.0
        prompt_avg_scores[pid] = avg_score

        prompt_details[pid] = {
            "sample_scores": sample_records
        }

    return {
        "model": model_name,
        "prompt_avg_scores": prompt_avg_scores,   # NEW: all avg scores at the front
        "prompt_details": prompt_details,         # sample-level details
        "per_output_records": per_output
    }

def main():
    results = []
    for mf in MODEL_FILES:
        if not mf.exists():
            print(f"[skip] {mf} not found (cwd={os.getcwd()})")
            continue
        res = evaluate_model_file(mf)
        results.append(res)
        out_path = OUTDIR / f"{res['model']}_eval.json"
        out_path.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")

    if not results:
        print("No model files processed.")
        return

if __name__ == "__main__":
    main()
