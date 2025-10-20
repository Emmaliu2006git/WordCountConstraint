# eval_segmented.py (outputs and inputs under ./data)
#!/usr/bin/env python3
import json, pathlib, re, statistics, os
from collections import defaultdict
from helpers.counting import word_count, paragraph_count, line_count
from helpers.metrics import parse_target, hard_metric

# --- Fixed inputs/outputs (everything lives under ./data) ---
DATA_DIR = pathlib.Path("data")
MODEL_FILES = [
    DATA_DIR / "deepseek-v3_output.txt",
    DATA_DIR / "gpt-4.1_output.txt",
    DATA_DIR / "llama4scout_output.txt",
]
VERIFICATION_PATH = DATA_DIR / "segmented.jsonl"  # verification tags live here
OUTDIR = DATA_DIR  # write JSON reports here (no separate folder)

PROMPT_SPLIT_RE = re.compile(r"^##\s*prompt_id:\s*(\d+)\s*,\s*type:\s*segmented\s*$", re.MULTILINE)
PART_HEADER_RE  = re.compile(r"(?mi)^#part\s*(\d+)\s*$")

LEVEL_COUNTERS = {
    "word": word_count,
    "paragraph": paragraph_count,
    "line": line_count,
}

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

#Split a model’s text into prompts
def parse_model_outputs(path: pathlib.Path):
    txt = path.read_text(encoding="utf-8", errors="ignore")
    matches = list(PROMPT_SPLIT_RE.finditer(txt))
    per_prompt = defaultdict(list)
    for i, m in enumerate(matches):
        pid = int(m.group(1))
        start = m.end()
        end   = matches[i+1].start() if i+1 < len(matches) else len(txt)
        content = txt[start:end].strip()
        per_prompt[pid].append((i, content))
    for pid in per_prompt:
        per_prompt[pid].sort(key=lambda t: t[0])
    return per_prompt

#Extract part content from a single output
def slice_parts(content: str):
    headers = list(PART_HEADER_RE.finditer(content))
    if not headers:
        return {}
    parts = {}
    for j, hdr in enumerate(headers):
        n = int(hdr.group(1))
        s = hdr.end()
        e = headers[j+1].start() if j+1 < len(headers) else len(content)
        parts[n] = content[s:e].strip()
    return parts

#Select and call the correct counting function
def measure(level: str, text: str) -> int:
    counter = LEVEL_COUNTERS.get(level.lower())
    return counter(text) if counter else 0

#Evaluate one output’s validity and correctness
def verify_output(prompt_id: int, content: str, vtag: dict):
    exp_n = int(vtag["part_number"])
    parts = slice_parts(content)
    if not parts:
        return {
            "format_valid": False,
            "parts_found": 0,
            "missing_parts": list(range(1, exp_n+1)),
            "extraneous_parts": [],
            "part_results": {},
            "output_pass": False
        }
    missing = [k for k in range(1, exp_n+1) if k not in parts]
    extra = sorted([k for k in parts.keys() if k < 1 or k > exp_n])
    all_ok = True
    part_results = {}
    for k in range(1, exp_n+1):
        spec = vtag[str(k)]
        level = spec["level"]
        relation = spec["relation"]
        target_raw = spec["target"]
        actual = measure(level, parts.get(k, ""))
        target = parse_target(relation, target_raw)
        ok = hard_metric(relation, actual, target)
        part_results[k] = {"level": level, "relation": relation, "target": target_raw, "measured": actual, "pass": ok}
        all_ok = all_ok and ok
    output_pass = (len(missing) == 0) and all_ok
    return {
        "format_valid": True,
        "parts_found": len(parts),
        "missing_parts": missing,
        "extraneous_parts": extra,
        "part_results": part_results,
        "output_pass": output_pass
    }

#Process all prompts for one model
def evaluate_model_file(model_file: pathlib.Path, vtags: dict):
    model_name = model_file.stem
    per_prompt = parse_model_outputs(model_file)
    prompt_acc, invalid, mismatches, per_output = {}, [], [], []
    for pid, outs in per_prompt.items():
        if pid not in vtags:
            continue
        expected_parts = int(vtags[pid]["part_number"])
        bins = []
        for j, (_ord, content) in enumerate(outs, start=1):
            res = verify_output(pid, content, vtags[pid])
            per_output.append({
                "model": model_name,
                "prompt_id": pid,
                "output_idx": j,
                "pass": int(bool(res["output_pass"])),
                "format_valid": res["format_valid"],
                "parts_found": res["parts_found"],
                "expected_parts": expected_parts,
                "missing_parts": res["missing_parts"],
                "extraneous_parts": res["extraneous_parts"],
                "part_results": res["part_results"]
            })
            bins.append(int(bool(res["output_pass"])))
            if not res["format_valid"]:
                invalid.append({"model": model_name, "prompt_id": pid, "output_idx": j, "reason": "missing_or_bad_part_headers", "parts_found": res["parts_found"]})
            elif res["parts_found"] != expected_parts:
                mismatches.append({"model": model_name, "prompt_id": pid, "output_idx": j, "expected_parts": expected_parts, "actual_parts": res["parts_found"]})
        prompt_acc[pid] = statistics.mean(bins) if bins else 0.0
    model_accuracy = statistics.mean(prompt_acc.values()) if prompt_acc else 0.0
    return {
        "model": model_name,
        "prompt_accuracy": prompt_acc,
        "model_accuracy": model_accuracy,
        "invalid_outputs": invalid,
        "part_count_mismatches": mismatches,
        "per_output_records": per_output
    }

#Run the entire pipeline for all three models
def main():
    vtags = load_verifications(VERIFICATION_PATH)
    results = []
    for mf in MODEL_FILES:
        if not mf.exists():
            print(f"[skip] {mf} not found (cwd={os.getcwd()})")
            continue
        res = evaluate_model_file(mf, vtags)
        results.append(res)
        out_path = OUTDIR / f"{res['model']}_segmented_eval.json"
        payload = {
            "model": res["model"],
            "prompt_accuracy": res["prompt_accuracy"],
            "model_accuracy": res["model_accuracy"],
            "invalid_outputs": res["invalid_outputs"],
            "part_count_mismatches": res["part_count_mismatches"],
            "per_output_records": res["per_output_records"]
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Wrote {out_path}")

    if not results:
        print("No model files processed.")
        return

    per_model_acc = {r["model"]: r["model_accuracy"] for r in results}
    all_bins = [rec["pass"] for r in results for rec in r["per_output_records"]]
    overall = statistics.mean(all_bins) if all_bins else 0.0

    print("\nPer-model accuracy:")
    for m, acc in per_model_acc.items():
        print(f"- {m}: {acc:.3f}")
    print(f"\nOverall accuracy (all models): {overall:.3f}")

if __name__ == "__main__":
    main()
