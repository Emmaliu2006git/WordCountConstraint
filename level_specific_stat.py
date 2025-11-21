import json
import pathlib
from collections import defaultdict

EVAL_DIR = pathlib.Path("eval")
OUTDIR = pathlib.Path("constraint_level_analysis")
OUTDIR.mkdir(exist_ok=True)

MODEL_EVAL_FILES = [
    "deepseek-v3_output_eval.json",
    "gpt-4.1_output_eval.json",
    "llama4scout_output_eval.json",
]

LEVELS = ["word", "paragraph", "line"]


def load_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_level_scores(model_data):
    per_output = model_data["per_output_records"]

    prompt_level_scores = defaultdict(lambda: {
        lvl: {"sum": 0.0, "count": 0, "pass": 0, "total": 0} for lvl in LEVELS
    })

    global_level = {lvl: {"sum": 0.0, "count": 0, "pass": 0, "total": 0} for lvl in LEVELS}

    for rec in per_output:
        pid = rec["prompt_id"]
        part_results = rec["part_results"]

        for _, pr in part_results.items():
            lvl = pr["level"]
            hard_score = pr["scores"]["hard"]

            prompt_level_scores[pid][lvl]["sum"] += hard_score
            prompt_level_scores[pid][lvl]["count"] += 1
            prompt_level_scores[pid][lvl]["total"] += 1
            if hard_score == 1.0:
                prompt_level_scores[pid][lvl]["pass"] += 1

            global_level[lvl]["sum"] += hard_score
            global_level[lvl]["count"] += 1
            global_level[lvl]["total"] += 1
            if hard_score == 1.0:
                global_level[lvl]["pass"] += 1

    return prompt_level_scores, global_level


def main():
    for fname in MODEL_EVAL_FILES:
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"[skip] {fname} not found")
            continue

        data = load_eval(path)
        prompt_levels, global_levels = compute_level_scores(data)

        out_data = {
            "model": fname,
            "per_prompt": {},
            "global": {}
        }

        for pid in sorted(prompt_levels.keys()):
            out_data["per_prompt"][pid] = {}
            for lvl in LEVELS:
                info = prompt_levels[pid][lvl]
                avg = info["sum"] / info["count"] if info["count"] else 0.0
                out_data["per_prompt"][pid][lvl] = {
                    "avg": avg,
                    "pass": info["pass"],
                    "total": info["total"]
                }

        for lvl in LEVELS:
            info = global_levels[lvl]
            avg = info["sum"] / info["count"] if info["count"] else 0.0
            out_data["global"][lvl] = {
                "avg": avg,
                "pass": info["pass"],
                "total": info["total"]
            }

        out_path = OUTDIR / (fname.replace("_eval.json", "_level_stats.json"))
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
