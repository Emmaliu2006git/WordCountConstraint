import json
import pathlib
from collections import defaultdict

EVAL_DIR = pathlib.Path("eval")
OUTDIR = pathlib.Path("relation_level_analysis")
OUTDIR.mkdir(exist_ok=True)

MODEL_EVAL_FILES = [
    "deepseek-v3_output_eval.json",
    "gpt-4.1_output_eval.json",
    "llama4scout_output_eval.json",
]

RELATIONS = ["range", "approx", "gte", "lte"]


def load_eval(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_relation_scores(model_data):
    per_output = model_data["per_output_records"]

    prompt_relation_scores = defaultdict(lambda: {
        rel: {"sum": 0.0, "count": 0, "pass": 0, "total": 0} for rel in RELATIONS
    })

    global_relation = {rel: {"sum": 0.0, "count": 0, "pass": 0, "total": 0} for rel in RELATIONS}

    for rec in per_output:
        pid = rec["prompt_id"]
        part_results = rec["part_results"]

        for _, pr in part_results.items():
            relation = pr["relation"]
            hard_score = pr["scores"]["hard"]

            if relation not in RELATIONS:
                continue

            prompt_relation_scores[pid][relation]["sum"] += hard_score
            prompt_relation_scores[pid][relation]["count"] += 1
            prompt_relation_scores[pid][relation]["total"] += 1
            if hard_score == 1.0:
                prompt_relation_scores[pid][relation]["pass"] += 1

            global_relation[relation]["sum"] += hard_score
            global_relation[relation]["count"] += 1
            global_relation[relation]["total"] += 1
            if hard_score == 1.0:
                global_relation[relation]["pass"] += 1

    return prompt_relation_scores, global_relation


def main():
    for fname in MODEL_EVAL_FILES:
        path = EVAL_DIR / fname
        if not path.exists():
            print(f"[skip] {fname} not found")
            continue

        data = load_eval(path)
        prompt_scores, global_scores = compute_relation_scores(data)

        out_data = {
            "model": fname,
            "per_prompt": {},
            "global": {}
        }

        for pid in sorted(prompt_scores.keys()):
            out_data["per_prompt"][pid] = {}
            for rel in RELATIONS:
                info = prompt_scores[pid][rel]
                avg = info["sum"] / info["count"] if info["count"] else 0.0
                out_data["per_prompt"][pid][rel] = {
                    "avg": avg,
                    "pass": info["pass"],
                    "total": info["total"]
                }

        for rel in RELATIONS:
            info = global_scores[rel]
            avg = info["sum"] / info["count"] if info["count"] else 0.0
            out_data["global"][rel] = {
                "avg": avg,
                "pass": info["pass"],
                "total": info["total"]
            }

        out_path = OUTDIR / fname.replace("_eval.json", "_relation_stats.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)

        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
