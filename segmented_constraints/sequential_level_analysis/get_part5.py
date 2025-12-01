import json
import pathlib
from collections import defaultdict

# input eval folder
EVAL_DIR = pathlib.Path("..") / "eval"

# output file
OUT_PATH = pathlib.Path()/ "eval" / "last_part_summary.json"

TARGET_PROMPTS = {1, 4, 5, 14, 16}
LAST_PART = 5


def extract_last_part_scores(eval_json):
    scores = defaultdict(list)

    # each record is a model-generated sample
    for record in eval_json["per_output_records"]:
        pid = int(record["prompt_id"])
        if pid not in TARGET_PROMPTS:
            continue

        part_results = record["part_results"]
        if str(LAST_PART) in part_results:
            hard_score = part_results[str(LAST_PART)]["scores"]["hard"]
            scores[pid].append(hard_score)

    return scores


def main():
    eval_files = list(EVAL_DIR.glob("*_eval.json"))

    if not eval_files:
        print("No evaluation result files found.")
        return

    final_summary = {}

    for f in eval_files:
        with open(f, "r", encoding="utf-8") as infile:
            data = json.load(infile)

        model_name = data["model"]
        collected = extract_last_part_scores(data)

        model_summary = {}
        for pid in sorted(collected):
            vals = collected[pid]
            avg = sum(vals) / len(vals) if vals else 0.0
            model_summary[pid] = avg

        final_summary[model_name] = model_summary

    # write output
    OUT_PATH.write_text(
        json.dumps(final_summary, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    print(f"Summary written to: {OUT_PATH}")


if __name__ == "__main__":
    main()
