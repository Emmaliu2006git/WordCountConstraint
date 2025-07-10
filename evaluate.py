import json, re, sys, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
DATA = ROOT / "data/topic_test_lte"
# the input file
PROMPTS_FILE = DATA / "prompts.jsonl"
WORD_RE = re.compile(r"\b\w+\b")

# create a dictionary with every prompt id matched to the relation of word
with PROMPTS_FILE.open() as f:
    VERIF = {p["prompt_id"]: p["verification"] for p in map(json.loads, f)}

def count_word(txt: str) -> int:
    return len(WORD_RE.findall(txt))  # count number of words in a text

def ok(count: int, v: dict) -> bool:
    rel = v["relation"]
    if rel == "gte":     return count >= v["target"]
    if rel == "lte":     return count <= v["target"]
    if rel == "approx":  return abs(count - v["target"]) <= 0.1 * v["target"]  # within 10% difference
    if rel == "range":   return v["lower"] <= count <= v["upper"]
    raise ValueError(rel)

def evaluate(model_key: str):
    f_out = DATA / f"{model_key}_outputs.jsonl"
    # if output file not found
    if not f_out.exists():
        print(f"[skip] {f_out} missing")
        return
    #dict, initialize hit with 0 for all prompts
    hits = {prompt_id: 0 for prompt_id in VERIF}

    # new: track how each relation type performs
    relation_hits = {} #dict, how many hits for each relation
    relation_total = {} #dict, how many output texts for each relation

    with f_out.open() as f:
        for row in map(json.loads, f):
            prompt_id = row["prompt_id"]
            c = count_word(row["text"])
            v = VERIF[prompt_id]
            rel = v["relation"]
            #update the total of rels, plus 1
            relation_total[rel] = relation_total.get(rel, 0) + 1
            if ok(c, v):
                hits[prompt_id] += 1
                relation_hits[rel] = relation_hits.get(rel, 0) + 1

    #statistics
    #put the accuracy for each prompt in a list
    per_prompt = [hits[p] / 8 for p in sorted(hits)]
    #accuracy accross 20 prompts
    overall = sum(per_prompt) / 20

    # compute per-relation satisfaction rate
    per_relation = {
        r: round(relation_hits.get(r, 0) / relation_total[r], 2) #keep two decimal places
        for r in relation_total
    }

    #store and print result
    res = {
        "overall_rate": round(overall, 2), #keep two decimal places
        "per_prompt": [round(x, 2) for x in per_prompt], #keep two decimal places
        "per_relation": per_relation
    }

    with (DATA / f"{model_key}_eval.json").open("w") as f:
        json.dump(res, f, indent=2)

    print(f"[done] {model_key}: {overall:.2%}")

if __name__ == "__main__":
    models = sys.argv[1:] or ["llama4scout", "deepseek-v3", "gpt-4.1"]
    for m in models:
        evaluate(m)
