import json
import time
from pathlib import Path
import sys
import concurrent.futures as cf
from openai import OpenAI
import os
import argparse

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

# OpenAI official client
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# Together.ai client
together_client = OpenAI(
    base_url="https://api.together.xyz/v1",
    api_key=os.getenv("TOGETHER_API_KEY")
)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT
PROMPTS_FILE = DATA_DIR / "prompts_complete.jsonl"

# Sampling settings
number_of_samples = 8  # 8 outputs per prompt per model
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 2800  # Approximately 2000 words

# Model definitions and corresponding clients
models = {
    "llama4scout": {"name": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "client": together_client},
#    "deepseek-v3": {"name": "deepseek-ai/DeepSeek-V3",   "client": together_client},
#    "gpt-4.1":     {"name": "gpt-4.1-2025-04-14",        "client": openai_client}
}

SYSTEM_PROMPT = "You are an AI model that strictly follows user instructions. Do not answer anything beyond the user's request."

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_prompts(path: Path):
    """Load prompts from a JSONL file and add prompt_id automatically."""
    prompts_list = []
    with path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):  # starting from 1
            prompt_dict = json.loads(line)
            prompt_dict["prompt_id"] = idx
            prompts_list.append(prompt_dict)
    return prompts_list

def call_llm(model_info: dict, prompt_text: str) -> str:
    """Generate one response from a given model and prompt."""
    client = model_info["client"]
    model_name = model_info["name"]
    # Sleep during Together.ai rate limits
    if "together.xyz" in str(getattr(client, "base_url", "")):
        time.sleep(1.1)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": prompt_text}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    return resp.choices[0].message.content

def generate_samples(model_key: str, prompt_row: dict):
    """Generate multiple samples for a single prompt using one model."""
    result = []
    for sid in range(1, number_of_samples + 1):
        while True:
            try:
                text = call_llm(models[model_key], prompt_row["prompt"])
                break
            except Exception as e:
                print(f"[{model_key}] Prompt {prompt_row.get('prompt_id')} error: {e}", file=sys.stderr)
                time.sleep(10)
        print(f"[{model_key}] Generated output for Prompt {prompt_row.get('prompt_id')} Sample {sid}")
        rec = {
            "category": prompt_row.get("category"),
            "constraint": prompt_row.get("constraint"),
            "prompt_id": prompt_row.get("prompt_id"),
            "sample_id": sid,
            "text": text
        }
        result.append(rec)
    return result

def write_jsonl(path: Path, records):
    """Append a list of records to a JSONL file (deduplicate by prompt_id+sample_id)."""
    existing = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing.add((rec.get("prompt_id"), rec.get("sample_id")))
                except:
                    continue

    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for rec in records:
            key = (rec["prompt_id"], rec["sample_id"])
            if key in existing:
                continue
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

def get_resume_point(path: Path):
    """Return the prompt_id to start from.
       If the last prompt is incomplete, start from that prompt again.
    """
    if not path.exists():
        return 1

    last_prompt_id = 0
    samples_seen = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
                pid = rec.get("prompt_id", 0)
                sid = rec.get("sample_id", 0)
                last_prompt_id = max(last_prompt_id, pid)
                samples_seen.setdefault(pid, set()).add(sid)
            except:
                continue

    if last_prompt_id == 0:
        return 1

    if len(samples_seen.get(last_prompt_id, [])) < number_of_samples:
        return last_prompt_id
    else:
        return last_prompt_id + 1

# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true", help="Continue from last unfinished prompt")
    args = parser.parse_args()

    prompts = load_prompts(PROMPTS_FILE)

    for mkey in models:
        out_file = DATA_DIR / f"{mkey}_outputs.jsonl"

        if args.resume:
            start_pid = get_resume_point(out_file)
            if start_pid > len(prompts):
                print(f"[{mkey}] All prompts are already completed. Skipping.")
                continue
            print(f"[{mkey}] Resume mode: starting from Prompt {start_pid}")
            remaining_prompts = [p for p in prompts if p["prompt_id"] >= start_pid]
        else:
            if out_file.exists():
                out_file.unlink()
            print(f"[{mkey}] Fresh mode: starting from Prompt 1")
            remaining_prompts = prompts

        print(f"==> {mkey}: {len(remaining_prompts)} prompts to generate")
        with cf.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(generate_samples, mkey, p) for p in remaining_prompts]
            for task in futures:
                write_jsonl(out_file, task.result())

        print(f"[{mkey}] Finished. Results saved to {out_file}")

if __name__ == "__main__":
    main()
