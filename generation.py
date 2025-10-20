
"""{
  "role": "system",
  "content": (
    "You are a structured writing assistant. "
    "For each part specified in the user instruction (e.g., 'Part 1', 'Part 2', etc.), "
    "always begin that section with a header in the format '#part <n>' before writing the content. "
    "Do not add any text before or after the tags."
  )
},
{
  "role": "user",
  "content ": prompt
} """


import json
import time
from pathlib import Path
import sys
import concurrent.futures as cf
from openai import OpenAI
import os

# ---------------------------------------------------------------------------
# API Clients
# ---------------------------------------------------------------------------

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
together_client = OpenAI(base_url="https://api.together.xyz/v1", api_key=os.getenv("TOGETHER_API_KEY"))

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROMPTS_FILE = DATA_DIR / "segmented.jsonl"

# Output params
number_of_samples = 8
TEMPERATURE = 1.0
TOP_P = 1.0
MAX_TOKENS = 2800  # ~2000 words

models = {
    "llama4scout": {"name": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "client": together_client},
    "deepseek-v3": {"name": "deepseek-ai/DeepSeek-V3", "client": together_client},
    "gpt-4.1": {"name": "gpt-4.1", "client": openai_client}  # minimal fix
}

SYSTEM_PROMPT = (
    "You are a structured writing assistant. "
    "For each part specified in the user instruction (e.g., 'Part 1', 'Part 2', etc.), "
    "always begin that section with a header in the format '#part <n>' before writing the content. "
    "Do not add any text before or after the tags. "
    "Your output must strictly start with '#part 1' and contain only '#part <n>' headers followed by content—no preface or trailing notes."
)

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_prompts(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

def call_llm(model_info: dict, prompt_text: str) -> str:
    client = model_info["client"]
    model_name = model_info["name"]
    # gentle pacing for Together
    if "together.xyz" in str(getattr(client, "base_url", "")):
        time.sleep(1.8)
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    return (resp.choices[0].message.content or "").strip()

def generate_samples(model_key: str, prompt_row: dict):
    results = []
    for sid in range(1, number_of_samples + 1):
        while True:
            try:
                text = call_llm(models[model_key], prompt_row["prompt"])
                formatted = (
                    f"##prompt_id: {prompt_row['prompt_id']}, type: {prompt_row['prompt_type']}\n"
                    f"********\n{text}\n\n"
                )
                break
            except Exception as e:
                print(f"[{model_key}] pid {prompt_row['prompt_id']} error: {e}", file=sys.stderr)
                time.sleep(6)  # brief backoff before retry
        print(f"[{model_key}] generated prompt {prompt_row['prompt_id']} sample {sid}")
        results.append(formatted)
    return results

def write_text(path: Path, texts):
    mode = "a" if path.exists() else "w"
    with path.open(mode, encoding="utf-8") as f:
        for t in texts:
            f.write(t)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    prompts = load_prompts(PROMPTS_FILE)
    # reduced concurrency to ease Together rate limits
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        for mkey in models:
            out_file = DATA_DIR / f"{mkey}_output.txt"
            if out_file.exists():
                out_file.unlink()
            print(f"==> Generating outputs with {mkey} ...")
            futures = [pool.submit(generate_samples, mkey, p) for p in prompts]
            for fut in futures:
                write_text(out_file, fut.result())
            print(f"Done, saved to {out_file}")

if __name__ == "__main__":
    main()
