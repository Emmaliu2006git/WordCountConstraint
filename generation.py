'''You are a rule-following model. You must produce output in the following strict format:

1. The response must begin with the exact token sequence:
   <-START->

2. Immediately after <-START->, there must be NO text, NO titles, and NO whitespace
   before the first <-SECTION-> tag.

3. Use <-SECTION-> to separate major parts of the response. Every <-SECTION-> must be followed by meaningful text on a newline.

4. The response must end with:
   <-END->

5. Do NOT include anything outside these tags.

Example of VALID format:
<-START->

<-SECTION-> 
This is the first section.

<-SECTION-> 
This is the second section.

<-END->

Example of INVALID format (do not do this):
<-START-> Introduction <-SECTION-> ...
<-START->  <-SECTION-> (empty section)
<-SECTION-> ... <-END-> extra text

If any rule is violated, your answer will be discarded and regenerated.
'''

import json
import time
from pathlib import Path
import sys
from openai import OpenAI
import os
from collections import defaultdict
from old_results.Numerical_Script.formatChecking_evaluation import slice_parts

openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
together_client = OpenAI(base_url="https://api.together.xyz/v1", api_key=os.getenv("TOGETHER_API_KEY"))

#path
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
PROMPTS_FILE = DATA_DIR / "segmented.jsonl"
invalid_log_path = DATA_DIR / "invalid_generation_log.txt"
#constants
number_of_samples = 8
TEMPERATURE = 0.6
TOP_P = 0.95
MAX_TOKENS = 8192

models = {
#    "llama4scout": {"name": "meta-llama/Llama-4-Scout-17B-16E-Instruct", "client": together_client},
#    "deepseek-v3": {"name": "deepseek-ai/DeepSeek-V3", "client": together_client},
    "gpt-4.1": {"name": "gpt-4.1", "client": openai_client}
}

SYSTEM_PROMPT = (
    "You are a structured writing assistant. "
    "For each part specified in the user instruction (e.g., 'Part 1', 'Part 2', etc.), "
    "always begin that section with a header in the format '#part <n>' before writing the content. "
    "Do not add any text before or after the tags. "
    "Your output must strictly start with '#part 1' and contain only '#part <n>' headers followed by content—no preface or trailing notes."
)

#jsonl -> list
def load_prompts(path: Path):
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(json.loads(line))
    return out

# call a specific llm, input the model you want to invoke, the input text, and the number of resp genearte at once
def call_llm(model_info: dict, prompt_text: str, n: int = 1):
    client = model_info["client"]
    model_name = model_info["name"]
    #from client object, get attribute base_url: if together ai, sleep 2.5s
    if "together.xyz" in str(getattr(client, "base_url", "")):
        time.sleep(2.5)
    #get response from model
    resp = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt_text}
        ],
        n=n,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS
    )
    # put the n ouput texts in a list
    return [(c.message.content or "").strip() for c in resp.choices]

# does the output text have expected parts
def check_valid_output(output_text: str, expected_parts: int) -> bool:
    try:
        # index + content
        parts = slice_parts(output_text)
        if not parts:
            return False
        return sorted(parts.keys()) == list(range(1, expected_parts + 1))
    except Exception:
        return False

# write to text file, readable
def write_txt_line(f, prompt_id, prompt_type, sample_id, text):
    f.write(f"##prompt_id: {prompt_id}, type: {prompt_type}, sample_id: {sample_id}\n")
    f.write("********\n")
    f.write(text.strip() + "\n\n")
    f.flush()

# write to json, for data processing
def write_jsonl_line(f, prompt_row, sample_id, text):
    '''
    prompt id
    prompt type
    sample id
    verification
    output
    '''
    record = {
        "prompt_id": prompt_row["prompt_id"],
        "prompt_type": prompt_row["prompt_type"],
        "sample_id": sample_id,
        "verification": prompt_row.get("verification", {}),
        "output": text
    }
    f.write(json.dumps(record, ensure_ascii=False) + "\n")
    f.flush()

#given a single prompt
#pass in model info, prompt row, txt_file and jsonl_file to be update
def generate_for_prompt(model_key, model_info, prompt_row, txt_file, jsonl_file):
    valid_samples = 0
    total_generated = 0
    sample_id = 1
    #get expected parts
    expected_parts = int(prompt_row["verification"]["part_number"])
    # store the log under data folder
    log_path = DATA_DIR / f"{model_key}_generation_log.txt"
    while valid_samples < number_of_samples:
        #remaining amount to generate
        n_to_generate = number_of_samples - valid_samples
        new_texts = call_llm(model_info, prompt_row["prompt"], n=n_to_generate)
        total_generated += len(new_texts)
        # write to log
        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(
                f"[{model_key}] pid {prompt_row['prompt_id']} size={len(new_texts)} "
                f"valid_so_far={valid_samples}/{number_of_samples} total_generated={total_generated}\n"
            )
        # for every response generated
        for text in new_texts:
            # if valid, update txt_file and jsonl_file
            if check_valid_output(text, expected_parts):
                write_txt_line(txt_file, prompt_row["prompt_id"], prompt_row["prompt_type"], sample_id, text)
                write_jsonl_line(jsonl_file, prompt_row, sample_id, text)
                #update valid samples
                valid_samples += 1
                sample_id += 1
            # write to invalid log
            else: 
                with open(invalid_log_path, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        f"\n[{model_key}] pid {prompt_row['prompt_id']} sample={sample_id}\n"
                        f"{text}\n{'-'*80}\n"
                    )
        print(f"[{model_key}] pid {prompt_row['prompt_id']} progress: {valid_samples}/8 valid (total {total_generated})")
    print(f"[{model_key}] pid {prompt_row['prompt_id']} reached 8 valid after {total_generated} generations.")

def main():
    prompts = load_prompts(PROMPTS_FILE)
    # for every model
    for mkey, minfo in models.items():
        print(f"==> Generating outputs with {mkey} ...")
        txt_path = DATA_DIR / f"{mkey}_output.txt"
        jsonl_path = DATA_DIR / f"{mkey}_output.jsonl"
        log_path = DATA_DIR / f"{mkey}_generation_log.txt"
        # if txt, jsonl, or log file already exists, delete them before starting a new round
        if txt_path.exists():
            txt_path.unlink()
        if jsonl_path.exists():
            jsonl_path.unlink()
        if log_path.exists():
            log_path.unlink()
        with txt_path.open("a", encoding="utf-8") as txt_file, jsonl_path.open("a", encoding="utf-8") as jsonl_file:
            for prompt_row in prompts:
                generate_for_prompt(mkey, minfo, prompt_row, txt_file, jsonl_file)
        print(f"Done, saved to {txt_path} and {jsonl_path}\n")

if __name__ == "__main__":
    main()
