import json
import time
from pathlib import Path
import sys
import concurrent.futures as cf
from openai import OpenAI
import os
#use of openrouter
client = OpenAI(
    base_url="https://openrouter.ai/api/v1", 
    api_key = os.getenv("OPENROUTER_API_KEY")           
)

# ---------------------------------------------------------------------------
# path and constants
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent        
DATA_DIR = ROOT / "data"                            
PROMPTS_FILE = DATA_DIR / "prompts.jsonl" 

# constants
number_of_samples = 8  # 8 outputs/prompt/model
TEMPERATURE = 0.7 
TOP_P = 1.0  
MAX_TOKENS = 2800    # allow the maximum of 2000 words

#values are name on together
models = {
    "llama4scout": "meta-llama/llama-4-scout:free",
    "deepseek-v3": "deepseek/deepseek-chat-v3-0324:free",
    "gpt-4.1":     "openai/gpt-4.1"
}

SYSTEM_PROMPT = "You are an AI model that strictly follows user instructions. Do not answer anything beyond the user's request."

# ---------------------------------------------------------------------------
# utils
# ---------------------------------------------------------------------------
def load_prompts(path: Path):
    #store the dics
    prompts_list = []
    # open the file
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            prompt_dict = json.loads(line)
            prompts_list.append(prompt_dict)
    return prompts_list

#given a model and an input text, get the output tet
def call_llm(model_name: str, prompt_text: str) -> str:
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
    return resp.choices[0].message.content #way to get the output message from an output dict

def generate_samples(model_key: str, prompt_row: dict):
    result = []
    for sid in range(1, number_of_samples + 1):
        while True:
            try:
                text = call_llm(models[model_key], prompt_row["prompt"]) #get the output text
                break                                     
            except Exception as e:
                print(f"[{model_key}] pid {prompt_row['prompt_id']} err: {e}",
                      file=sys.stderr) #print out the error message
                time.sleep(10)
        print(f"[{model_key}] Generated Output for Prompt {prompt_row['prompt_id']} Sample {sid}")
        result.append({
            "prompt_id": prompt_row["prompt_id"], #i (1-20)
            "sample_id": sid, #j(1-8)
            "text": text
        })
    return result

def write_jsonl(path: Path, records):
    with path.open("a") as f:
        for rec in records:
            json.dump(rec, f, ensure_ascii=False)
            f.write("\n")

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    prompts = load_prompts(PROMPTS_FILE)  # get the 20 prompts in a list 

    for mkey in models:                      
        out_file = DATA_DIR / f"{mkey}_outputs.jsonl"
        # clear the folder
        if out_file.exists():
            out_file.unlink()           

        print(f"==> Generating outputs with {mkey} ...")
        # 4 as experience value 
        with cf.ThreadPoolExecutor(max_workers=4) as pool: #4 threads in the pool
            futures = [] #all the tasks
            for p in prompts: # every line in json
                task = pool.submit(generate_samples, mkey, p) #task = processing one intput prompt
                futures.append(task)
            for task in futures: #every task has 8 lines
                write_jsonl(out_file, task.result())

        print(f"Done, saved to {out_file}")

if __name__ == "__main__":
    main()
