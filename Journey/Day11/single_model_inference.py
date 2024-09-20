import json
import hashlib
from litgpt import LLM
from tqdm import tqdm
from litgpt.prompts import MicroStories


def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()


ms = MicroStories()
llm = LLM.load(model="../../Experiments/Output/sft/microstories/mask_prompt_5e-4/final")

sft_data = json.load(
    open("../../Data/TinyStoriesInstruct/sft_data_v2.json", "r", encoding="utf-8")
)

try:
    with open("dpo_cache.json", "r", encoding="utf-8") as f:
        cache = json.load(f)
except FileNotFoundError:
    cache = {}

try:
    for case in tqdm(sft_data):
        prompt = ms.apply(prompt=case["instruction"], input=case["input"])
        hash_key = hash_prompt(prompt)
        if hash_key in cache:
            continue
        else:
            generated = llm.generate(prompt=prompt, max_new_tokens=350)
            dpo_sample = {
                "prompt": prompt,
                "rejected": generated,
                "chosen": case["output"],
            }
            cache[hash_key] = dpo_sample

except Exception as e:
    print(repr(e))
    with open("dpo_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False)
