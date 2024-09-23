import asyncio
import aiohttp
import json
import argparse
import hashlib
import time
import atexit
from tqdm import tqdm
from litgpt.prompts import MicroStories
from loguru import logger


def hash_prompt(prompt):
    return hashlib.md5(prompt.encode()).hexdigest()


cache = {}
error_cache = {}


def save_caches():
    with open("dpo_cache.json", "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)
    with open("error_cache.json", "w", encoding="utf-8") as f:
        json.dump(error_cache, f, ensure_ascii=False, indent=2)
    logger.info("缓存已保存")


atexit.register(save_caches)


async def generate_response(session, prompt, semaphore):
    prompt_hash = hash_prompt(prompt)
    if prompt_hash in cache:
        return cache[prompt_hash]

    async with semaphore:
        try:
            async with session.post(
                "http://127.0.0.1:8000/predict", json={"prompt": prompt}
            ) as response:
                result = await response.json()
                cache[prompt_hash] = result
                return result
        except Exception as e:
            error_msg = f"生成响应时出错: {str(e)}"
            logger.error(error_msg)
            error_cache[prompt_hash] = error_msg
            return None


async def main(concurrency, test_mode):
    global cache, error_cache
    ms = MicroStories()

    with open(
        "../../Data/TinyStoriesInstruct/sft_data_v2.json", "r", encoding="utf-8"
    ) as f:
        sft_data = json.load(f)

    if test_mode:
        sft_data = sft_data[:100]

    # 读取缓存
    try:
        with open("dpo_cache.json", "r", encoding="utf-8") as f:
            cache = json.load(f)
            logger.info(f"加载缓存: {len(cache)}条")
    except FileNotFoundError:
        cache = {}

    try:
        with open("error_cache.json", "r", encoding="utf-8") as f:
            error_cache = json.load(f)
    except FileNotFoundError:
        error_cache = {}

    semaphore = asyncio.Semaphore(concurrency)

    async with aiohttp.ClientSession() as session:
        tasks = []
        for i, case in enumerate(tqdm(sft_data, desc="生成DPO数据")):
            prompt = ms.apply(prompt=case["instruction"], input=case["input"])
            task = asyncio.create_task(generate_response(session, prompt, semaphore))
            tasks.append(task)

            # 每处理100个样本保存一次缓存
            if (i + 1) % 100 == 0:
                save_caches()

        responses = await asyncio.gather(*tasks)

    dpo_data = []
    for case, response in zip(sft_data, responses):
        prompt = ms.apply(prompt=case["instruction"], input=case["input"])
        dpo_sample = {
            "prompt": prompt,
            "rejected": response.get("output") or response.get("rejected"),
            "chosen": case["output"],
        }
        dpo_data.append(dpo_sample)

    # 保存错误缓存
    save_caches()  # 最后再保存一次缓存

    output_file = "dpo_data_test.json" if test_mode else "dpo_data.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dpo_data, f, ensure_ascii=False, indent=2)

    logger.info(f"DPO数据已生成并保存到 {output_file}")
    logger.info(f"缓存已更新并保存到 dpo_cache.json")
    logger.info(f"错误缓存已保存到 error_cache.json")

    end_time = time.time()
    execution_time = end_time - start_time
    logger.info(f"总执行时间: {execution_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="生成DPO数据")
    parser.add_argument("--concurrency", type=int, default=10, help="并发数量")
    parser.add_argument("--test", action="store_true", help="测试模式")
    args = parser.parse_args()

    logger.add("generate_dpo_data.log", rotation="500 MB")
    start_time = time.time()
    asyncio.run(main(args.concurrency, args.test))
