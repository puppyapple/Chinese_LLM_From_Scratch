import json
import multiprocessing
from functools import partial
from litgpt import LLM
from litgpt.prompts import MicroStories
import click
import torch

# 设置多进程启动方法为'spawn'
multiprocessing.set_start_method("spawn", force=True)


def init_model():
    model = LLM.load(
        model="../../Experiments/Output/sft/microstories/mask_prompt_5e-4/final"
    )
    return model


def process_chunk(model, chunk):
    ms = MicroStories()
    results = []
    for case in chunk:
        prompt = ms.apply(prompt=case["instruction"], input=case["input"])
        with torch.no_grad():
            response = model.generate(prompt=prompt, max_new_tokens=350)
        results.append(
            {"prompt": prompt, "rejected": response, "chosen": case["output"]}
        )
    return results


@click.command()
@click.option("-n", "--num_processes", default=4, help="并发进程数")
@click.option("--test", is_flag=True, help="测试模式，只处理前100条数据")
def main(num_processes, test):
    # 加载SFT数据
    with open(
        "../../Data/TinyStoriesInstruct/sft_data_v2.json", "r", encoding="utf-8"
    ) as f:
        sft_data = json.load(f)

    if test:
        sft_data = sft_data[:100]

    # 确定进程数量
    n_processes = min(multiprocessing.cpu_count(), num_processes)

    # 初始化模型
    model = init_model()

    # 使用partial创建一个新的函数，将model作为第一个参数
    process_chunk_with_model = partial(process_chunk, model)

    # 将数据分成n_processes份
    chunk_size = len(sft_data) // n_processes
    chunks = [sft_data[i : i + chunk_size] for i in range(0, len(sft_data), chunk_size)]

    # 使用进程池并行处理数据
    with multiprocessing.Pool(n_processes) as pool:
        results = pool.map(process_chunk_with_model, chunks)

    # 合并结果
    dpo_samples = [item for sublist in results for item in sublist]

    # 保存结果
    output_file = "dpo_samples_test.json" if test else "dpo_samples.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dpo_samples, f, ensure_ascii=False, indent=2)

    print(f"处理完成，结果已保存到 {output_file}")


if __name__ == "__main__":
    main()
