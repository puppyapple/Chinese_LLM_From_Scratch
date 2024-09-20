# 从零手搓中文大模型｜🚀Day11

之前已经把`SFT`阶段给跑通了，尽管整体效果差强人意，但至少证明在这么小的参数量级上也是可行的。

接下来我继续尝试一下`DPO`阶段，但是首先依然得**搞数据**。

## DPO数据构造

`DPO`数据主要是需要获得`rejected`和`chosen`的数据对。

`chosen`的数据很好说，直接使用`SFT`数据里的`response`即可。

而`rejected`的数据其实就是就是回答质量相对较差的数据，很容易就能想`SFT`之后的模型根据`prompt`给出的`response`肯定是质量低于`ground truth`的，天然就可以作为`rejected`的数据。

构造路径倒是很容易，但是根据之前跑生成的经验，单条`prompt`数据生成`response`的时间大概在0.5秒左右，如果使用`SFT`数据全量（在我机制的数据增强之下从1.5w变成了7w多）生成`DPO`数据，那么可能需要10小时左右的时间。

那么并发生成就显得尤为重要，可行的方法有两种：
1. 加载多个模型的实例，将数据均等切分成多个`chunks`每个模型生成一部分数据，最后再合并。
2. 将模型部署成`API`接口，使用`aiohttp`异步请求。

> 其实应该同时用上`batch inference`，但`litgpt`库这块的`feature`还在开发中，我自己魔改担心搞不定，就先不尝试了。

显然后者的稳定性会更好，那么话不多说，直接上代码👇：


```python
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
```

上面的脚本做了这样几件事：
1. 构建了一个`generate_response`的函数，用于根据`prompt`生成`response`
2. 对上面的函数做了异步调度，可以控制并发数量来生成`response`
3. 设置了对已经生成的样本的结果的缓存以及异常样本的缓存（每100个样本保存一次，且如果脚本异常退出`atexit`会自动保存）
4. 最后将`SFT`数据和生成的`response`进行拼接，得到最终的`DPO`数据。


```python
# ! python generate_dpo_data.py  --concurrency 25 
```

聪明的你肯定会问了：异步调度啥的倒是都有了，**可我哪儿来的接口呢？**

其实`litgpt`库同时也提供了模型的`serving`功能，只要安装了额外的`litserve`依赖，就可以一键部署：


```python
# ! litgpt serve out/custom-model/final
```

不过这样得到的服务是单实例的，无法满足我们批量刷数据的需求。

大家别忘了，我们的模型尺寸只有`0.044B`，显存占用才`600M`，这意味着我们在一张卡上可以轻松部署多个实例。

其实`litserve`是支持多`workers`的，不过在`litgpt`库集成的时候没有暴露出参数，问题不大，我们自己基于[litgpt里的serve.py](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/deploy/serve.py)魔改一下就好了。

代码太长这里就不完整地贴出了，感兴趣的可以看[这里](https://github.com/puppyapple/Chinese_LLM_From_Scratch/blob/main/Journey/Day11/service.py)。

修改其实很简单，就是把`workers_per_device`参数暴露了出来，这样就可以在启动服务的时候指定`workers_per_device`的值了。


```python
@click.command()
@click.option("--checkpoint_dir", type=str)
@click.option("--quantize", type=str, default=None)
@click.option("--precision", type=str, default="bf16-true")
@click.option("--temperature", type=float, default=0.8)
@click.option("--top_k", type=int, default=50)
@click.option("--top_p", type=float, default=1.0)
@click.option("--max_new_tokens", type=int, default=50)
@click.option("--devices", type=int, default=1)
@click.option("--workers_per_device", type=int, default=20)
@click.option("--port", type=int, default=8000)
@click.option("--stream", type=bool, default=False)
@click.option("--accelerator", type=str, default="auto")
def run_server(
    checkpoint_dir: Path,
    quantize: Optional[
        Literal["bnb.nf4", "bnb.nf4-dq", "bnb.fp4", "bnb.fp4-dq", "bnb.int8"]
    ] = None,
    precision: Optional[str] = None,
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 1.0,
    max_new_tokens: int = 50,
    devices: int = 1,
    port: int = 8000,
    accelerator: str = "auto",
    workers_per_device: int = 20,
    stream: bool = False,
    access_token: Optional[str] = None,
) -> None:
    # ...
    pass


if __name__ == "__main__":
    run_server()
```

我设置了`25`个`workers`，然后生成的脚本配置了`--concurrency 25`。

运行时的整体`GPU`占用是`20G`左右。

![image](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/image.png)

截止写这篇文章，数据还在运行中，具体耗时多久等我跑完了在同步给大家。

> 另外我调研的时候发现`litserve`的`batch inference`其实已经支持了，后面有时间尝试一下，如果有效会更新到项目里。
> 仓库里我也同时提供了单模型和多模型实例来跑数据的脚本。
>
## 小结

今天的内容其实很简单，就是构造了`DPO`数据，并且通过异步请求的方式提高了数据构造的效率。

等数据跑完了我会着手进行`DPO`的训练。

由于`litgpt`库自身还不支持`DPO`，所以这部分需要完全自己`DIY`了，可能会稍微费点劲，请大家拭目以待！


