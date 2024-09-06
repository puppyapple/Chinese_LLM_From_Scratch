# 从零手搓中文大模型｜🚀 Day07

## SFT 数据准备

`TinyStories`数据集其实也提供了[Instruct数据](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)，我可以基于这个数据集在之前的预训练模型上进行指令微调。

先看看数据集的格式：


```python
! head -10 ../../Data/TinyStoriesInstruct/TinyStories-Instruct-valid.txt
```

这些指令有四种类型：
1. 一个单词列表，包含在故事中。
2. 一个句子，应该出现在故事的某个地方。
3. 一个特征列表（可能的特征：对话、坏结局、道德价值、情节转折、伏笔、冲突）。
4. 一个简短的总结（1-2行）。

现在面临两个问题：
- 数据集是英文的，我需要想办法给整成中文的。
- 数据集的形式和主流的SFT数据集不太一样，需要做一些适配。

> 个人理解这里是因为这里的指令相对单一（生成故事），只是约束有一些区别，所以作者采取了简单的拼接方式。
>
> 这里出于学习的目的还是往主流的SFT数据集上靠拢。

### 吴恩达老师的翻译Agent测试

这里直接试了下[吴恩达老师的translation-agent](https://github.com/andrewyng/translation-agent)项目（`translation-agent.py`文件），使用的是`gpt-4o-mini`的`api`（也尝试过`Ollama`本地部署的`qwen14b`、`qwen7b`，相对来说不太稳定）。

可以看到这里单次翻译的耗时在10秒左右（因为单词翻译的时候`agent`逻辑里有多次`api`调用），因此这里为了后面能够并发调用刷数据，我将代码全部改造成了`async`的异步调用。

大家如果有其他的翻译`api`或者模型也可以替换，这里纯属心血来潮玩一玩儿。

`translation-agent`项目其实只有一个`utils.py`文件，但因为太长了，这里就不把改造后的代码贴出来了，有兴趣的同学可以去仓库里查看。


```python
from translation_agent import translate

text = """
Random sentence: They are very excited and want to fly too.
Features: Dialogue
Summary: Tom and Anna are excited to go on a holiday with their parents, and they fly on a big plane to a place with sun and sand.
Story: 
Tom and Anna are brother and sister. They like to play with their toys and read books. They are very happy because they are going on a holiday with their mum and dad. They will fly on a big plane to a place with a lot of sun and sand.
The day of the holiday comes and they pack their bags. They go to the airport and wait for their plane. They see many other planes flying in the sky. They are very excited and want to fly too.
"Look, Anna, that plane is so big and fast!" Tom says.
"Yes, Tom, and it has wings and a tail. I wonder where it is going," Anna says.
They hear their mum call them. "Come on, kids, it's time to board our plane. We have to show our tickets and go through the gate."
They follow their mum and dad and get on their plane. They find their seats and buckle their belts. They look out the window and see the ground and the cars and the people. They hear the pilot say something on the speaker.
"Hello, everyone, this is your pilot speaking. Welcome aboard flight 123 to Sunny Beach. We are ready to take off. Please sit back and enjoy the flight."
The plane starts to move and makes a loud noise. Tom and Anna feel the plane go faster and faster. They see the ground get smaller and smaller. They see the clouds get closer and closer. They are flying!
"Wow, Anna, we are flying! We are in the sky!" Tom says.
"I know, Tom, it's amazing! We are so high! Look, there is the sun!" Anna says.
They smile and laugh and clap their hands. They are not sad at all. They are very happy. They are flying to their holiday.
"""


result = await translate(
    source_lang="English",
    target_lang="Chinese",
    source_text=text,
    country="China",
)
print(result)
```
```
ic| num_tokens_in_text: 416
ic| 'Translating text as a single chunk'


随机句子：他们非常兴奋，也想飞起来。  
特点：对话  
摘要：汤姆和安娜很兴奋要和父母一起度假，他们乘坐一架大飞机飞往阳光明媚、沙滩众多的地方。  
故事：  
汤姆和安娜是兄妹。他们喜欢玩玩具和读书。他们非常高兴，因为他们要和妈妈爸爸一起去度假。他们将乘坐一架大飞机去一个阳光明媚、沙滩众多的地方。  
度假日终于到了，他们收拾好行李。他们去机场，等待他们的飞机。他们看到许多飞机在天空中飞。他们非常兴奋，也想飞起来。  
“看，安娜，那架飞机真大又快！”汤姆说。  
“是的，汤姆，它有翅膀和尾巴。我想知道它要去哪里，”安娜说。  
他们听到妈妈叫他们。“快来，孩子们，是时候登机了。我们得出示机票，通过登机口。”  
他们跟着妈妈和爸爸上了飞机。他们找到座位，系好安全带。他们望向窗外，看到地面、汽车和行人。他们听到飞行员在扬声器上说话。  
“大家好，我是你们的机长。欢迎乘坐123航班前往阳光海滩。我们准备起飞。请坐好，祝大家旅途愉快。”  
飞机开始移动，发出轰鸣声。汤姆和安娜感到飞机越来越快。他们看到地面变得越来越小。他们看到云朵越来越近。他们飞起来了！  
“哇，安娜，我们飞起来了！我们在天空中！”汤姆说。  
“我知道，汤姆，真是太神奇了！我们这么高！看，那是太阳！”安娜说。  
他们微笑、欢笑，拍着手。他们一点都不难过。他们非常快乐。他们正在飞往度假地。
```



### 数据采样

我先看看训练集有多少条数据，可以发现文本都是以`<|endoftext|>`结尾的，所以通过统计`endoftext`的个数就可以知道数据集的条数。


```python
! grep -o "endoftext" ../../Data/TinyStoriesInstruct/TinyStories-Instruct-train.txt  | wc -l 
```
```
2476532
```


接近250w的量级有点大（因为微软的论文里是直接在整个数据集上做的`pretrain`的）。

其实很多研究表明，`SFT`数据的量级不重要，质量够高的时候即使很少的数据也能训练出很好的效果。

所以这里我打算随机抽取11000条数据来试试。

我的策略如下：
1. 遍历`train`数据集，让四类指令的组合尽量均衡（需要先统计指令组合的的分布）
2. 用得到的11000条数据调用上面的`translation-agent`进行翻译
3. 将翻译后的数据整理成`SFT`数据集的`json`格式

先来做数据的采样：


```python
from collections import Counter
import random


def count_field_combinations(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    blocks = content.split("<|endoftext|>")
    combinations = []

    for block in blocks:
        fields = set()
        if "Words:" in block:
            fields.add("Words")
        if "Random sentence:" in block:
            fields.add("Random sentence")
        if "Features:" in block:
            fields.add("Features")
        if "Summary:" in block:
            fields.add("Summary")

        if fields:  # 只有当字段不为空时才添加组合
            combinations.append(frozenset(fields))

    return Counter(combinations)


def sample_data(file_path, total_samples=11000):
    with open(file_path, "r", encoding="utf-8") as file:
        content = file.read()

    blocks = content.split("<|endoftext|>")
    blocks = [block.strip() for block in blocks if block.strip()]  # 移除空块

    combinations = count_field_combinations(file_path)
    combination_more_than_1 = {k: v for k, v in combinations.items() if v > 1}
    samples_per_combination = total_samples // len(combination_more_than_1)

    sampled_data = []
    for combination in combinations:
        matching_blocks = [
            block for block in blocks if set(get_fields(block)) == set(combination)
        ]
        sampled_data.extend(
            random.sample(
                matching_blocks, min(samples_per_combination, len(matching_blocks))
            )
        )

    return sampled_data


def get_fields(block):
    fields = set()
    if "Words:" in block:
        fields.add("Words")
    if "Random sentence:" in block:
        fields.add("Random sentence")
    if "Features:" in block:
        fields.add("Features")
    if "Summary:" in block:
        fields.add("Summary")
    return fields
```

执行一下看看效果（为了有备无患，多采样了5000条数据），耗时1-2分钟，肯定还有优化空间，但是可以接受。

同时将采样后的数据保存为`pkl`文件，方便后续使用。


```python
import pickle

sft_raw = sample_data(
    "../../Data/TinyStoriesInstruct/TinyStories-Instruct-train.txt", 15000
)
print(f"采样数据总数: {len(sft_raw)}")

pickle.dump(sft_raw, open("sft_raw.pkl", "wb"))
```

```
采样数据总数: 15001
```



### 批量翻译

接下来就可以调用`translation-agent`进行翻译了。

这里我除了用异步加速，还使用了`json`文件缓存来避免重复翻译（`gpt-4o-mini`的`api`也不算便宜，能省则省）。


```python
import json
import aiofiles
import asyncio

cache_file = "translation_cache.json"


async def translate_and_cache(block, cache, semaphore):
    cache_key = hash(block)

    if str(cache_key) in cache:
        return cache[str(cache_key)]

    async with semaphore:
        try:
            result = await translate(
                source_lang="English",
                target_lang="Chinese",
                source_text=block,
                country="China",
            )
            cache[str(cache_key)] = result
            return result
        except Exception as e:
            print(f"翻译失败: {e}")
            return None


async def batch_translate(sampled_data, cache_file, max_workers=10):
    translated_data = []

    try:
        async with aiofiles.open(cache_file, "r") as f:
            cache = json.loads(await f.read())
    except (FileNotFoundError, json.JSONDecodeError):
        cache = {}

    semaphore = asyncio.Semaphore(max_workers)
    tasks = [translate_and_cache(block, cache, semaphore) for block in sampled_data]
    results = await asyncio.gather(*tasks)

    translated_data = [result for result in results if result]

    async with aiofiles.open(cache_file, "w") as f:
        await f.write(json.dumps(cache, ensure_ascii=False, indent=2))

    return translated_data


translated_data = await batch_translate(sft_raw, cache_file, max_workers=100)
```

使用了100路的并发，翻译了15000条数据，耗时48分钟，也就是大概每分钟翻译300条数据。

### 后续处理

翻译完成了，最后一步就是将数据整理成`SFT`数据集的格式。


```python
from collections import Counter
from pprint import pprint

instruction_template = """按照给定的要求讲故事，
其中‘摘要’表示故事的总结，
‘单词/词汇/关键词’表示故事中必须包含的单词，
‘随机句子’表示故事中必须包含的句子，
‘特征/特点’表示故事的特征，如对话、坏结局、道德价值、情节转折、伏笔、冲突等。
"""


def process_translated_data(input_file, output_file):
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    constraint_keys = Counter()

    for key, value in data.items():
        if "故事：" not in value:
            continue
        parts = value.split("故事：")

        if len(parts) == 2:
            input_text = parts[0].strip()
            output_text = parts[1].strip()

            # 提取约束描述文本的关键字段
            lines = input_text.split("\n")
            for line in lines:
                if "：" in line:
                    key, _ = line.split("：", 1)
                    constraint_keys[key.strip()] += 1

            processed_item = {
                "instruction": instruction_template,
                "input": input_text,
                "output": output_text,
            }

            processed_data.append(processed_item)

    # 将处理后的数据写入输出文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)

    return processed_data, constraint_keys


processed_data, constraint_keys = process_translated_data(
    "translation_cache.json", "../../Data/TinyStoriesInstruct/sft_data.json"
)

pprint(constraint_keys.most_common(20))
```
```
('摘要', 8433),
('随机句子', 5372),
('词汇', 4824),
('特点', 4603),
('特征', 1269),
('单词', 1073),
('总结', 1001),
('关键词', 349),
('随机的句子', 246),
('随机句', 196),
('故事特点', 146),
('主题', 122),
('词语', 108),
('随便一句话', 95),
('随机一句话', 87),
('随机的一句话', 24),
('词', 21),
('故事特征', 20),
('随机句子是', 19),
('随便说一句', 17)]
```



看一看处理的结果，这样就和经典的`SFT`数据格式一致了。


```python
pprint(processed_data[0])
```

```
{
    'input': '特点：对话  \n'
              '摘要：蒂米和妈妈一起去商店，对商店里所有的玩具和糖果感到惊讶。他请求触摸一个玩具，妈妈允许他，这让他非常开心。  \n'
              '词汇：触摸、商店、宽敞',
    'instruction': '按照给定的要求讲故事，\n'
                   '其中‘摘要’表示故事的总结，\n'
                   '‘单词/词汇/关键词’表示故事中必须包含的单词，\n'
                   '‘随机句子’表示故事中必须包含的句子，\n'
                   '‘特征/特点’表示故事的特征，如对话、坏结局、道德价值、情节转折、伏笔、冲突等。\n',
    'output': '很久以前，有一个小男孩名叫蒂米。蒂米喜欢在外面玩耍和探索。  \n'
              '一天，蒂米和妈妈一起去商店。商店非常大，宽敞。蒂米对他看到的玩具和糖果真是太多了，感到惊讶。  \n'
              '突然，蒂米看到一个他非常想触摸的玩具。“妈妈，我可以触摸那个玩具吗？”他问。  \n'
              '“当然可以啊，蒂米，”妈妈说。蒂米非常开心，他轻轻摸了摸玩具。摸起来软软的，特别有弹性。  \n'
              '离开商店后，蒂米对妈妈说他有多喜欢和她一起逛商店。“我玩得可开心了，摸玩具真有意思，”他说。妈妈微笑着把他抱住了。'}
```


## 小结
1. 基于`TinyStories`的`Instruct`数据进行指令组合层面均衡的采样，获得了15000条原始数据
2. 构造了翻译函数，异步使用吴恩达老师的`translation-agent`对数据进行翻译
3. 基于翻译后的数据，构造了经典格式的`SFT`数据集