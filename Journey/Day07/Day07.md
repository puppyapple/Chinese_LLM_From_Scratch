# 从零手搓中文大模型｜🚀 Day07

## SFT 数据准备

`TinyStories`数据集其实也提供了[Instruct数据](https://huggingface.co/datasets/roneneldan/TinyStoriesInstruct)，我可以基于这个数据集在之前的预训练模型上进行指令微调。

先看看数据集的格式：


```python
! head -10 ../../Data/TinyStoriesInstruct/TinyStories-Instruct-train.txt
```

    Features: Dialogue
    Words: quit, oak, gloomy
    Summary: Sara and Ben were playing in the park, but Sara wanted to go home because it was cold and dark. Ben convinced her to stay and play, but eventually agreed to go home and have hot cocoa.
    Story: 
    
    Sara and Ben were playing in the park. They liked to climb the big oak tree and pretend they were birds. They made nests with leaves and twigs and sang songs.
    But today, the sky was gloomy and the wind was cold. Sara felt sad and cold. She wanted to go home and have some hot cocoa.
    "Ben, I want to quit," she said. "It's too cold and dark. Let's go home."
    Ben looked at Sara and frowned. He liked the oak tree and the park. He wanted to stay and play.
    "No, Sara, don't quit," he said. "It's fun here. Look, there's a squirrel. Let's chase it."


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

    ic| num_tokens_in_text: 416
    ic| 'Translating text as a single chunk'


    随机句子：他们非常兴奋，也想飞。  
    特点：对话  
    摘要：汤姆和安娜兴奋地和父母一起去度假，他们乘坐一架大飞机飞往阳光明媚、沙滩细腻的地方。  
    故事：  
    汤姆和安娜是兄妹。他们喜欢玩玩具和读书。他们非常开心，因为他们要和爸爸妈妈一起去度假。他们将乘坐一架大飞机去一个阳光明媚、沙滩细腻的地方。  
    度假的日子到了，他们开始整理行李。他们去机场，等待他们的飞机。他们看到许多其他飞机在天空中飞。他们非常兴奋，也想飞。  
    “看，安娜，那架飞机又大又快！”汤姆说。  
    “是的，汤姆，它有翅膀和尾巴。我想知道它要去哪里，”安娜说。  
    他们听到妈妈叫他们。“快点，孩子们，差不多该登机了。我们必须出示机票，然后通过登机口。”  
    他们跟着爸爸妈妈上了飞机。他们找到自己的座位，系好安全带。他们望向窗外，看到地面、汽车和人。他们听到飞行员在扬声器上说话。  
    “大家好，我是你们的机长。欢迎乘坐123航班前往阳光海滩。我们准备起飞。请坐好，享受旅程。”  
    飞机开始移动，发出轰鸣的声音。汤姆和安娜感觉飞机越来越快。他们看到地面变得越来越小。云朵越来越近。他们在飞！  
    “哇，安娜，我们在飞！我们在天空中！”汤姆说。  
    “我知道，汤姆，太神奇了！我们这么高！看，那里是太阳！”安娜说。  
    他们微笑、欢笑，拍手欢呼。他们一点都不难过。他们非常快乐。他们正在飞往度假的地方。


### 数据采样

我先看看训练集有多少条数据，可以发现文本都是以`<|endoftext|>`结尾的，所以通过统计`endoftext`的个数就可以知道数据集的条数。


```python
! grep -o "endoftext" ../../Data/TinyStoriesInstruct/TinyStories-Instruct-train.txt  | wc -l 
```

    2476532


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

# sft_raw = sample_data(
#     "../../Data/TinyStoriesInstruct/TinyStories-Instruct-train.txt", 15000
# )
sft_raw = pickle.load(open("sft_raw.pkl", "rb"))
print(f"采样数据总数: {len(sft_raw)}")

# pickle.dump(sft_raw, open("sft_raw.pkl", "wb"))
```

    采样数据总数: 15001


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

（这里还发现了个小问题，翻译统一将**总结**字段放到了最后，导致顺序出现了问题，所以这里需要先处理一下。）


```python
import itertools
import json
import random
from collections import Counter
from pprint import pprint

instruction_template = "按照下面输入的约束生成故事"


def process_translated_data(input_file):
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
                "input": f"{input_text}",
                "output": output_text,
            }

            processed_data.append(processed_item)
    # 根据constraint_keys的频率排序，选取出现频率大于10的关键字
    constraint_keys = {k: v for k, v in constraint_keys.items() if v > 10}
    return constraint_keys, processed_data
```


```python
constraint_keys, processed_data = process_translated_data("translation_cache.json")
```


```python
keywords_normalization = {
    "词汇": "词汇",
    "关键词": "词汇",
    "单词": "词汇",
    "词语": "词汇",
    "词": "词汇",
    "字": "词汇",
    "特征": "特征",
    "特点": "特征",
    "故事特点": "特征",
    "故事特征": "特征",
    "对话特点": "特征",
    "主题": "特征",
    "随机句子": "随机句子",
    "随便一句话": "随机句子",
    "随机一句话": "随机句子",
    "随机句": "随机句子",
    "随机的一句话": "随机句子",
    "随机的句子": "随机句子",
    "随机句子是": "随机句子",
    "随便说一句": "随机句子",
    "随便一句": "随机句子",
    "随机句子示例": "随机句子",
    "摘要": "摘要",
    "总结": "摘要",
    "故事概要": "摘要",
}
```


```python
def split_data(data, keys):
    result = []
    current_key = None
    current_content = ""

    for line in data.split("\n"):
        line = line.strip()
        if any(key in line for key in keys):
            if current_key:
                result.append((current_key, current_content.strip()))
            for key in keys:
                if key in line:
                    current_key, current_content = line.split(key, 1)
                    current_key = key.strip()
                    current_content = current_content.strip().lstrip("：").strip()
                    break
        else:
            current_content += " " + line

    if current_key:
        result.append((current_key, current_content.strip()))

    return result


def filter_and_normalize(
    processed_data, constraint_keys, output_file, expand_data=True
):
    final_data = []
    for item in processed_data:
        input_text = item["input"]
        output_text = item["output"]
        has_keyword = False
        for keyword in keywords_normalization:
            if f"{keyword}：" in output_text:
                content = output_text.split(f"{keyword}：")[1].strip()
                input_text += f"\n{keyword}：{content}"
                output_text = output_text.split(f"{keyword}：")[0].strip()
                has_keyword = True
            if f"{keyword}：" in input_text:
                input_text = input_text.replace(
                    f"{keyword}：", f"{keywords_normalization[keyword]}："
                )
                has_keyword = True
        if not has_keyword:
            continue

        # 数据增强
        if expand_data:
            input_tuple_list = split_data(input_text, keywords_normalization)
            if not input_tuple_list:
                continue

            for permutation in itertools.permutations(input_tuple_list):
                new_item = {
                    "instruction": instruction_template,
                    "input": "\n".join(
                        [f"{key}：{value}" for key, value in permutation]
                    ),
                    "output": output_text,
                }
                final_data.append(new_item)
        else:
            item.update({"input": input_text, "output": output_text})
            final_data.append(item)

    # 对结果做一个打乱
    random.shuffle(final_data)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(final_data, f, ensure_ascii=False, indent=2)

    return final_data
```


```python
final_data = filter_and_normalize(
    processed_data,
    keywords_normalization,
    "../../Data/TinyStoriesInstruct/sft_data_v2.json",
    True,
)
```

看一看处理的结果，这样就和经典的`SFT`数据格式一致了。


```python
for i in range(5):
    pprint(final_data[i])
```

    {'input': '随机句子：随机句子：蒂姆的网球水平越来越好，但有时他在错过球的时候会感到不耐烦。\n'
              '特征：对话\n'
              '摘要：蒂姆和他的爸爸一起打网球，但当蒂姆错过球时会感到不耐烦。他的爸爸鼓励他坚持练习，最终蒂姆成功把球打过了网，并为自己感到骄傲。',
     'instruction': '按照下面输入的约束生成故事',
     'output': '从前，有个叫蒂姆的小男孩。蒂姆喜欢和他的爸爸打网球。他们会去公园，来回击球。蒂姆的网球水平越来越好，但有时他在错过球的时候会感到不耐烦。  \n'
               '一天，蒂姆和他的爸爸在打网球，蒂姆错过了很多次球。他变得很不耐烦，甚至哭了起来。他的爸爸说：“别担心，蒂姆。你通过练习会变得更好的。”  \n'
               '爸爸给蒂姆发了个球，蒂姆把球打过了网。他高兴得不得了！蒂姆说：“我成功了，爸爸！”  \n'
               '他的爸爸微笑着说：“没错，你成功了！现在我们继续玩，享受其中的乐趣。”蒂姆感到非常自豪，一直打球，直到该回家的时候。'}
    {'input': '词汇：撒谎，打架，大\n'
              '摘要：艾莉这只大象帮助朋友们蒂米和萨米和解，强调了友谊和一起玩的重要性。\n'
              '随机句子：艾莉和她的朋友们住在一片大丛林里。\n'
              '特征：对话',
     'instruction': '按照下面输入的约束生成故事',
     'output': '从前，有一只叫艾莉的大象。艾莉和她的朋友们住在一片大丛林里。一天，艾莉看到她的朋友老虎蒂米躺在地上。  \n'
               '艾莉问：“蒂米，你怎么躺着？”  \n'
               '“我和蛇萨米打架，”蒂米悲伤地回答。  \n'
               '艾莉看到朋友们打架，心里很难过。她说：“打架可不好，我们应该做朋友，一起玩。”  \n'
               '蒂米同意了艾莉，他们一起去找萨米。当他们找到萨米时，他们互相道歉，重新成为了好朋友。从那天起，他们都一起玩，在丛林里玩得很开心。'}
    {'input': '词汇：滚，比萨，打开\n'
              '摘要：蒂姆试图从一家开着的比萨店拿一块大比萨，但它太大了，所以他决定把它滚走。一只狗看到了比萨，追着蒂姆，吃掉了比萨，留下蒂姆感到伤心。',
     'instruction': '按照下面输入的约束生成故事',
     'output': '一天，一个名叫蒂姆的男孩去了比萨店。他特别喜欢比萨。在比萨店里，他看到桌子上有一个大比萨。它看起来很好吃！  \n'
               '蒂姆说：“哇，我想吃那块比萨！”他试图拿起比萨，但它太大了。所以，他决定把比萨滚走。他把比萨滚出了比萨店。  \n'
               '当蒂姆把比萨滚下街的时候，一只大狗看到了比萨。那只狗很饿。狗说：“我也想吃那块比萨！”狗开始追着蒂姆和他的比萨。  \n'
               '蒂姆跑得很快，但狗跑得更快。狗追上了蒂姆和他的比萨。狗吃掉了整块比萨，蒂姆感到伤心。那天他一口比萨都没吃到。'}
    {'input': '词汇：鼓掌，海洋，危险\n'
              '特征：对话\n'
              '摘要：莉莉和萨姆想在海洋中游泳，但他们的父母说太危险了。他们在岸边和新朋友一起玩球和放风筝，玩得很开心。他们在水中看到一只海豚，了解到海洋既美妙又危险。',
     'instruction': '按照下面输入的约束生成故事',
     'output': '莉莉和萨姆和他们的爸爸妈妈在海滩上。他们喜欢在沙子里玩耍，欣赏海洋。海洋又大又蓝，发出隆隆的声音。  \n'
               '“妈妈，我们可以下水吗？”莉莉问。  \n'
               '“不行，亲爱的，今天水太危险了。有大浪和强流。你们可能会受伤或者迷路，”妈妈说。  \n'
               '“但是我想游泳，妈妈。我游得很好。你教过我怎么游泳，记得吗？”萨姆说。  \n'
               '“我知道，亲爱的，但在海洋里游泳和在游泳池里游泳是不同的。海洋对你们这种小孩来说不安全。你们必须听爸爸妈妈的话，待在岸边，好吗？”爸爸说。  \n'
               '莉莉和萨姆感到难过。他们想在水里玩得开心。他们看到其他小朋友在玩球和放风筝。他们决定加入他们，交一些新朋友。  \n'
               '他们玩球和放风筝玩得特别开心。他们互相扔球，追着风筝跑。他们欢笑、喊叫、欢呼。他们忘记了水，享受着阳光和微风。  \n'
               '不久，到了回家的时间。爸爸妈妈收拾好东西，叫莉莉和萨姆。他们和新朋友道别，感谢他们一起玩。  \n'
               '当他们走向车时，他们看到码头上有一群人。他们在看水里的东西。他们听到了一些鼓掌和欢呼声。  \n'
               '“他们在看什么，爸爸？”莉莉问。  \n'
               '“我们去看看，亲爱的，”爸爸说。  \n'
               '他们走到码头，看到一条大鱼从水里跳出来。它是灰色的，闪闪发亮，鼻子还很长。它是一只海豚。它在波浪中玩耍和跳舞。它发出一些有趣的声音，溅起水花。  \n'
               '“哇，看看那个，萨姆。是一只海豚。太酷了，”莉莉说。  \n'
               '“太神奇了，莉莉。它聪明又友好。它不像水那样危险。真不错，”萨姆说。  \n'
               '他们看了一会儿海豚。每次海豚跳跃、旋转或挥手时，他们都鼓掌欢呼。他们微笑着挥手回应。他们感到快乐和兴奋。  \n'
               '他们那天学到了很多新东西。他们明白了海洋不仅危险，还有很多美妙的地方。他们学到在海滩上有很多东西可以看、可以做和可以享受。他们学到可以在不下水的情况下玩得开心。他们学到可以交新朋友，看到新动物。他们学到可以为海豚鼓掌。'}
    {'input': '特征：对话，道德价值\n'
              '随机句子：一天，班尼看到地上有一把梳子。\n'
              '摘要：兔子班尼在意外拿走了一把属于小女孩的梳子，并后来遇到一只死鸟后，明白了诚实和尊重生命的重要性。',
     'instruction': '按照下面输入的约束生成故事',
     'output': '从前，有一只叫班尼的兔子。班尼喜欢整天跳跃和玩耍。一天，班尼看到地上有一把梳子。他觉得挺有意思的，决定捡起来。  \n'
               '当班尼在梳理自己的毛发时，他听到一个声音说：“嘿，兔子！那把梳子是我的！”这是一个小女孩，她掉了梳子。班尼因为没有询问就拿走了梳子而感到很不好意思，迅速把梳子还给了她。  \n'
               '小女孩很高兴，谢班尼的诚实。她告诉他，在拿东西之前一定要先问是很重要的。班尼为自己做对了事情而感到自豪，高高兴兴地跳开了。  \n'
               '当他跳来跳去时，班尼看到地上有一只死鸟。他想起了小女孩的话，知道尊重生命是很重要的，即使它们已经不再活着。班尼为那只鸟默哀，继续他的路程，心里感激自己学到的教训。'}


## 小结
1. 基于`TinyStories`的`Instruct`数据进行指令组合层面均衡的采样，获得了15000条原始数据
2. 构造了翻译函数，异步使用吴恩达老师的`translation-agent`对数据进行翻译
3. 基于翻译后的数据，构造了经典格式的`SFT`数据集


