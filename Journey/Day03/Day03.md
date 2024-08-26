# 从零手搓中文大模型｜🚀Day03
![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/image.png)
## 数据预处理

虽然省略了数据清洗的逻辑，但是我们还是需要对数据进行预处理，以便于后续的模型训练。

包括以下两个细节：

1. 在每个文本后添加`eos`标记，以便于模型识别句子的结束。
2. 将文本转换为`数字序列`，以便于模型处理。
   
   这一步其实也可以放到模型训练的时候进行，但提前处理可以减少训练时的计算量。

### 数据集划分

解压数据集，得到`48`个jsonl文件，共计`3952863`行json数据。

我之前已经解压过了，并且将原始数据和处理过后的数据分别存在了不同路径下。

这里把命令贴出来以供参考。


```python
# !mkdir -p ../../Data/TinyStoriesChinese/raw_data/train
# !mkdir -p ../../Data/TinyStoriesChinese/raw_data/val
# !mkdir -p ../../Data/TinyStoriesChinese/processed_data/train
# !mkdir -p ../../Data/TinyStoriesChinese/processed_data/val

# !tar zxvf ../../Data/TinyStoriesChinese/TinyStories_all_data_zh.tar.gz -C ../../Data/TinyStoriesChinese/raw_data/train
```

我把最后一个文件`data47_zh.jsonl`（共计78538行）里切分出来4w行作为`eval`数据。


```python
# !mv ../../Data/TinyStoriesChinese/raw_data/train/data47_zh.jsonl ../../Data/TinyStoriesChinese/raw_data/val/
# !head -n 40000 ../../Data/TinyStoriesChinese/raw_data/val/data47_zh.jsonl > ../../Data/TinyStoriesChinese/raw_data/val/val.jsonl
# !tail -n +40000 ../../Data/TinyStoriesChinese/raw_data/val/data47_zh.jsonl > ../../Data/TinyStoriesChinese/raw_data/train/data47_zh.jsonl
# !rm ../../Data/TinyStoriesChinese/raw_data/val/data47_zh.jsonl
```

### 先看一条数据
（都打印出来太长了，所以只输出前100个字符）


```python
import json

with open("../../Data/TinyStoriesChinese/raw_data/train/data00_zh.jsonl", "r") as f:
    for line in f.readlines():
        js = json.loads(line)
        print(js["story_zh"][:100])
        break
```
```
莉莉和本是朋友。他们喜欢在公园里玩。有一天，他们在一棵大树下看到了一个秋千。莉莉想试试那个秋千。她跑到树下，爬上了秋千。
"推我，本！"她说。本轻轻地推了她一下。莉莉感到很开心。她越荡越高，笑着喊叫。
```


### 适配框架API

由于选择了使用[⚡️litgpt](https://github.com/Lightning-AI/litgpt/tree/main)框架进行训练，所以需要引入框架相关的`Class`和`API`来封装我们的数据准备逻辑。

这里我们可以参考[源码里集成的Tinyllama的数据预处理代码](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/data/prepare_slimpajama.py)里的代码，稍作修改。

主要是需要将**Day02**里的`line`处理逻辑封装到`ligtgpt`的`API`中。

但在此之前我们先熟悉一下`litgpt`的Tokenizer的使用方法：

先安装一下`litgpt`以及它所以赖的`litdata`:


```python
# !pip install litgpt
# !pip install litdata
```


```python
import torch
from litgpt import Tokenizer

litgpt_tokenizer = Tokenizer("../../References/chatglm3-6b")
```

这里也实验了一下结果，对比发现和咱们之前**Day02**里用原生`Tokenizer`处理的**结果一致**。

结果这里就不贴出来了，有兴趣的可以自己试一下。

> ⚠️不过需要注意`litgpt`的`Tokenizer.encode`返回的是一个`torch`的`Tensor`


```python
import numpy as np

litgpt_encoded = litgpt_tokenizer.encode(
    json.loads(line)["story_zh"][:100], eos=True
)  # 记得设置eos=True
print(litgpt_encoded)
# print(np.array(litgpt_encoded, dtype=np.uint16))
print(litgpt_tokenizer.decode(litgpt_encoded))
```
```
tensor([30910, 56623, 56623, 54542, 50154, 31761, 31155, 31633, 31815, 54534,
        32693, 54662, 55409, 31155, 35632, 31123, 31633, 34383, 57427, 47658,
        54578, 34518, 31623, 55567, 55226, 31155, 56623, 56623, 54695, 39887,
        32437, 55567, 55226, 31155, 54790, 41309, 52624, 31123, 56856, 32660,
        55567, 55226, 31155,    13, 30955, 54834, 54546, 31123, 54613, 31404,
        30955, 36213, 31155, 54613, 36660, 54563, 54834, 43881, 32024, 31155,
        56623, 56623, 32707, 54657, 33436, 31155, 54790, 54937, 56567, 40714,
        31123, 38502, 56653, 55483, 31155,     2], dtype=torch.int32)
莉莉和本是朋友。他们喜欢在公园里玩。有一天，他们在一棵大树下看到了一个秋千。莉莉想试试那个秋千。她跑到树下，爬上了秋千。
"推我，本！"她说。本轻轻地推了她一下。莉莉感到很开心。她越荡越高，笑着喊叫。
```


### 数据处理代码
数据处理直接参考了上面给出的[litgpt samples](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/data/prepare_slimpajama.py)，我们需要仿照`prepare_slimpajama.py`实现里面相关函数（之前**Day 02**里实现的函数需要稍加改造一下）。


```python
# Copyright Lightning AI. Licensed under the Apache License 2.0, see LICENSE file.

import json
import os
import time
import numpy as np
from pathlib import Path

from litgpt.tokenizer import Tokenizer
from litgpt.data.prepare_starcoder import DataChunkRecipe
from litdata import TokensLoader
from litgpt.utils import extend_checkpoint_dir


class TinyStoriesZhDataRecipe(DataChunkRecipe):
    is_generator = True

    def __init__(self, tokenizer: Tokenizer, chunk_size: int):
        super().__init__(chunk_size)
        self.tokenizer = tokenizer

    def prepare_structure(self, input_dir):
        files = Path(input_dir).rglob("*.jsonl")
        return [str(file) for file in files]

    def prepare_item(self, filepath):

        with open(filepath, "rb") as f:
            for line in f.readlines():
                js = json.loads(line)
                story = js["story_zh"]
                # 注意这里要添加eos
                # 还记得吗：我们的vocab size在int16范围内，所以可以转换为uint16来节省内存
                # story_ids = np.array(
                #     self.tokenizer.encode(story, eos=True), dtype=np.uint16
                # )
                # 很遗憾，实际使用的时候发现如果按照上面这样写，
                # litdata反序列化数据的时候会错误地得到torch.int64且超界的Tensor，
                # 但直接存torch.Tensor没问题（加上litdata不支持torch.uint16），
                # 所以最后实际使用的时候还是用下面这种写法
                story_ids = self.tokenizer.encode(story, eos=True)
                yield story_ids


def prepare(
    input_dir: Path = Path("../../Data/TinyStoriesChinese/raw_data/train"),
    output_dir: Path = Path("../../Data/TinyStoriesChinese/processed_data/train"),
    tokenizer_path: Path = Path("../../References/chatglm3-6b"),
    chunk_size: int = (2049 * 8012),
    fast_dev_run: bool = False,
) -> None:
    from litdata.processing.data_processor import DataProcessor

    tokenizer_path = extend_checkpoint_dir(tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)
    data_recipe = TinyStoriesZhDataRecipe(tokenizer=tokenizer, chunk_size=chunk_size)
    data_processor = DataProcessor(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        fast_dev_run=fast_dev_run,
        num_workers=os.cpu_count(),
        num_downloaders=1,
        # 这里有个「巨坑」，如果不加这一行，处理好的数据配对的index.json里
        # 有一个名为"dim"的key值会为null，导致后续有一个无法规避的报错
        # 但是官方的例子里是没有这一行的，很奇怪为何会有这个问题
        item_loader=TokensLoader(),
    )

    start_time = time.time()
    data_processor.run(data_recipe)
    elapsed_time = time.time() - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")
```

首先，我这里主要就是把之前实现的`line`处理逻辑封装到`litgpt`的`DataChunkRecipe`中：
- `prepare_structure`函数给定路径返回符合我们期望的数据文件的路径列表
- `prepare_item`函数给定一个上面的数据文件的路径，根据我们**自定义**的`tokenization`处理逻辑返回一个`np.array`对象
  
然后，定义了一个`prepare`函数，指定我们数据的输入路径和输出路径以及一些其它参数配置（其实用默认的即可），其余的都交给了`litdata`的`DataProcessor`，它基于我前面定义的`DataChunkRecipe`来处理数据。

感兴趣的可以看看`DataProcessor`的源码，里面做了很多并行之类的数据处理优化。

#### 先用eval数据集测试


```python
prepare(
    input_dir=Path("../../Data/TinyStoriesChinese/raw_data/val"),
    output_dir=Path("../../Data/TinyStoriesChinese/processed_data/val"),
    tokenizer_path=Path("../../References/chatglm3-6b"),
)
```

（也可以设置`fast_dev_run=True`来处理更少的数据，尤其是debug时十分有用）

执行完可以在`processed_data/eval`目录下看到生成的`.bin`文件以及记录了每个`chunk`文件信息的`index.json`。

比较一下可以发现从原先的`83m`的`.jsonl`文件压缩到了`13m`的`.bin`，压缩比（83/13≈6.385）还是很可观的。

#### 处理train数据集
在32核的CPU上处理`train`数据集耗时不到`1min`。


```python
prepare(
    input_dir=Path("../../Data/TinyStoriesChinese/raw_data/train"),
    output_dir=Path("../../Data/TinyStoriesChinese/processed_data/train"),
    tokenizer_path=Path("../../References/chatglm3-6b"),
)
```

## 小结

1. 数据预处理的逻辑主要是将文本转换为数字序列，以便于模型处理。
2. 通过`litgpt`的`Tokenizer`可以方便的实现文本到数字序列的转换。
3. `litdata`提供了数据处理的`API`，可以方便的封装我们的数据处理逻辑。
4. 基于上面的开发，将`TinyStoriesChinese`数据集做了数据划分并完成了预处理。
