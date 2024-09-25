<style>
pre {
  overflow-y: auto;
  max-height: 600px;
}
</style>

之前学习的时候就有所耳闻，即便是比`RLHF`简化了很多的`DPO`，想要训练好也不是那么容易的。

实际试了一下，当`SFT`模型质量不高时，果然是**屎上雕花**，`DPO`之后的效果还不如`SFT`。

尽管如此，毕竟还是走通了流程，还是值得记录一下。

## DPO实现

在经历了上期说到的`transformers`库的**巨坑**之后，我重新进行了一次`pretrain`，并在此基础上进行了`SFT`。

这次终于将`litgpt`得到的模型`checkpoint`转换为`Hugging Face`的模型格式并成功加载了。

然后之前通过并行的方式构建了约`15000`条`DPO`数据，于是开始了`DPO`的训练。

基于`trl`库的`DPOTrainer`，训练脚本的实现非常容易，直接贴在这里了：


```python
import os
import click

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer
from trl import DPOConfig
from datasets import load_dataset
from litgpt.utils import num_parameters


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


def init_model(model_name_or_path, device="cuda:0"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"Total parameters: {num_parameters(model):,}")
    model = model.to(device)
    return model, tokenizer


@click.command()
@click.option("--model_name_or_path", type=str)
def main(model_name_or_path):
    model, tokenizer = init_model(model_name_or_path)
    dpo_config = DPOConfig(
        output_dir="../../Experiments/Output/dpo/microstories_lora_v2",
        per_device_train_batch_size=16,
        remove_unused_columns=False,
        num_train_epochs=2,
        learning_rate=1e-5,
        do_eval=True,
        eval_strategy="steps",
        eval_steps=200,
        save_steps=200,
        logging_steps=10,
    )

    data_files = {
        "train": "../../Data/TinyStoriesInstruct/dpo_data_train.json",
        "eval": "../../Data/TinyStoriesInstruct/dpo_data_eval.json",
    }
    dataset_dpo = load_dataset("json", data_files=data_files)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=dpo_config,
        beta=0.1,
        train_dataset=dataset_dpo["train"],
        eval_dataset=dataset_dpo["eval"],
        tokenizer=tokenizer,
        max_length=512,
        max_prompt_length=512,
    )
    dpo_trainer.train()


if __name__ == "__main__":
    main()
```

基于上面的实现，跑起来非常容易，没有遇到什么问题。

`DPO`这块儿的**炼丹**我确实没什么经验，这里也就不随便发表什么意见了；等后续深入学习之后再来补充。

训练过程中的指标波动比较大，简单贴个图：

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/dpo_train.png)

`DPO`的原理细节这里就不展开了，大家可以自行参考论文和开源的代码实现。

其实不论是`loss type`层面还是各种变体的实现，还有很多可以测试的点，但我时间有限，还没来得及做丰富的测试，后面有机会也一定补上。

训练完之后，我随便取了几个样本跑了一下：

1. 如很多网上分享的经验一样，`DPO`对数据的质量和`SFT`模型的质量要求都很高，我的场景里这两条件都不是非常理想，所以效果不是很好也在预期之内

2. 和`SFT`一样，训练轮次不宜过多，`SFT`只是过拟合，而`DPO`是越往后训练，模型能力反而出现退化，已经开始胡言乱语了。。。（`SFT`模型的指令遵循效果虽然表现一般，但至少生成结果是连贯的）



```python
import torch
import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from litgpt.prompts import Phi2

path = "../../Experiments/Output/dpo/microstories_lora_v2/checkpoint-300"
prompt_style = Phi2()
model_sft = AutoModelForCausalLM.from_pretrained(
    "../../Experiments/Output/sft/microstories_v2/bf16_true_1e-4/saved_by_tf"
)
model_hf = AutoModelForCausalLM.from_pretrained(path)
tokenizer = AutoTokenizer.from_pretrained(path)
model_hf.generation_config.pad_token_id = tokenizer.eos_token_id
model_sft.generation_config.pad_token_id = tokenizer.eos_token_id
pipeline_sft = transformers.pipeline(
    "text-generation",
    model=model_sft,
    torch_dtype=torch.bfloat16,
    device="cuda",
    tokenizer=tokenizer,
)

pipeline_dpo = transformers.pipeline(
    "text-generation",
    model=model_hf,
    torch_dtype=torch.bfloat16,
    device="cuda",
    tokenizer=tokenizer,
)


def generate_text(prompt, pipeline, tokenizer):
    sequences = pipeline(
        prompt,
        do_sample=True,
        top_k=50,
        top_p=0.5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=400,
    )

    return sequences[0]["generated_text"]


sft_data = json.load(
    open("../../Data/TinyStoriesInstruct/sft_data_v2_simple_few.json", "r")
)
```

## 随便取个样本试试看


```python
data_index = 10001
prompt = prompt_style.apply(sft_data[data_index]["instruction"])
print(
    "DPO:\n",
    generate_text(prompt, pipeline_dpo, tokenizer),
)
print("-" * 100)
print(
    "SFT:\n",
    generate_text(prompt, pipeline_sft, tokenizer),
)
print("-" * 100)
print("Ground Truth:\n", sft_data[data_index]["output"])
```

```
DPO:
Instruct: 词汇：挤压，章鱼，兴奋
摘要：莉莉在海滩上发现了一只紫色的章鱼，和它玩，然后把它放回海里。
Output:从前，有个小女孩叫莉莉。她喜欢在海滩的沙子里玩耍。一天，她看到一只紫色的章鱼在海底深处游泳。  
“你好呀，小章鱼！”莉莉兴奋地说。“你想和我玩吗？”  
章鱼没有回答，只是继续游走。莉莉跟着它，惊讶地发现海滩好像很大，到处都是大浪。  
“我不知道自己能不能移动，”莉莉建议。“我太小了，不能和章鱼玩。”  
但章鱼依然沉着，游动的本领可爱。最后，莉莉决定把章鱼放回海里，跟它说再见。当她走开时，章鱼突然再次出现，这次还挤着她！  
莉莉惊讶地看到章鱼躲在一块石头后面，他们在海滩上一起欢笑玩耍。从那天起，莉莉总是期待在海滩上看到紫色章鱼，和新朋友一起玩。
----------------------------------------------------------------------------------------------------
SFT:
Instruct: 词汇：挤压，章鱼，兴奋
摘要：莉莉在海滩上发现了一只紫色的章鱼，和它玩，然后把它放回海里。
Output:从前，有个小女孩叫莉莉。她喜欢在海滩上玩耍和捡贝壳。一天，她在沙子里发现了一只大紫色的章鱼。它有长长的触手，莉莉兴奋地想和它玩。  
莉莉轻轻地捡起章鱼，紧紧握在手里。但随后，她感到有点累，决定在沙子里打个盹。她把章鱼放在身边，闭上了眼睛。  
当莉莉醒来时，章鱼还在那儿。她睁开眼睛，看到章鱼在她的手中。她微笑着说：“谢谢你陪我玩，章鱼！”然后，她把章鱼放回沙子里，让它回家。莉莉向章鱼挥手告别，高高兴兴地回家了，交到了一个新朋友。
----------------------------------------------------------------------------------------------------
Ground Truth:
从前，有一个快乐的小女孩叫莉莉。她喜欢在海滩上玩耍，寻找沙子里的宝藏。一天，她发现了一只大而软绵绵的章鱼！它是紫色的，长着长长的触手。莉莉看到它非常兴奋！  
她轻轻地捡起章鱼，给了它一个拥抱。她喜欢捏它的感觉。章鱼似乎并不介意，甚至用触手缠绕住莉莉的手臂。他们一起玩了一会儿，但莉莉知道是时候让章鱼放回海里了。  
看着它游走时，莉莉感到有点伤心，但也为能遇到这么有趣的生物而感到高兴。她知道那天在海滩上发现章鱼的事会永远记在心里。
```

可以看到上面`DPO`里的故事里已经出现很多语法错误和严重的逻辑错误了。

不过没关系，在小尺寸的模型上，这类尝试本来就是一个探索，跑通流程就已经能学到很多东西了。

不过无论如何，我的**从零手搓中文大模型**之旅到这里也算是阶段性地告一段落了。

## 小结一下

接下来自己大概有这么几个计划：

1. 尝试一下`DPO`的变体以及其他的参数配置，多做一些实验

2. 将自己这段时间**从零手搓**过程中遇到的各种问题、经验、教训等等都整理一下，系统地记录下来
   
3. 时间和条件允许的话，在更通用更大的数据集上（或者相反，一个更垂直的领域上）来做更细致的实现


