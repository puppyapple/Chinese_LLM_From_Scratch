# 从零手搓中文大模型｜🚀 Day09

微软的`Tinystories`论文里，是直接在200w条`Instruction`数据上做的全量`pretrain`来验证小参数`LLM`的指令遵从效果的。

为了挖掘`SLM`的潜力，我想看看在超小规模参数的情况下，少量（相比于`pretrain`）数据的`SFT`是否能起作用。

（当然还有一个原因是要把`Instruction`数据全通过`GPT API`翻译一遍还是相当贵的😂）。

## 全参数SFT训练实验🧪

上一期分享了一些`SFT`训练相关的知识点，里面提到了关于训练模式的选择。

我的这个项目里，用于`SFT`训练的数据和之前预训练的数据分布是非常相似的，所以这里不打算将`SFT`数据用于`continue pretrain`，而是直接将`SFT`数据用于`finetuning`。

由于`SFT`全量`finetuning`其实本质上和`pretrain`没有什么差别，只是在计算`loss`的时候对`prompt`部分做了一个`mask`，所以这里就不对训练参数配置做过多的介绍了。

> 这里额外提一点，我在上构造的数据基础上做了一个增强的操作（用`GPT API`翻译还是太贵了😂）。
> 
> 具体操作是：将上期用吴恩达老师的`translation-agent`翻译构造的数据里的指令部分里的多个约束抽取成了`key: value`，然后随机排列，输出还是故事本身不变，这样就得到了很多新的数据（从之前的1.3w条数据增加到了7.1w条）。
> 
> 另外还有一个潜在的好处就是可以让模型知道指令内部的多个约束的顺序是不敏感的，只要输出符合所有指令的约束就可以。

我简单地做了几组实验：

🟣 `learning_rate = 1e-4, bf16-true`

🔴 `learning_rate = 5e-4, bf16-true`

🟢 `learning_rate = 5e-4, bf16-true`，但学习率下降比前两者速度更快

🔵 `learning_rate = 5e-4, bf16-mixed`，学习率和上一个一样

![image_v2](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/image_v2.png)

> 为了方便观察，图里的曲线都是经过平滑之后的。

可以发现几个问题🤔：

1. 学习率使用`pretrain`的1/5的时候（`1e-4`），收敛程度不如使用和`pretrain`时一样的`5e-4`。

   和上一期里搜集的经验描述有些不一致（`SFT`阶段的`learning_rate`使用`pretrain`的1/10的建议）。
   
   我个人理解是因为我的`SFT`数据和`pretrain`数据非常相似，且指令相对简单/单一（只是在故事前面加了一些约束文本），所以即使用比较大的学习率也没有出现震荡发散的情况，反而很容易收敛。

2. 学习率被设置得下降更快的这一组，收敛速度也更快一些，这个也容易理解：在后期，模型已经非常接近最优解了，这时候学习率下降得快，可以更精细地学习以逼近最优解。
3. 使用`bf16-mixed`的这一组，收敛速度和前一个差不多，但是loss整体还要更低一些

### 结果测试
#### 单一约束

我随便构造了几个简单的测试用例，其中的指令都只包含单一的约束。

结果如下👇：


```python
from litgpt import LLM
from litgpt.prompts import MicroStories

ms = MicroStories()
llm = LLM.load(model="../../Experiments/Output/sft/microstories/mask_prompt_5e-4/final")
test_cases = [
    {
        "instruction": "请用给定的约束生成故事",
        "input": "词汇：铲子，草地\n",
    },
    {
        "instruction": "请用给定的约束生成故事",
        "input": "特点：转折\n",
    },
    {
        "instruction": "请用给定的约束生成故事",
        "input": "摘要：一只小蚂蚁在花园里寻找吃的，最后找到了一个苹果。\n",
    },
    {
        "instruction": "请用给定的约束生成故事",
        "input": "随机句子：天空中飘着几朵白云。\n",
    },
]
prompts = [
    ms.apply(prompt=case["instruction"], input=case["input"]) for case in test_cases
]
for prompt in prompts:
    text = llm.generate(
        prompt=prompt,
        max_new_tokens=300,
        temperature=0.8,
        top_p=0.9,
        top_k=30,
    )
    print(text)
    print("-" * 100)
```
```text
从前，有一个大草地。它大得让人惊叹，因为有一群喜欢玩沙子。孩子们会围着草滚来滚去，用铲子在草地上挖沙。
一天，一个叫莉莉的小女孩决定去玩那个大草地。她拿起铲子开始挖沙。她挖呀挖，直到她发现了一块大石头。她把石头给她的朋友们看，并问能不能用它来挖。
她的朋友们也挖沙子，但他们挖不出来。然后，他们又开始挖那块石头。突然，莉莉听到了一声巨响。她把沙子挖出来，看到她的朋友们在玩一个玩具。她想起了自己的铲子，微笑着。她知道可以用它来挖沙。
从那以后，莉莉就爱上了她的大草地和她的铲子。她总是用那块大铲子，确保她的玩具保持安全。每次找到新玩意儿，她都会把它拿出来，用它来挖沙。
----------------------------------------------------------------------------------------------------
从前，有一个小男孩。小男孩，他喜欢玩玩具。他喜欢玩玩具，也喜欢和妈妈和爸爸一起玩。  
一天，他看到一只狗。那只狗非常活泼，也想玩耍。他觉得和狗一起玩一定很有趣。  
那只狗的爪子一碰到小水，竟然开始在地上倒水！小男孩非常开心，他捡起水，开始在水面上摇晃。他玩得非常开心，没多久就该把水倒回水里了。小男孩把水倒在地上，狗也跟着他跑。他们玩得特别开心，小男孩说：“谢谢你，小狗，帮了我！”  
小狗用摇晃的水瓶摇晃着，小男孩玩得不亦乐乎。他们玩了很长时间，回家的时候，小男孩和狗都很累。他们都带着灿烂的笑容，小男孩带着他的新朋友小狗，开心地回家。
----------------------------------------------------------------------------------------------------
从前，有一只小蚂蚁。蚂蚁在花园里玩耍，发现了一颗好吃的苹果。他吃了苹果，开始感到快乐。
但随后，蚂蚁看到花园里有一只大狗。狗看到小蚂蚁后，开始追它。
小蚂蚁感到很害怕。它不知道该怎么办。然后，小蚂蚁听到了他的声音。
他找到了一些种子，吃了。他感到安全。
蚂蚁很高兴，感到很满足。他不再害怕狗，继续在花园里享受吃草。

故事结束了。
----------------------------------------------------------------------------------------------------
从前，有一个小男孩，他感到非常伤心。他想要一些东西来让他的感到快乐，但不知道该怎么做。他看到云朵，他试着想一些特别的东西，但不知道是什么。他环顾四周，看到天空中的星星，感到很快乐。他知道自己想要一些特别的东西。

他决定试着让那闪烁的星星消失。他拿起他最喜欢的玩具，一个拼图块。他开始拼这个拼图，拼了很久。他拼了很久，直到拼图拼得整幅画。

天空中飘着白云。它们又黑又亮。他感到很开心。他不再感到伤心了，因为他知道闪烁的星星是特别的。

小男孩微笑着，心里充满了快乐。然后他知道自己做了一件特别的事情。
----------------------------------------------------------------------------------------------------
```

可以看到对简单的约束的支持意外地还是不错的：

**关键词**能完全命中，**转折**虽然很**生硬**，但是看得出来理解了要加入转折。

根据**摘要**生成也比较准确，**随机句子**方面没有办法完全包含原句，但是大差不差（感觉完全包含还是有点难为这个尺寸的模型了）。

#### 组合约束
再来看看组合约束的效果👇：


```python
test_cases = [
    {
        "instruction": "请用给定的约束生成故事",
        "input": "词汇：铲子，草地\n特点：转折\n",
    },
    {
        "instruction": "请用给定的约束生成故事",
        "input": "词汇：铲子，草地\n摘要：一只小蚂蚁在花园里寻找吃的，最后找到了一个苹果。\n",
    },
    {
        "instruction": "请用给定的约束生成故事",
        "input": "词汇：铲子，草地\n随机句子：天空中飘着几朵白云。\n摘要：一只小蚂蚁在花园里寻找吃的，最后找到了一个苹果。\n",
    },
]
prompts = [
    ms.apply(prompt=case["instruction"], input=case["input"]) for case in test_cases
]
for prompt in prompts:
    text = llm.generate(
        prompt=prompt,
        max_new_tokens=300,
        temperature=0.8,
        top_p=1,
        top_k=30,
    )
    print(text)
    print("-" * 100)
```
```text
一天，一个名叫蒂姆的男孩去了公园。他看到一个标志。标志上写着：“今天今天我们要去外面用铲子玩。”蒂姆非常高兴，因为他喜欢铲子。他觉得挖个大洞会很有趣。  
蒂姆开始用铲子挖土。他挖呀挖，挖出了一个大洞。沙子让他感到很累，但他在地下埋了虫子和石头。他非常喜欢这个公园，想要找到更多可以玩的东西。  
然后，突然发生了意外的事情。一个大盒子从公园的盖子开了它开进了公园。蒂姆感到非常惊讶。他打开盒子，发现里面有一只小虫子。虫子没有死，它只是想和蒂姆一起玩。他们在公园里一起玩得非常开心。
----------------------------------------------------------------------------------------------------
从前，有一只小蚂蚁住在一个大花园里。这个花园很大，有很多草和树。小蚂蚁喜欢到处跑和探索。  
一天，小蚂蚁想要找到一些美味的食物。于是，小蚂蚁开始寻找吃的。小蚂蚁在土里挖呀挖，直到找到了一些苦涩的苹果。苹果太美味了！  
小蚂蚁高兴得跑回去给妈妈看。妈妈也喜欢这些苹果。他们坐在花园里，一起吃苹果。小蚂蚁为自己找到了吃的而感到无比自豪。
----------------------------------------------------------------------------------------------------
从前，有一只小蚂蚁住在一个大花园里。这个花园里有许多蚂蚁在花园里努力工作，挖土。一天，这只小蚂蚁饿得不行。  
突然，小蚂蚁看到一朵朵下有一片闪亮的大白云。小蚂蚁心里想：“那是什么东西？”小蚂蚁继续行进，发现那是一棵苹果树。小蚂蚁吃了苹果，感到很开心。  
过了一会儿，小蚂蚁回到了小花园。小蚂蚁为找到这么好的一个家感到自豪。小蚂蚁四处游荡，给所有蚂蚁朋友看了这棵树。他们都微笑着，玩得特别开心。完。
----------------------------------------------------------------------------------------------------
```


混合约束的难度明显上升了，虽然看得出模型在努力地理解指令，但是结果并不理想。

一方面可能我的`base`模型训练得可能还不够充分，另一方面`SFT`数据量少了。

对于`SFT`之后模型生成的连贯性和逻辑性出现明显下降的问题，简单地检索了一下，一个可能的优化方法是在`SFT`数据里加入一些`pretrain`里的数据，这种做法称为`Replay`。

时间有限还没来得及尝试，等后面有时间了可以试试，在之后的更新里同步分享结果给大家吧。

## LORA微调⌛

我也尝试了在`SFT`数据上用`LORA`微调，发现效果并不好，loss下降得很慢，且远高于`SFT`全量微调的loss。

![image4](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/image4.png)

如上图，黄色🟡的是全量微调的loss，红色🔴的是`LORA`微调的loss，这里虽然只有两条，但实际上我尝试了不少`learning rate`和其他参数的组合，但结果都差不多。

我猜测是因为模型太小了，用`Lora`微调时候使用较小的`r`和`alpha`，可训练参数量就更小，所以效果不好。

于是我试了下将`Lora`的`r`和`alpha`调大（🟠从`8_16`调到`256_512`），发现效果好了不少，loss下降得更快了，但收敛速度还是要**远远慢于**全量微调。

这时候的可训练参数量级已经接近`22M`，正好是模型自身的一半了，效果变好也是理所当然的，但这样显然已经失去了`LORA`微调的意义。

> 关于`Lora`的正常使用，后面等有机会训练一个更大的`base model`的时候再尝试吧。

## 小结
1. 分享了`SFT`全量微调的一些实验结果
2. 测试了一下`SFT`全量微调之后的指令遵循效果
3. 分享了用`LORA`微调的一些实验结果