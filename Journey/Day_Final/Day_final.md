这个系列陆陆续续更新了`13`期，总算是在**粗粒度**上把大模型的训练流程走了一遍。

在`64G`内存 + `3090 Ti`单卡配置上完成了以下的主要内容：

1. 预训练数据的预处理：批量并行的`tokenization`
2. `0.044B`参数量的`Tinystories(Chinese)`故事模型预训练
3. 基于大模型（吴恩达`translation-agent`）的`SFT`数据批量翻译
4. `SFT`训练（包含全参数和`LoRA`两种方式）
5. `DPO`数据生成
6. `DPO`训练

过程中也不乏许多对**模型实现细节**和**训练框架源码**的深入阅读和理解，以及一些**算法原理**的学习/复习。

无论效果好坏，细节是否到位，自己还是觉得收获颇丰的，这一期打算做个大汇总，也算是给这段学习一个完整的交代，同时方便有需要的小伙伴查阅。

> 整个过程里的尝试和经验不一定具备广泛的普适性，个人水平也十分有限，欢迎大家批评指正。

当然，这次总结并不代表这部分的学习就彻底结束了，一些更多的**尝试和思考**还在继续，后面会陆续进行**补充更新**。




## 「从零手搓中文大模型」传送门

先列一波整个系列的全部相关内容传送门，**想直接看经验总结的可以跳到后面哈**。

### 代码传送门

> https://github.com/puppyapple/Chinese_LLM_From_Scratch

### 视频传送门

> https://space.bilibili.com/341251360/channel/collectiondetail?sid=3724215

（视频制作不易，所以进度相比公众号文章会慢很多，目前还在努力更新中🤣）

### 公众号合集传送门

[#从零手搓中文大模型](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzkyMzczMjkxMA%3D%3D&action=getalbum&album_id=3599032183991779337&scenenote=https://mp.weixin.qq.com/s?__biz%3DMzkyMzczMjkxMA%3D%3D%26mid%3D2247484081%26idx%3D1%26sn%3Da740d8346704d27dc215950d4ce1d99b%26chksm%3Dc00d71f4e1214c2fe35302fe98ec6f076f29eea2dcd67568a3861863ba6c7e6f4832e89bd955%26scene%3D126%26sessionid%3D1727572824%26subscene%3D91%26clicktime%3D1727572828%26enterid%3D1727572828%26ascene%3D3%26devicetype%3DiOS18.0%26version%3D18003424%26nettype%3D3G+%26abtest_cookie%3DAAACAA%253D%253D%26lang%3Dzh_CN%26countrycode%3DCN%26fontScale%3D100%26exportkey%3Dn_ChQIAhIQW8oJys10cUzrj5zuSU%252BgVxLZAQIE97dBBAEAAAAAAHzMJ7V4CqgAAAAOpnltbLcz9gKNyK89dVj0sfQwZoaQr6vXojy9g0gYCI6hqbyXfvmfGWyzXj89VcxBNbuR8UWmidc%252BZmUF7swRYb8m%252B2xvF3w5fnuadG8%252BJGgBjcdjxf7HCNRfjL1PgQtGWh2VnTyvx%252FXzrqVXYUBpVTQtVBuyiX1nHVUMHr794yaLscaxaDJW497sreHaivMobvVpeRHMVxsd2nDJ%252FQ1SZhYwoN6ZvQUwB9iQsbEyBN7Idn2CKc7%252F8I2Sx%252BXt7%252FTf8cI%253D%26pass_ticket%3DuYco99quUGGX4QRAQ9HXe0zE2X3NtF%252FtqemWE%252BE9lnPOBmDjyzH5rX2KPimEbPQK%26wx_header%3D3&nolastread=1&devicetype=iMac+MacBookAir10,1+OSX+OSX+14.5+build(23F79)&version=13080810&lang=zh_CN&nettype=WIFI&ascene=0&fontScale=100&uin=&key=)

### 公众号每日传送门

[从零手搓中文大模型｜Day01｜打卡第一天，欢迎大家监督催更](https://mp.weixin.qq.com/s/kbmkdkukkvnGMCzRD2Z1mQ)

[从零手搓中文大模型｜Day02｜Tokenizer & BPE](https://mp.weixin.qq.com/s/LD2LTtEz1bvSdxZkIO1ePg)

[从零手搓中文大模型｜Day03｜数据预处理](https://mp.weixin.qq.com/s/bVueRGPp_JqXh4N74A-OPg)

[从零手搓中文大模型｜Day04｜模型参数配置和训练启动｜我的micro模型跑起来啦！](https://mp.weixin.qq.com/s/ZpaO2cxVrTOlBFw45rIqiQ)

[我的超迷你大模型会讲故事啦｜从零手搓中文大模型｜Day05](https://mp.weixin.qq.com/s/M7RmebRDvfMXomHln6R5PQ)

[从零手搓中文大模型｜Day06｜预训练阶段代码汇总和整理](https://mp.weixin.qq.com/s/D2DCf7iq0A6BRY46X1KN8g)

[吴恩达老师帮我构造指令微调（SFT）数据｜从零手搓中文大模型｜Day07](https://mp.weixin.qq.com/s/NnPNkiYoXAwD21Bb3id31g)

[大模型SFT敲黑板知识点（这次吴恩达老师也帮不了我了）｜从零手搓中文大模型｜Day08](https://mp.weixin.qq.com/s/_me0CJVrxlQ1Y_JaFfEXhg)

[说出来你可能不信：0.044B的大模型也能指令遵从呀｜从零手搓中文大模型｜Day09](https://mp.weixin.qq.com/s/c8FwjnVSr4Il4JYVroWdRQ)

[手搓了好多天搓累了，歇下来聊聊自己对大模型和人工智能的一些拙见吧｜从零手搓中文大模型｜中秋特别篇](https://mp.weixin.qq.com/s/vRVhDCPmUybgy5T2jurPiA)

[只要我的大模型参数量够小，刷数据就再也没有爆显存的烦恼｜从零手搓中文大模型｜Day11](https://mp.weixin.qq.com/s/K9hnUpQ0mAl0400QCbkgpQ)

[万万没想到，我被transformers库里这个不负责任的简化代码给整破防了｜从零手搓中文大模型｜Day12](https://mp.weixin.qq.com/s/GnHxguhtZjh5SiFLU2CGkA)

[哈哈哈哈果不其然，SFT质量不到位的情况下进行DPO，无异于屎上雕花｜从零手搓中文大模型｜Day13](https://mp.weixin.qq.com/s/aGpKf4a0iFrIgPqZ0VAotA)


## 踩坑/经验记录

这里不光记录自己实践过程中学到的**个人认为比较重要的点**，也会记录一些来自留言区小伙伴的**高质量反馈**，包括我还未来得及去尝试实践的建议。

重要性不分先后。

### Pretrain

1. `learning rate`的选择对模型的收敛效率影响真的很大，尤其是**模型参数**和**数据量**都不是完全参考已有的开源实现而是有很多**自定义**的情况下，学习率就不能迷信已有的经验值，需要自己尝试。例如设置多个`learning rate`，然后训练少量`steps`，观察损失的下降趋势，从而选择一个合适的`learning rate`。

2. 无论是垂域还是通用域数据，务必做好数据的**去重**；目前没有确切的研究表明同一条数据被模型学习多少次是合适的，但无论如何，增大两条近似重复数据之间的**距离**总是有益的。

3. 如果和我一样想训练一个**小语言模型（SLM）**，`tokenizer`最好是根据自己的数据集专门训练一个，主要是为了减少不必要的`vocabulary`，从而减少`embedding`参数在整个模型中的占比，能有更多的参数用来学习数据本身。

    > 举个例子，在我的故事数据集里其实`6-7k`的词汇量就基本上覆盖了全部的数据了，采用一些开源的`tokenizer`得到的`vocab size`往往至少也有`30k`以上，这样就会导致`embedding`层占用了过大的参数。

4. 预训练数据集量级大的时候，一定要做数据`tokenization`的预处理，在训练过程中转`token`是对`GPU`的一种「**侮辱**」。

## SFT

1. 无论是全参数微调还是`LoRA`，`SFT`首当其冲的**必要条件**是数据集的质量，否则再怎么优化炼丹术也只可能是**garbage in, garbage out**。

2. 指令**数据的配比**目前看来对模型的指令遵循能力影响很大（尤其是偏通用能力搭建的场景下），有条件的话可以多做一些不同配比的实验。这次我是纯故事生成场景，只有单一能力要求和垂直数据，所以还没有做这方面的实践尝试。

3. `SFT`数据量级较大的时候，的训练**轮数**不宜过多，否则容易导致模型过拟合（当然过拟合总好过欠拟合）；相反，数据量级较小的时候，可以适当增加训练轮数让模型学习指令。

4. 根据`SFT`领域数据和预训练模型的差异，以及对模型能力最终效果（通用还是纯垂直领域）的期待，需要决策是否**使用SFT数据加入continue pretrain**，以及**SFT阶段是否要混入通用指令数据**。

5. 更高级一些的（包括在预训练阶段也是），学习数据的**顺序**对模型效果的影响也很大，甚至会造成**灾难性遗忘**；这个也容易理解，模型学知识和我们人是一样的，先学什么，后学什么，顺序不同，结果也会不同。

## RLHF

这一部分了解的确实不多，只是跑通了`DPO`的流程，这里记录一下来自**留言区小伙伴**的反馈和建议：

1. `offline dpo`无法更新采样，一定是不如`iterative online dpo`的 —— @tomiaoooo。不过后者的实践难度可能会大很多，作为一个个人学习目的主导的实践项目，我可能没有能力去尝试了，后面有理论学习方面的收获的话也尽量分享出来。

2. `DPO`这类相比于`PPO`更简单的`RLHF`算法，其实对数据集质量的要求反而更大了 —— @阿白。

3. 基座模型比较好的话，`KTO（Model Alignment as Prospect Theoretic Optimization）`的效果不错，甚至可以省去`SFT`阶段，直接用`KTO`进行`RLHF`—— @风飘絮。

4. 目前看起来，实际落地的模型测评显示`simpo`表现不错，值得测试 —— @最美的梦给未来的自己。

5. `DPO`之类的算法对`SFT`之后的`Instruct Model`的质量有很大依赖，如果`SFT`效果不佳，`DPO`大概起不到预期的效果，甚至可能适得其反。


## 重要参考资料大汇总
`13 Days`的学习实践工程中，我重点参考过的全部资料，包括**技术博客和开源的Repo**，为我的项目顺利走通提供了极大的帮助，强烈推荐给大家，排名不分先后。

1. [LLM训练-pretrain](https://zhuanlan.zhihu.com/p/718354385) 知乎大佬`ybq`分享的预训练经验帖，非常详细，强烈推荐当作**八股文**学习背诵。

2. [LLM-SFT-trick](https://zhuanlan.zhihu.com/p/682604566) 详细介绍了作者大模型指令微调的实践经验，干货非常多。

3. [minimind](https://github.com/jingyaogong/minimind) 一个大佬开源的和我这个类似的「**从零**」构建大模型项目，**中文开放域** + **小参数量级**，从零实现了对话大模型。

4. [Lightning-AI/litgpt](https://github.com/Lightning-AI/litgpt/tree/main) 一个轻量级的大模型训练框架，没有复杂的抽象很封装，非常适合`DIY`。

5. [jzhang38/TinyLlama](https://github.com/jzhang38/TinyLlama) 基于上面的`litgpt`框架实现的`TinyLlama`，在3万亿`token`数据上训练的`1.1B`参数量模型。

6. [zh-babyllama2chinese](https://github.com/DLLXW/baby-llama2-chinese) 一个小参数量（`500M-1B`）的**中文**`Llama2`仓库

## 后续计划

我个人还是对大模型的**应用**更感兴趣一些，因此后面除了逐渐完成此前一些还没有做过的尝试，我下一步的主要计划是「**从零构建**」一个基于**语言模型（Large or small都可能）**的**应用**，学习实践更多应用层的工程和理论知识。

至于具体选择什么样的应用，目前还在绞尽脑汁思索中，如果大家有什么建议也欢迎留言交流哈。

