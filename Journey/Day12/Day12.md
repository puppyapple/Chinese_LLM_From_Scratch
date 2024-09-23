之前说准备好`DPO`数据就着手开始自己`DIY`一下训练的实现了（因为`litgpt`库暂时还没有集成相关实现）。
虽说是`DIY`但是纯从零手撸的成本还是有点高的，因此还是打算参考已有的实现来弄，比如：
- [eric-mitchell/direct-preference-optimization: Reference implementation for DPO (Direct Preference Optimization)](https://github.com/eric-mitchell/direct-preference-optimization)
- [huggingface的trl库里的实现](https://github.com/huggingface/trl/blob/main/trl/trainer/dpo_trainer.py)

`transformers`库毕竟是大主流，因此还是优先考虑基于`transformers`库的`API`来实现。

那么面临的第一个问题就是需要将`litgpt`框架训练的模型转换为`huggingface`库的模型格式。

而在这个过程中我遇到了一个**意想不到的坑**，直接给我整**破防**了。

大家听我慢慢道来。



## litgpt模型转换到huggingface format
`litgpt`框架本身考虑得很周全，它从简化模型实现的角度完全基于`torch`实现的[model.py](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py)

同时也提供了将`checkpoint`转换为`huggingface`模型格式的`API`：


```python
# ! litgpt convert_from_litgpt input_checkpoint_dir output_dir
```

这里会得到一个`model.pth`文件，为了后续加载方便，大家可以直接改名为`pytorch_model.bin`。

另外需要注意的是`transformers`库依赖`config.json`文件，如果大家的模型架构选择的是使用`litgpt`框架已经支持的`huggingface`上的模型，那么可以直接去下载；但如果是自己定义的模型架构，那么就需要大家自己动手来写这个`config.json`文件了。

> 后面我会写一个简单的脚本，基于`litgpt`框架的`config.yaml`文件来生成`config.json`文件，可能在一些场景下能具备通用性。


## transformers库模型加载
 
模型的转换还是比较简单且顺利的，但是按照[litgpt的convert文档](https://github.com/Lightning-AI/litgpt/blob/main/tutorials/convert_lit_models.md)，通过`transformers`库加载转换后的模型文件的时候，问题来了。

我的到了一个意想不到的报错，追溯到的是`transformers`库里对模型参数尺寸检查的这段代码：

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/image.png)

**它强制要求了`embedding`的向量尺寸必须正好等于注意力头数`n_heads`和注意力头尺寸`head_size`的乘积**。

原因是他们不知出于什么考虑，在对`attention`的`kqv`转换矩阵实现的时候，做了一个简化，大家看下图里我用红框标出的地方。

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/imag_4.png)


如果不理解的话，再对比看下面这张图应当就能理解了，是`Karpathy`大神的`nanoGPT`里同样模块的实现。

![](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/image_3.png)

但凡看过原始论文里的公式推导就会知道，`hidden_size = n_heads * head_size`这个要求是**完全没有必要**的。

`W_k/W_q/W_v`矩阵理论上可以将`hidden_size`投影到任何维度上去，只要最后再通过`W_o`的线性层映射回`hidden_size`即可。

而`transformers`库的实现里将这个任何维度简化为了`n_heads * head_size`，这对我这种需要从自定义的模型架构转换过来的场景造成了**毁灭性打击**🤦‍♂️。

因为我当时**拍脑袋**定的`0.044b`参数模型里，`embedding`维度选的`768`，`n_heads`是`6`，而`head_size`定了个`48`，导致这里没有办法加载了。

我当然可以通过改源码的方式来弥补这个问题，但是如果我希望我的模型能够被更多使用`transformers`库的人使用，这个方式就不合适了。

最保险的方式是按照它这个**不合理的**要求来调整我的模型架构，从而得到一个满足`hidden_size = n_heads * head_size`的模型。

## 教训总结


没办法，我最终选择了重新预训练我的故事模型；坑虽然踩了，但是也得到了一些收获。

大部分人一般情况下主要是基于`huggingface`上已有的模型架构来训练，我这类**自定义模型架构**的情况相对少见，因此踩坑踩得有点狠。

即便是`transformers`库这样的大主流，也难免会有一些设计不合理的地方。

也提醒我，对开源库的使用，如果时间精力允许，还是要**多花点功夫看看源码**，理解其背后的设计思想，这样在遇到问题的时候才能从更本质的地方找到解决方案。


> 一个更让我欲哭无泪的事实：
> 
> 就在我的新架构模型快完成预训练的时候，
> 我发现这个不合理的逻辑其实在[这个PR](https://github.com/huggingface/transformers/pull/32857)里得到了修复。
> 
> 在`transformers`库的最新版本里已经没有这个问题了。
> 
> 合着是我更新库不够积极呗？🤷‍♂️


