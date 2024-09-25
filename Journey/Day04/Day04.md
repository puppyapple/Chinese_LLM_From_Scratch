# 从零手搓中文大模型｜🚀Day04

前面已经完成了**数据预处理**，今天我们来研究一下**模型的配置**。

`litgpt`使用的配置文件和`transformers`有点不太一样，它的仓库里提供了一些预训练所用的`yaml`[配置文件样例](https://github.com/Lightning-AI/litgpt/tree/main/config_hub)。这个主要用于需要自定义模型的场景。

另外`litgpt`也内置了一些`huggingface`上的[现成模型](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/config.py)，可以直接拿来使用。

## 训练配置文件
以下是我这次定义的一个配置文件。

内容有点多，但是还是都列举出来了，可以直接跳到后面对一些关键参数的解释。

```yaml
# The name of the model to pretrain. Choose from names in ``litgpt.config``. Mutually exclusive with
# ``model_config``. (type: Optional[str], default: null)
model_name: microstories

# A ``litgpt.Config`` object to define the model architecture. Mutually exclusive with
# ``model_config``. (type: Optional[Config], default: null)
model_config:
  name: microstories
  hf_config: {}
  scale_embeddings: false
  block_size: 512
  padded_vocab_size: 65024
  vocab_size: 64798
  n_layer: 6
  n_head: 6
  n_query_groups: 6
  n_embd: 512
  head_size: 48
  rotary_percentage: 1.0
  parallel_residual: false
  bias: false
  norm_class_name: RMSNorm
  mlp_class_name: LLaMAMLP
  intermediate_size: 768

# Directory in which to save checkpoints and logs. If running in a Lightning Studio Job, look for it in
# /teamspace/jobs/<job-name>/share. (type: <class 'Path'>, default: out/pretrain)
out_dir: Chinese_LLM_From_Scratch/Experiments/Output/pretrain/microstories

# The precision to use for pretraining. Possible choices: "bf16-true", "bf16-mixed", "32-true". (type: Optional[str], default: null)
precision: bf16-mixed

# Optional path to a checkpoint directory to initialize the model from.
# Useful for continued pretraining. Mutually exclusive with ``resume``. (type: Optional[Path], default: null)
initial_checkpoint_dir:

# Path to a checkpoint directory to resume from in case training was interrupted, or ``True`` to resume
# from the latest checkpoint in ``out_dir``. An error will be raised if no checkpoint is found. Passing
# ``'auto'`` will resume from the latest checkpoint but not error if no checkpoint exists.
# (type: Union[bool, Literal["auto"], Path], default: False)
resume: true

# Data-related arguments. If not provided, the default is ``litgpt.data.TinyLlama``.
data:
  # TinyStories
  class_path: litgpt.data.LitData
  init_args:
    data_path: Chinese_LLM_From_Scratch/Data/TinyStoriesChinese/processed_data
    split_names:
      - train
      - val

# Training-related arguments. See ``litgpt.args.TrainArgs`` for details
train:
  # Number of optimizer steps between saving checkpoints (type: Optional[int], default: 1000)
  save_interval: 1000

  # Number of iterations between logging calls (type: int, default: 1)
  log_interval: 1

  # Number of samples between optimizer steps across data-parallel ranks (type: int, default: 512)
  global_batch_size: 512

  # Number of samples per data-parallel rank (type: int, default: 4)
  micro_batch_size: 32

  # Number of iterations with learning rate warmup active (type: int, default: 2000)
  lr_warmup_steps: 1000

  # Number of epochs to train on (type: Optional[int], default: null)
  epochs:

  # Total number of tokens to train on (type: Optional[int], default: 3000000000000)
  max_tokens: 3000000000000

  # Limits the number of optimizer steps to run. (type: Optional[int], default: null)
  max_steps:

  # Limits the length of samples. Off by default (type: Optional[int], default: null)
  max_seq_length: 512

  # Whether to tie the embedding weights with the language modeling head weights. (type: Optional[bool], default: False)
  tie_embeddings: true

  #   (type: Optional[float], default: 1.0)
  max_norm: 1.0

  #   (type: float, default: 4e-05)
  min_lr: 0.0

# Evaluation-related arguments. See ``litgpt.args.EvalArgs`` for details
eval:
  # Number of optimizer steps between evaluation calls (type: int, default: 1000)
  interval: 2000

  # Number of tokens to generate (type: Optional[int], default: null)
  max_new_tokens:

  # Number of iterations (type: int, default: 100)
  max_iters: 100

  # Whether to evaluate on the validation set at the beginning of the training
  initial_validation: false

  # Whether to evaluate on the validation set at the end the training
  final_validation: false

# Optimizer-related arguments
optimizer:
  class_path: torch.optim.AdamW

  init_args:
    #   (type: float, default: 0.001)
    lr: 0.0005

    #   (type: float, default: 0.01)
    weight_decay: 0.1

    #   (type: tuple, default: (0.9,0.999))
    betas:
      - 0.9
      - 0.95

# How many devices/GPUs to use. Uses all GPUs by default. (type: Union[int, str], default: auto)
devices: auto

# How many nodes to use. (type: int, default: 1)
num_nodes: 1

# Optional path to the tokenizer dir that was used for preprocessing the dataset. Only some data
# module require this. (type: Optional[Path], default: null)
tokenizer_dir: Chinese_LLM_From_Scratch/References/chatglm3-6b

# The name of the logger to send metrics to. (type: Literal['wandb', 'tensorboard', 'csv'], default: tensorboard)
logger_name: wandb

# The random seed to use for reproducibility. (type: int, default: 42)
seed: 42
```

### model_config

```yaml
model_config:
  name: microstories
  hf_config: {}
  scale_embeddings: false
  block_size: 512
  padded_vocab_size: 65024
  vocab_size: 64798
  n_layer: 6
  n_head: 6
  n_query_groups: 6
  n_embd: 512
  head_size: 48
  rotary_percentage: 1.0
  parallel_residual: false
  bias: false
  norm_class_name: RMSNorm
  mlp_class_name: LLaMAMLP
  intermediate_size: 768
```

- `scale_embeddings`控制是否对embedding进行缩放。
  
![scale_embedding](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/scale_embedding.png)
  
  如果为`True`，那么在`forward`函数中会对`embedding`进行缩放。注意个缩放和`sefl-attention`中的缩放不是一回事，不要弄混了。
  其实也有很多讨论关于这个地方这一步**是否有必要**的，目前看来似乎是区别不大，可以设置为`False`。
- `transformer`中的`block_size`，也就是`max_seq_length`。
- `padded_vovab_size`和`vocab_size`直接取自`tokenizer`。
- `n_layer`和`n_head`都是`6`，构建了一个`6`层`6`头的`transformer`。
- `n_query_groups`是`6`，这是`GQA(Grouped-Query Attention)`的一个参数，控制`query`的分组。当`n_query_groups`等于`n_head`时，其实就是`MHA(Multi-Head Attention)`。下面这个图比较直观：
  
![GQA](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/GQA_2.png)

- 头的大小`head_size`是`48`，`n_embd`是`512`。
- `rotary_percentage`是`1.0`，这个是`旋转编码（Rotary Position Embedding, RoPE）`的有关参数，这里先不展开介绍了。
- `parallel_residual`是`false`，关于`parallel residual`和`non-parallel residual`的解释可以参考这个图：
  
![parallel_residual](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/parallel_residual.png)
- `bias`控制`Linear`层的`bias`是否存在，现在大多模型一般都是`false`。
- `norm_class_name`是`RMSNorm`，`mlp_class_name`是`LLaMAMLP`，具体可以参见`litgpt`里[`model.py`](https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py#L30)中的实现。
- `intermediate_size`是`768`，这个是上面的`MLP`中间层的大小。

按照上面的配置得到的模型参数量在`44M`左右，也就是只有`0.044B`的大小。

但根据微软的[TinyStories](https://arxiv.org/pdf/2305.07759)论文结论，`10-80M`级别的模型能在小故事生成这种简单的语言任务上达到不错的效果（依旧能说人话）。

### 其他参数

其余的都是一些训练的参数，比如`batch_size`，`lr`，`weight_decay`等等，这些都是比较常见的参数，不再赘述。

`logger`我这里选择的是`wandb`，可以直接在`wandb`上查看训练过程中的一些指标。

`data`设置成之前预处理好的数据集的路径（其中指定了加载数据所用的`litdata`的类名）

`tokenizer_dir`是选用的或者自己训练好的`tokenizer`的路径。

## 启动训练

```bash
litgpt pretrain --config Experiments/configs/microstories.yaml
```
预训练启动的命令非常简单，只需要指定上面的配置文件的路径即可。

不出意外地话模型就能开始训练了，可以在`wandb`上查看训练过程中的指标。

我的模型其实已经训练了一段时间，show一下训练过程中的图表：

![pretrain_wandb](https://erxuanyi-1257355350.cos.ap-beijing.myqcloud.com/pretrain_wandb.png)

## 小结
1. 详细介绍了`litgpt`的预训练模型配置文件。
2. 顺带解释了一些重要参数的原理。
3. 训练启动。


