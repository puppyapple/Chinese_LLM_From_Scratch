import os
import click

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
from peft import get_peft_model, LoraConfig, TaskType
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
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
        # local_files_only=True,
        # state_dict=torch.load(f"{model_name_or_path}/pytorch_model.bin"),
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
