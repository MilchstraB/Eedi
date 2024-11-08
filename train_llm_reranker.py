import os
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from llm_reranker.model import BiEncoderModel
from llm_reranker.data import TrainDatasetForRerank, RerankCollator, ValDatasetForRerank
from llm_reranker.trainer import RerankTrainer as Trainer
from utils import print_rank_0, get_optimizer_grouped_parameters, mapk, recall

os.environ["WANDB_PROJECT"] = "eedi"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-Math-1.5B-Instruct")
    
    load_in_4bit: bool = field(default=False)
    load_in_8bit: bool = field(default=False)

    lora_enable: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = "none"
    lora_target: str = field(default="all-linear")

    pretrain_lora: str = field(default=None)

@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/split/train.csv", metadata={"help": "Path to the training data."}
    )
    val_data_path: str = field(
        default="data/split/val.csv", metadata={"help": "Path to the validation data."}
    )
    misconception_mapping: str = field(
        default="data/misconception_mapping.csv", metadata={"help": "Path to the misconception mapping."}
    )
    max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    template: str = field(
        default="<|im_start|>A: {query}\nB: {passage}\n\n", metadata={"help": "Template for the input text."}
    )
    prompt: str = field(
        default=(
            "Given a query A and a passage B, determine whether the passage contains an answer"
            "to the query by providing a prediction of either 'Yes' or 'No'."
        ), 
        metadata={"help": "Prompt for the input text."}
    )
    add_eos_token: bool = field(default=False)


@dataclass
class TrainingArguments(TrainingArguments):
    train_group_size: int = field(
        default=6,
        metadata={"help": "The total numbers of postive samples and negative samples."}
    )

    llrd_enable: bool = field(default=False)
    score_lr: float = field(default=None)

    gradient_checkpointing: bool = field(default=True)
    eval_steps: float = field(default=0.2)
    eval_strategy: str = field(default="steps")
    bf16_full_eval: bool = field(default=False)
    output_dir: str = field(default="output")
    group_by_length: bool = field(default=False)

    label_smoothing_factor: float = field(default=0.0)
    warmup_ratio: float = field(default=0.05)
    logging_steps: float = field(default=0.005)
    report_to: str = field(default="wandb")


def compute_metrics(preds, labels) -> dict:
    mAP = mapk(labels, preds, 25)
    Recall = recall(preds, labels)

    return {"mAP": mAP, "Recall": Recall}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    if model_args.lora_target != "all-linear":
        model_args.lora_target = eval(model_args.lora_target)

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    # prepare tokenizer and model
    # Since the add_eos_token method sometimes fails, we manually add the eos token.
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
        # add_eos_token=model_args.add_eos_token,
    )

    bnb_config = None
    if model_args.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    elif model_args.load_in_8bit:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        num_labels=1, # https://github.com/FlagOpen/FlagEmbedding/issues/634
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.enable_input_require_grads()
    model.config.use_cache = False

    if "llama" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = "<|endoftext|>"
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.lora_enable:
        lora_config = LoraConfig(
            r=model_args.lora_r,
            lora_alpha=model_args.lora_alpha,
            target_modules=model_args.lora_target,
            lora_dropout=model_args.lora_dropout,
            bias=model_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )

        if model_args.pretrain_lora:
            print_rank_0(f"Loading pretrain lora weight from {training_args.pretrain_lora}...")
            model = PeftModel.from_pretrained(model, training_args.pretrain_lora, is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)

    yes_loc = tokenizer('Yes', add_special_tokens=False)['input_ids'][-1]
    model = BiEncoderModel(
        model=model,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        train_group_size=training_args.train_group_size,
        yes_loc=yes_loc,
    )

    # prepare data
    train_dataset = TrainDatasetForRerank(data_args, tokenizer)
    val_dataset = ValDatasetForRerank(data_args, tokenizer)

    trainer = Trainer(
        args=training_args,
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=RerankCollator(tokenizer=tokenizer),
    )


    if training_args.llrd_enable:
        optimizer_grouped_parameters = get_optimizer_grouped_parameters(
            model,
            base_lr=training_args.learning_rate,
            score_lr=training_args.score_lr,
            weight_decay=training_args.weight_decay,
        )
        optimizer = AdamW(optimizer_grouped_parameters)
        trainer.optimizer = optimizer


    trainer.train()
    val_result = trainer.evaluate(val_dataset, metric_key_prefix="val")
    print_rank_0(f"Val result: {val_result}")


if __name__ == "__main__":
    train()