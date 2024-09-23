import os
from typing import Optional
from dataclasses import dataclass, field

import torch
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)

from peft import LoraConfig, TaskType, get_peft_model, PeftModel

from retrieval.Embedding_model import BiEncoderModel
from retrieval.Embedding_data import TrainDatasetForEmbedding, EmbedCollator
from retrieval.Embedding_trainer import RetrievalTrainer as Trainer
from utils import print_rank_0, mapk, recall

os.environ["WANDB_PROJECT"] = "eedi"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-Math-1.5B-Instruct")
    add_eos_token: bool = field(default=False)

    normlized: bool = field(default=False)
    negatives_cross_device: bool = field(default=False)
    temperature: float = field(default=1.0)
    use_inbatch_neg: bool = field(default=True)
    sentence_pooling_method: str = field(default="last")

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
    max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    query_instruction: str = field(default=None, metadata={"help": "Instruction before query."})
    passage_instruction: str = field(default=None, metadata={"help": "Instruction before retrieval passages."})


@dataclass
class TrainingArguments(TrainingArguments):
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
    group_size = len(preds[0])
    mAP = mapk(labels, preds, group_size)
    Recall = recall(preds, labels)

    return {"mAP": mAP, "Recall": Recall}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    if model_args.lora_target != "all-linear":
        model_args.lora_target = eval(model_args.lora_target)

    # training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=True,
        add_eos_token=model_args.add_eos_token,
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

    if bnb_config is not None:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            quantization_config=bnb_config,
        )
    else:
        model = AutoModel.from_pretrained(
            model_args.model_name_or_path,
            torch_dtype=torch.bfloat16,
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
            task_type=TaskType.FEATURE_EXTRACTION,
        )

        if model_args.pretrain_lora:
            print_rank_0(f"Loading pretrain lora weight from {model_args.pretrain_lora}...")
            model = PeftModel.from_pretrained(model, model_args.pretrain_lora, is_trainable=True)
        else:
            model = get_peft_model(model, lora_config)

    retrievaler = BiEncoderModel(
        model=model,
        normlized=model_args.normlized,
        negatives_cross_device=model_args.negatives_cross_device,
        temperature=model_args.temperature,
        use_inbatch_neg=model_args.use_inbatch_neg,
        sentence_pooling_method=model_args.sentence_pooling_method,
    )

    # prepare data
    train_dataset = TrainDatasetForEmbedding(
        data_args,
        tokenizer=tokenizer,
        mode="train",
    )
    val_dataset = TrainDatasetForEmbedding(
        data_args,
        tokenizer=tokenizer,
        mode="val",
    )


    trainer = Trainer(
        args=training_args,
        model=retrievaler,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=EmbedCollator(tokenizer=tokenizer),
    )


    trainer.train()
    val_result = trainer.evaluate(val_dataset, metric_key_prefix="val")
    print_rank_0(f"Val result: {val_result}")


if __name__ == "__main__":
    train()