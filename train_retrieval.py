import os
from typing import Optional
from dataclasses import dataclass, field

import torch
from torch.optim import AdamW
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    TrainingArguments,
    BitsAndBytesConfig,
)

from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from sklearn.metrics import accuracy_score, log_loss

from retrieval.Embedding_model import BiEncoderModel
from retrieval.Embedding_trainer import RetrievalTrainer as Trainer
from utils import print_rank_0, get_optimizer_grouped_parameters
from text_processor.reranker_processor import reranker_processor

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
    template: str = field(
        default="<|im_start|>{QUERY}\n\n{DOC}<|im_end|>", metadata={"help": "Template for the input text."}
    )


@dataclass
class TrainingArguments(TrainingArguments):
    llrd_enable: bool = field(default=False)
    score_lr: float = field(default=None)

    lora_enable: bool = field(default=True)
    lora_r: int = field(default=16)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_bias: str = "none"
    lora_target: str = field(default="all-linear")

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

    pretrain_lora: str = field(default=None)


def compute_metrics(eval_preds: EvalPrediction) -> dict:
    preds = eval_preds.predictions
    labels = eval_preds.label_ids
    probs = torch.from_numpy(preds).float().softmax(-1).numpy()
    loss = log_loss(y_true=labels, y_pred=probs)
    acc = accuracy_score(y_true=labels, y_pred=preds.argmax(-1))
    return {"acc": acc, "log_loss": loss}


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.output_dir = os.path.join(training_args.output_dir, training_args.run_name)

    if training_args.lora_target != "all-linear":
        training_args.lora_target = eval(training_args.lora_target)

    training_args.gradient_checkpointing_kwargs = {"use_reentrant": True}

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

    if training_args.lora_enable:
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=training_args.lora_target,
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )

        if training_args.pretrain_lora:
            print_rank_0(f"Loading pretrain lora weight from {training_args.pretrain_lora}...")
            model = PeftModel.from_pretrained(model, training_args.pretrain_lora, is_trainable=True)
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
    train_dataset = Dataset.from_json(data_args.train_data_path)
    val_dataset = Dataset.from_json(data_args.val_data_path)

    preprocess = reranker_processor(tokenizer, max_length=data_args.max_length, template=data_args.template)
    train_dataset = train_dataset.map(
        preprocess,
        batched=True,
        remove_columns=train_dataset.column_names,
        load_from_cache_file=False,
    )
    val_dataset = val_dataset.map(
        preprocess,
        batched=True,
        remove_columns=val_dataset.column_names,
        load_from_cache_file=False,
    )


    trainer = Trainer(
        args=training_args,
        model=retrievaler,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
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