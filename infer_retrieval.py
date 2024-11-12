import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import Optional
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from accelerate import Accelerator

from peft import PeftModel
from datasets import Dataset, DatasetDict
from sklearn.metrics.pairwise import cosine_similarity

from retrieval.Embedding_data import plain_processor
from retrieval.Embedding_trainer import sentence_embedding
from text_processor.processor_v1 import misconception_processor


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2-Math-1.5B-Instruct")
    lora_dir: str = field(default=None, metadata={"help": "Path to the lora model."})

@dataclass
class DataArguments:
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
    add_eos_token: bool = field(default=True)
    query_instruction: str = field(default=None, metadata={"help": "Instruction before query."})
    passage_instruction: str = field(default=None, metadata={"help": "Instruction before retrieval passages."})


@dataclass
class TrainingArguments:
    eval_batch_size: int = field(default=8, metadata={"help": "Batch size for evaluation."})
    save_path: str = field(default="reranker_val.json", metadata={"help": "Path to save the reranker validation data."})
    half_precision: bool = field(default=False, metadata={"help": "Whether to use FP16 for inference."})
    top_k_for_retrieval: int = field(default=25, metadata={"help": "Top k for retrieval."})
    sentence_pooling_method: str = field(default="last", metadata={"help": "Sentence pooling method."})


@torch.no_grad()
def inference(
    model: AutoModel,
    tokenizer: AutoTokenizer,
    val_dataset: DatasetDict,
    misconception_mapping: dict,
    val_raw_text: str,
    top_k: int = 25,
    eval_batch_size: int = 8,
    sentence_pooling_method: str = "last",
):
    model.eval()

    data_collator = DataCollatorWithPadding(tokenizer)
    text_dataloader = DataLoader(
        val_dataset["text"],
        batch_size=eval_batch_size,
        pin_memory=True,
        collate_fn=data_collator,
    )
    text_dataloader = accelerator.prepare(text_dataloader)

    text_embeddings, labels = [], []
    for _, inputs in enumerate(tqdm(text_dataloader, desc="Encoding text: ")):
        target = inputs.pop("labels")
        outputs = model(**inputs).last_hidden_state
        sentence_embeddings = sentence_embedding(outputs, inputs['attention_mask'], sentence_pooling_method)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        results = accelerator.gather_for_metrics(sentence_embeddings.contiguous())
        target = accelerator.gather_for_metrics(target.contiguous())
        text_embeddings.extend(results.tolist())
        labels.extend(target.tolist())

    text_embeddings = np.stack(text_embeddings, axis=0)

    mis_dataloader = DataLoader(
        val_dataset["misconception"],
        batch_size=eval_batch_size,
        pin_memory=True,
        collate_fn=data_collator,
    )
    mis_dataloader = accelerator.prepare(mis_dataloader)

    misconception_embeddings = []
    for _, inputs in enumerate(tqdm(mis_dataloader, desc="Encoding misconception: ")):
        outputs = model(**inputs).last_hidden_state
        sentence_embeddings = sentence_embedding(outputs, inputs['attention_mask'], sentence_pooling_method)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        results = accelerator.gather_for_metrics(sentence_embeddings.contiguous())
        misconception_embeddings.extend(results.tolist())
    
    misconception_embeddings = np.stack(misconception_embeddings, axis=0)

    cos_sim_arr = cosine_similarity(text_embeddings, misconception_embeddings)
    sorted_indices = np.argsort(-cos_sim_arr, axis=1)
    preds = sorted_indices[:, :top_k].tolist()

    raw_text = Dataset.from_json(val_raw_text)["query"]

    reranker_val = []
    for query, label, pred in zip(raw_text, labels, preds):
        reranker_val.append({
            "query": query,
            "label": misconception_mapping[label],
            "candidates": [misconception_mapping[p] for p in pred]
        })
    
    return reranker_val

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        padding_side="right",
        use_fast=False,
        # add_eos_token=model_args.add_eos_token,
    )

    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=torch.float16 if training_args.half_precision else torch.float32,
        device_map="auto",
    )

    if "llama" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
    
    if "qwen" in model_args.model_name_or_path.lower():
        tokenizer.pad_token = "<|endoftext|>"
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.lora_dir:
        print(f"Loading lora weight from {model_args.lora_dir}...")
        model = PeftModel.from_pretrained(model, model_args.lora_dir)

    # bulid val dataset
    text_dataset = Dataset.from_json(data_args.val_data_path)
    plain_preprocess = plain_processor(
        tokenizer, 
        data_args.max_length,
        data_args.misconception_mapping,
        add_eos_token=data_args.add_eos_token,
        query_instruction=data_args.query_instruction,
    )
    text_dataset = text_dataset.map(plain_preprocess, batched=True, remove_columns=text_dataset.column_names)

    misconception_mapping = Dataset.from_csv(data_args.misconception_mapping)
    mis_preprocess = misconception_processor(
        tokenizer, 
        data_args.max_length,
        add_eos_token=data_args.add_eos_token,
        passage_instruction=data_args.passage_instruction,
    )
    misconception_mapping = misconception_mapping.map(mis_preprocess, batched=True, remove_columns=misconception_mapping.column_names)

    val_dataset = DatasetDict({
        "text": text_dataset,
        "misconception": misconception_mapping
    })

    id_to_misconeption = pd.read_csv(data_args.misconception_mapping)
    id_to_misconeption = {
        row["MisconceptionId"]: row["MisconceptionName"]
        for _, row in id_to_misconeption.iterrows()
    }

    reranker_val = inference(
        model,
        tokenizer,
        val_dataset,
        id_to_misconeption,
        data_args.val_data_path,
        top_k=training_args.top_k_for_retrieval,
        eval_batch_size=training_args.eval_batch_size,
        sentence_pooling_method=training_args.sentence_pooling_method,
    )

    with open(training_args.save_path, "w") as f:
        json.dump(reranker_val, f, indent=4)


if __name__ == "__main__":
    accelerator = Accelerator()
    main()