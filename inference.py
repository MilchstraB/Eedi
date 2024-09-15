from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    HfArgumentParser,
    AutoTokenizer,
    AutoModel,
)

from datasets.dataset_v1 import plain_processor, misconception_processor

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="BAAI/bge-base-en-v1.5")
    model_max_length: int = field(
        default=1024,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    half_precision: bool = field(default=True, metadata={"help": "Whether to use half precision."})


@dataclass
class DataArguments:
    train_data_path: str = field(
        default="data/train_after_process.csv", metadata={"help": "Path to the training data."}
    )
    misconception_mapping: str = field(
        default="data/misconception_mapping.csv", metadata={"help": "Path to the misconception mapping."}
    )


@dataclass
class TrainingArguments:
    output_dir: str = field(default="output", metadata={"help": "The output directory where the model predictions and checkpoints will be written."})
    batch_size: int = field(default=8, metadata={"help": "Batch size per GPU for inference."})


@torch.no_grad()
@torch.cuda.amp.autocast()
def inference(model, dataset, batch_size: int = 8):
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    embeddings = []
    for batch in tqdm(data_loader):
        if "labels" in batch.keys():
            batch.pop("labels")
        sentence_embeddings = model(**batch)[0][:, 0]
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        embeddings.append(sentence_embeddings.detach().cpu().numpy())
    return np.concatenate(embeddings, axis=0)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # prepare tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        device_map="auto",
        torch_dtype=torch.float16 if model_args.half_precision else torch.float32,
    )
    model.eval()

    # prepare data
    # TODO: Inference for test data.
    train_dataset = Dataset.from_csv(data_args.train_data_path)
    preprocess = plain_processor(train_dataset, tokenizer, model_args.model_max_length)
    train_dataset = train_dataset.map(preprocess, batched=True, remove_columns=train_dataset.column_names)

    misconception_mapping = Dataset.from_csv(data_args.misconception_mapping)
    mis_preprocess = misconception_processor(misconception_mapping, tokenizer, model_args.model_max_length)
    misconception_mapping = misconception_mapping.map(mis_preprocess, batched=True, remove_columns=misconception_mapping.column_names)
    
    # inference
    train_embeddings = inference(model, train_dataset, batch_size=training_args.batch_size)
    misconception_embeddings = inference(model, misconception_mapping, batch_size=training_args.batch_size)
