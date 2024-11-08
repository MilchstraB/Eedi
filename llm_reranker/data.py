import os
import random
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import datasets
from transformers import AutoTokenizer, DataCollatorWithPadding, BatchEncoding


class TrainDatasetForRerank(Dataset):
    def __init__(self, args, tokenizer: AutoTokenizer):
        if os.path.isdir(args.train_data_path):
            train_datasets = [os.path.join(args.train_data_path, file) for file in os.listdir(args.train_data_path)]
            self.dataset = datasets.load_dataset('json', data_files=train_datasets, split='train')
        else:
            self.dataset = datasets.load_dataset('json', data_files=args.train_data_path, split='train')

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.template = args.template
        self.prompt = args.prompt
        self.max_length = args.max_length - 1 if self.tokenizer.bos_token_id else args.max_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']

        pos = random.choice(self.dataset[item]['pos'])
        negs = self.dataset[item]['neg']
        passages = [pos] + negs

        # Encoding
        inputs = []
        prompt = self.tokenizer(
            self.prompt + "Yes",
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

        for passage in passages:
            text = self.template.format_map({"query": query, "passage": passage})
            encoding = self.tokenizer(
                text,
                max_length=self.max_length - len(prompt),
                padding=False,
                truncation=True,
            )["input_ids"]
            encoding = [self.tokenizer.bos_token_id] + encoding + prompt \
                if self.tokenizer.bos_token_id else encoding + prompt
            attention_mask = [1] * len(encoding)
            label = [-100] * (len(encoding) - 1) + [encoding[-1]]
            inputs.append({"input_ids": encoding, "attention_mask": attention_mask, "labels": label})

        return inputs

 
@dataclass
class GroupCollator(DataCollatorWithPadding):
    def __call__(
        self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        if isinstance(features[0], list):
            features = sum(features, [])
        return super().__call__(features)
    

class ValDatasetForRerank(Dataset):
    def __init__(self, args, tokenizer: AutoTokenizer):
        self.dataset = datasets.load_dataset('json', data_files=args.val_data_path, split='train')

        misconceptions_map = pd.read_csv(args.misconception_mapping)
        self.misconception_map = {
            row["MisconceptionName"]: row["MisconceptionId"]
            for _, row in misconceptions_map.iterrows()
        } 

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.dataset)
        self.template = args.template
        self.prompt = args.prompt
        self.max_length = args.max_length - 1 if self.tokenizer.bos_token_id else args.max_length

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> List[BatchEncoding]:
        query = self.dataset[item]['query']
        candidates = self.dataset[item]['candidates']

        # Encoding
        inputs, mis_ids = [], []
        prompt = self.tokenizer(
            self.prompt + "Yes",
            return_tensors=None,
            add_special_tokens=False,
        )["input_ids"]

        for passage in candidates:
            mis_ids.append(self.misconception_map[passage])
            text = self.template.format_map({"query": query, "passage": passage})
            encoding = self.tokenizer(
                text,
                max_length=self.max_length - len(prompt),
                padding=False,
                truncation=True,
            )["input_ids"]
            encoding = [self.tokenizer.bos_token_id] + encoding + prompt \
                if self.tokenizer.bos_token_id else encoding + prompt
            attention_mask = [1] * len(encoding)
            label = [-100] * (len(encoding) - 1) + [encoding[-1]]
            inputs.append({"input_ids": encoding, "attention_mask": attention_mask, "labels": label})

        return inputs, mis_ids


@dataclass
class ValCollator(DataCollatorWithPadding):
    def __call__(
        self, features
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        batch_data = [f[0] for f in features]
        mis_ids = [f[1] for f in features]

        if isinstance(batch_data[0], list):
            batch_data = sum(batch_data, [])
        if isinstance(mis_ids[0], list):
            mis_ids = sum(mis_ids, [])

        batch_data = self.tokenizer.pad(batch_data, padding="longest", return_tensors="pt")
        return {"inputs": batch_data, "mis_ids": mis_ids}