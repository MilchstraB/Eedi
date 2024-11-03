import os
import random
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Dict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

class TrainDatasetForRerank(Dataset):
    def __init__(self, args, tokenizer: AutoTokenizer, mode="train"):
        data_path = args.train_data_path if mode == "train" else args.val_data_path
        
        if os.path.isdir(data_path):
            train_datasets = [os.path.join(data_path, file) for file in os.listdir(data_path)]
            self.dataset = datasets.load_dataset('json', data_files=train_datasets, split='train')
        else:
            self.dataset = datasets.load_dataset('json', data_files=data_path, split='train')

        self.args = args
        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer
        self.add_eos_token = args.add_eos_token
        self.max_length = args.max_length - 1 if args.add_eos_token else args.max_length

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        if self.add_eos_token:
            item["input_ids"].append(self.tokenizer.eos_token_id)
            item["attention_mask"].append(1)

        return item

    def __len__(self):
        return self.total_len
    
    def get_detailed_instruct(self, task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction is not None:
            query = self.get_detailed_instruct(self.args.query_instruction, query)
        batch_data = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        if self.args.passage_instruction is not None:
            pos = self.args.passage_instruction + pos
        batch_data.append(self.create_one_example(query, pos))

        negs = self.dataset[item]['neg']
        for neg in negs:
            if self.args.passage_instruction is not None:
                neg = self.args.passage_instruction + neg
            batch_data.append(self.create_one_example(query, neg))

        return batch_data
    

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
        data_path = args.val_data_path
        if os.path.isdir(data_path):
            train_datasets = [os.path.join(data_path, file) for file in os.listdir(data_path)]
            self.dataset = datasets.load_dataset('json', data_files=train_datasets, split='train')
        else:
            self.dataset = datasets.load_dataset('json', data_files=data_path, split='train')

        misconceptions_map = pd.read_csv(args.misconception_mapping)
        self.misconception_map = {
            row["MisconceptionName"]: row["MisconceptionId"]
            for _, row in misconceptions_map.iterrows()
        } 

        self.args = args
        self.total_len = len(self.dataset)
        self.tokenizer = tokenizer
        self.add_eos_token = args.add_eos_token
        self.max_length = args.max_length - 1 if args.add_eos_token else args.max_length

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        item = self.tokenizer.encode_plus(
            qry_encoding,
            doc_encoding,
            truncation=True,
            max_length=self.max_length,
            padding=False,
        )

        if self.add_eos_token:
            item["input_ids"].append(self.tokenizer.eos_token_id)
            item["attention_mask"].append(1)

        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction is not None:
            query = self.get_detailed_instruct(self.args.query_instruction, query)
        batch_data, mis_ids = [], []

        candidates = self.dataset[item]['candidates']
        for c in candidates:
            mis_ids.append(self.misconception_map[c])
            if self.args.passage_instruction is not None:
                c = self.args.passage_instruction + c
            batch_data.append(self.create_one_example(query, c))

        return batch_data, mis_ids
    

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