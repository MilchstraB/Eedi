import os
import random
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

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        batch_data = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        batch_data.append(self.create_one_example(query, pos))

        negs = self.dataset[item]['neg']
        for neg in negs:
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