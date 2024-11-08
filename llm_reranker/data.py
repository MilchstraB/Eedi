import os
import random
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset

import datasets
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, BatchEncoding


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
class RerankCollator(DataCollatorForSeq2Seq):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __call__(self, features):
        if isinstance(features[0], list):
            features = sum(features, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        collated = self.tokenizer.pad(
            features,
            padding="longest",
            return_tensors="pt",
        )

        return {"inputs": collated}
    

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
class ValCollator(RerankCollator):
    def __call__(self, features):
        batch_data = [f[0] for f in features]
        mis_ids = [f[1] for f in features]
        if isinstance(batch_data[0], list):
            features = sum(batch_data, [])
        if isinstance(mis_ids[0], list):
            mis_ids = sum(mis_ids, [])

        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            if self.pad_to_multiple_of is not None:
                max_label_length = (
                    (max_label_length + self.pad_to_multiple_of - 1)
                    // self.pad_to_multiple_of
                    * self.pad_to_multiple_of
                )

            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                if isinstance(feature["labels"], list):
                    feature["labels"] = (
                        feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                    )
                elif padding_side == "right":
                    feature["labels"] = np.concatenate([feature["labels"], remainder]).astype(np.int64)
                else:
                    feature["labels"] = np.concatenate([remainder, feature["labels"]]).astype(np.int64)

        collated = self.tokenizer.pad(
            features,
            padding="longest",
            return_tensors="pt",
        )

        return {"inputs": collated, "mis_ids": mis_ids}