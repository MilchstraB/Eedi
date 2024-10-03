import os
import json
import random
import pandas as pd
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict

import datasets
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

class TrainDatasetForEmbedding(Dataset):
    def __init__(self, args, mode="train"):
        data_path = args.train_data_path if mode == "train" else args.val_data_path
        
        if os.path.isdir(data_path):
            train_datasets = []
            for file in os.listdir(data_path):
                temp_dataset = datasets.load_dataset('json', data_files=os.path.join(data_path, file), split='train')
                if len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
            self.dataset = datasets.concatenate_datasets(train_datasets)
        else:
            self.dataset = datasets.load_dataset('json', data_files=data_path, split='train')

        self.args = args
        self.total_len = len(self.dataset)

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[str, List[str]]:
        query = self.dataset[item]['query']
        if self.args.query_instruction is not None:
            query = self.args.query_instruction + query

        passages = []

        assert isinstance(self.dataset[item]['pos'], list)
        pos = random.choice(self.dataset[item]['pos'])
        passages.append(pos)

        negs = self.dataset[item]['neg']
        passages.extend(negs)

        if self.args.passage_instruction is not None:
            passages = [self.args.passage_instruction + p for p in passages]

        return query, passages
    

@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    def __init__(self, tokenizer, add_eos_token=True, **kwargs):
        super().__init__(tokenizer=tokenizer, **kwargs)
        self.add_eos_token = add_eos_token
        if add_eos_token: self.max_length -= 1
    
    def padding_score(self, teacher_score):
        group_size = None
        for scores in teacher_score:
            if scores is not None:
                group_size = len(scores)
                break
        if group_size is None:
            return None

        padding_scores = [100.0] + [0.0] * (group_size - 1)
        new_teacher_score = []
        for scores in teacher_score:
            if scores is None:
                new_teacher_score.append(padding_scores)
            else:
                new_teacher_score.append(scores)
        return new_teacher_score

    def mask_pad_token(self, q):
        if random.random() > 0.9:
            tensor = q['input_ids'].float()

            eos_token_id = self.tokenizer.eos_token_id
            eos_mask = (tensor == eos_token_id).float() 
            
            mask = torch.rand(tensor.shape)
            mask = (mask > 0.9).float() * (1 - eos_mask)
            
            tensor = tensor * (1 - mask) + 2 * mask
            tensor = tensor.long()
            q['input_ids'] = tensor
        return q

    def _add_eos_token(self, output):
        for input_ids, attention_mask in zip(output['input_ids'], output['attention_mask']):
            input_ids.append(self.tokenizer.eos_token_id)
            attention_mask.append(1)

        return output

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]

        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            truncation=True,
            max_length=self.max_length,
        )
        if self.add_eos_token:
            q_collated = self._add_eos_token(q_collated)
        q_collated = self.tokenizer.pad(q_collated, padding="longest", return_tensors="pt")
        q_collated = self.mask_pad_token(q_collated)

        d_collated = self.tokenizer(
            passage,
            truncation=True,
            max_length=self.max_length,
        )
        if self.add_eos_token:
            d_collated = self._add_eos_token(d_collated)
        d_collated = self.tokenizer.pad(d_collated, padding="longest", return_tensors="pt")
        d_collated = self.mask_pad_token(d_collated)

        return {"query": q_collated, "passage": d_collated}
    

class plain_processor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        misconception_mapping: str,
        add_eos_token: bool = True,
    ):
        self.tokenizer = tokenizer
        self.add_eos_token = add_eos_token
        self.max_length = max_length - 1 if add_eos_token else max_length

        misconception_mapping = pd.read_csv(misconception_mapping)
        self.misconception_mapping = {
            row["MisconceptionName"]: row["MisconceptionId"]
            for _, row in misconception_mapping.iterrows()
        }
    
    def __call__(self, batch_data):
        querys, pos_samples = [], []
        for q, p in zip(batch_data["query"], batch_data["pos"]):
            querys.append(q)
            assert len(p) == 1, "Only one positive sample is allowed in plain processor"
            pos_samples.append(p[0])
        
        results = defaultdict(list)
        outputs = self.tokenizer(
            querys,
            max_length=self.max_length,
            truncation=True,
        )
        
        if self.add_eos_token:
            for input_ids, attention_mask in zip(outputs['input_ids'], outputs['attention_mask']):
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)

        results["input_ids"] = outputs["input_ids"]
        results["attention_mask"] = outputs["attention_mask"]

        labels = map(self.misconception_mapping.get, pos_samples)
        results["labels"] = list(labels)

        return results