import random
from collections import defaultdict
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer


class reranker_processor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        template: Optional[str] = "<|im_start|>{QUERY}\n\n{DOC}<|im_end|>",
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template

    def create_one_example(self, qry_encoding: str, doc_encoding: str):
        data_dic = {"QUERY": qry_encoding, "DOC": doc_encoding}
        text = self.template.format_map(data_dic)
        item = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
        )
        return item["input_ids"], item["attention_mask"]
    
    def __call__(self, batch_data):
        batch_query = batch_data["query"]
        batch_pos = [random.choice(p) for p in batch_data["pos"]] # only one pos now
        batch_negs = batch_data["neg"]

        results = defaultdict(list)
        input_ids, attention_masks, labels = [], [], []

        for query, pos, negs in zip(batch_query, batch_pos, batch_negs):
            q_ids, q_mask = self.create_one_example(query, pos)
            input_ids.append(q_ids)
            attention_masks.append(q_mask)
            labels.append(1)

            for neg in negs:
                n_ids, n_mask = self.create_one_example(query, neg)
                input_ids.append(n_ids)
                attention_masks.append(n_mask)
                labels.append(0)

        results["input_ids"] = input_ids
        results["attention_mask"] = attention_masks
        results["labels"] = labels

        return results