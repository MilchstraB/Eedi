from collections import defaultdict
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from transformers import AutoTokenizer


class plain_processor:
    def __init__(
        self,
        train_data: str,
        misconception_mapping: str,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.train_data = pd.read_csv(train_data)
        self.misconception_mapping = pd.read_csv(misconception_mapping)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def preprocess_batch(self, batch_data: Dict[str, List[str]]):
        subject_name = batch_data["SubjectName"]
        contruct_name = batch_data["ConstructName"]
        question_text = batch_data["QuestionText"]
        answer_text = batch_data["Value"]
        return subject_name, contruct_name, question_text, answer_text
    
    def format_texts(self, subject_name, contruct_name, question_text, answer_text):
        texts = []
        for subj, cont, ques, ans in zip(subject_name, contruct_name, question_text, answer_text):
            text = f"{cont} {ques} {ans}"
            texts.append(text)
        return texts
    
    def __call__(self, batch_data):
        batch = self.preprocess_batch(batch_data)
        texts = self.format_texts(*batch)
        results = defaultdict(list)
        outputs = self.tokenizer(
            texts,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
        )

        results["input_ids"] = outputs["input_ids"]
        results["attention_mask"] = outputs["attention_mask"]

        if batch_data.get("MisconceptionId") is not None:
            misconception_ids = batch_data["MisconceptionId"].tolist()
            results["labels"] = misconception_ids

        return results