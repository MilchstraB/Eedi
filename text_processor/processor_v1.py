from collections import defaultdict
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer


class plain_processor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
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
            max_length=self.max_length,
            truncation=True,
        )

        results["input_ids"] = outputs["input_ids"]
        results["attention_mask"] = outputs["attention_mask"]

        if batch_data.get("MisconceptionId") is not None:
            labels = [int(id) if id else id for id in batch_data["MisconceptionId"]]
            results["labels"] = labels

        return results
    

class misconception_processor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch_data):
        batch_data = batch_data["MisconceptionName"]
        results = defaultdict(list)
        outputs = self.tokenizer(
            batch_data,
            max_length=self.max_length,
            truncation=True,
        )

        results["input_ids"] = outputs["input_ids"]
        results["attention_mask"] = outputs["attention_mask"]

        return results
