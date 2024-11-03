from collections import defaultdict
from typing import Dict, List, Optional, Union

from transformers import AutoTokenizer


class plain_processor:
    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_length: int,
        template: Optional[str] = "{ConstructName} {QuestionText} {Answer}",
        add_eos_token: bool = True,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.add_eos_token = add_eos_token
        self.max_length = max_length - 1 if add_eos_token else max_length

    def preprocess_batch(self, batch_data: Dict[str, List[str]]):
        subject_name = batch_data["SubjectName"]
        contruct_name = batch_data["ConstructName"]
        question_text = batch_data["QuestionText"]
        answer_text = batch_data["Value"]
        return subject_name, contruct_name, question_text, answer_text
    
    def format_texts(self, subject_name, contruct_name, question_text, answer_text):
        texts = []
        for subj, cont, ques, ans in zip(subject_name, contruct_name, question_text, answer_text):
            data_dic = {
                "SubjectName": subj,
                "ConstructName": cont,
                "QuestionText": ques,
                "Answer": ans,
            }
            text = self.template.format_map(data_dic)
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

        if self.add_eos_token:
            for input_ids, attention_mask in zip(outputs['input_ids'], outputs['attention_mask']):
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)

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
        add_eos_token: bool = True,
        passage_instruction: Optional[str] = None,
    ):
        self.tokenizer = tokenizer
        self.add_eos_token = add_eos_token
        self.max_length = max_length - 1 if add_eos_token else max_length
        self.passage_instruction = passage_instruction
    
    def __call__(self, batch_data):
        batch_data = batch_data["MisconceptionName"]
        if self.passage_instruction is not None:
            batch_data = [self.passage_instruction + text for text in batch_data]
        results = defaultdict(list)
        outputs = self.tokenizer(
            batch_data,
            max_length=self.max_length,
            truncation=True,
        )

        if self.add_eos_token:
            for input_ids, attention_mask in zip(outputs['input_ids'], outputs['attention_mask']):
                input_ids.append(self.tokenizer.eos_token_id)
                attention_mask.append(1)

        results["input_ids"] = outputs["input_ids"]
        results["attention_mask"] = outputs["attention_mask"]

        return results