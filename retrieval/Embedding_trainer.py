import os
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict

import torch
from torch import Tensor
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalLoopOutput
from transformers import DataCollatorWithPadding

from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def last_token_pool(
    last_hidden_states: Tensor,            
    attention_mask: Tensor
) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def sentence_embedding(hidden_state, mask, sentence_pooling_method):
    if sentence_pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif sentence_pooling_method == "cls":
        return hidden_state[:, 0]
    elif sentence_pooling_method == "last":
        return last_token_pool(hidden_state, mask)
    

class RetrievalTrainer(Trainer):
    """Trainer with retrieval-based evaluation in batch negatives."""
    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return
        
        self.model.eval()
        sentence_pooling_method = self.model.sentence_pooling_method
        model_base = self.model.module if hasattr(self.model, 'module') else self.model
        model_base = model_base.model

        data_collator = DataCollatorWithPadding(self.data_collator.tokenizer)

        text_dataloader = DataLoader(
            self.eval_dataset["text"],
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            collate_fn=data_collator,
        )
        text_dataloader = self.accelerator.prepare(text_dataloader)

        text_embeddings, labels = [], []
        for _, inputs in enumerate(tqdm(text_dataloader, desc="Encoding text: ")):
            target = inputs.pop("labels")
            outputs = model_base(**inputs).last_hidden_state
            sentence_embeddings = sentence_embedding(outputs, inputs['attention_mask'], sentence_pooling_method)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
  
            results = self.accelerator.gather_for_metrics(sentence_embeddings.contiguous())
            target = self.accelerator.gather_for_metrics(target.contiguous())
            text_embeddings.extend(results.tolist())
            labels.extend(target.tolist())

        text_embeddings = np.stack(text_embeddings, axis=0)
        labels = [[e] for e in labels]
        del text_dataloader

        mis_dataloader = DataLoader(
            self.eval_dataset["misconception"],
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            collate_fn=data_collator,
        )
        mis_dataloader = self.accelerator.prepare(mis_dataloader)

        misconception_embeddings = []
        for _, inputs in enumerate(tqdm(mis_dataloader, desc="Encoding misconception: ")):
            outputs = model_base(**inputs).last_hidden_state
            sentence_embeddings = sentence_embedding(outputs, inputs['attention_mask'], sentence_pooling_method)
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            results = self.accelerator.gather_for_metrics(sentence_embeddings.contiguous())
            misconception_embeddings.extend(results.tolist())
        
        misconception_embeddings = np.stack(misconception_embeddings, axis=0)
        del mis_dataloader
        torch.cuda.empty_cache()
        
        cos_sim_arr = cosine_similarity(text_embeddings, misconception_embeddings)
        sorted_indices = np.argsort(-cos_sim_arr, axis=1)
        preds = sorted_indices[:, :25].tolist()

        if self.args.process_index == 0:
            metrics = [self.compute_metrics(preds, labels)]
        else:
            metrics = [None]
            
        # NOTE: broadcast across devices
        dist.broadcast_object_list(metrics, src=0)
        metrics = metrics[0]
        self.accelerator.wait_for_everyone()

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_") and key != "epoch":
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        output = EvalLoopOutput(predictions=preds, metrics=metrics, label_ids=None, num_samples=len(preds))
        self.log(output.metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, output.metrics)
        self._memory_tracker.stop_and_update_metrics(output.metrics)

        return output.metrics
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save = model_to_save.model
        model_to_save.save_pretrained(
            output_dir, safe_serialization=self.args.save_safetensors
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)