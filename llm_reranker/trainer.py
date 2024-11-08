import os
import logging
import numpy as np
from tqdm import tqdm
from typing import Optional, List, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalLoopOutput
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled

from .data import ValCollator

logger = logging.getLogger(__name__)


class RerankTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.can_return_loss = True # can return loss without labels
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save = model_to_save.model
        
        if is_deepspeed_zero3_enabled():
            prefix = "model."
            assert all(k.startswith(prefix) for k in state_dict.keys()), list(state_dict.keys())
            state_dict = {k[len(prefix):]: v for k, v in state_dict.items()}
        
        model_to_save.save_pretrained(
            output_dir, safe_serialization=self.args.save_safetensors, state_dict=state_dict,
        )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return
        
        self.model.eval()
        raw_data = self.eval_dataset.dataset
        group_size = len(raw_data[0]["candidates"])

        data_collator = ValCollator(self.data_collator.tokenizer)
        eval_dataloader = DataLoader(
            self.eval_dataset,
            collate_fn=data_collator,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
        )
        eval_dataloader = self.accelerator.prepare(eval_dataloader)

        probas, mis_ids = [], []
        for _, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating: ")):
            logits = self.model.encode(batch["inputs"])

            # It is neccessary to reshape the logits to the original shape with the first dimension as the batch size.
            # Because the gather_for_metrics() function will automatically removes the duplicated data according to the 
            # total length of the data. And will save the last batch as [:len(dataset) - process_index * batch_size]. 
            proba = logits.sigmoid().reshape(-1, group_size)
            mis_id = np.array(batch["mis_ids"]).reshape(-1, group_size)

            proba = self.accelerator.gather_for_metrics(proba.contiguous())
            mis_id = self.accelerator.gather_for_metrics(mis_id) 
            probas.extend(proba.tolist())
            mis_ids.extend(mis_id)

        results = np.stack([probas, mis_ids], axis=2, dtype=np.float32)
        sorted_indices = np.argsort(-results[:, :, 0], axis=1)
        sorted_results = np.take_along_axis(results, sorted_indices[:, :, np.newaxis], axis=1)
        preds = sorted_results[:, :25, 1].astype(int).tolist()
        labels = [[self.eval_dataset.misconception_map[e]] for e in raw_data["label"]]

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