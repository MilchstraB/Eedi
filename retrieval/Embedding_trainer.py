from tqdm import tqdm
from typing import Optional, List, Dict

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from transformers.trainer import Trainer
from transformers.trainer_utils import EvalLoopOutput


class RetrievalTrainer(Trainer):
    """Trainer with retrieval-based evaluation in batch negatives."""
    @torch.no_grad()
    def evaluate(self, eval_dataset: Optional[Dataset] = None, ignore_keys: Optional[List[str]] = None, metric_key_prefix: str = "eval") -> Dict[str, float]:
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return
        
        self.model.eval()

        dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            pin_memory=True,
            collate_fn=self.data_collator,
        )

        preds, labels = [], []
        for _, inputs in enumerate(tqdm(dataloader, desc="Evaluation")):
            outputs = self.model(**inputs)
            group_size = outputs["p_reps"].size(0) // outputs["q_reps"].size(0)
            target = torch.arange(
                outputs["scores"].size(0), device=outputs["scores"].device, dtype=torch.long
            )
            target = torch.unsqueeze(target * group_size, dim=1)
            results = outputs["scores"].topk(k=group_size, dim=1, largest=True, sorted=True)[1]
            results = self.accelerator.gather_for_metrics(results.contiguous())
            target = self.accelerator.gather_for_metrics(target.contiguous())
            preds.extend(results.tolist())
            labels.extend(target.tolist())

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