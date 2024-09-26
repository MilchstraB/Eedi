from typing import Optional, Union

import torch
from torch import nn, Tensor
import torch.distributed as dist
from transformers import AutoModel
from peft import PeftModel

from utils import print_rank_0


class BiEncoderModel(nn.Module):
    def __init__(
        self,
        model: Union[AutoModel, PeftModel],
        normlized: bool = False,
        negatives_cross_device: bool = False,
        temperature: float = 1.0,
        use_inbatch_neg: bool = True,
        sentence_pooling_method: str = "last",
    ):
        super().__init__()
        self.model = model
        self.cross_entropy = nn.CrossEntropyLoss(reduction="mean")
        self.normlized = normlized
        self.sentence_pooling_method = sentence_pooling_method
        self.temperature = temperature
        self.use_inbatch_neg = use_inbatch_neg
        self.config = self.model.config
        self.device = self.model.device

        if not normlized:
            self.temperature = 1.0
            print_rank_0(
                "reset temperature = 1.0 due to using inner product to compute similarity"
            )
        if normlized:
            if self.temperature > 0.5:
                raise ValueError(
                    "Temperature should be smaller than 1.0 when use cosine similarity (i.e., normlized=True). Recommend to set it 0.01-0.1"
                )

        self.negatives_cross_device = negatives_cross_device
        if self.negatives_cross_device:
            if not dist.is_initialized():
                raise ValueError(
                    "Distributed training has not been initialized for representation all gather."
                )
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def last_token_pool(
        self, last_hidden_states: Tensor, attention_mask: Tensor
    ) -> Tensor:
        left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]

    def sentence_embedding(self, hidden_state, mask):
        if self.sentence_pooling_method == "mean":
            s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
            d = mask.sum(axis=1, keepdim=True).float()
            return s / d
        elif self.sentence_pooling_method == "cls":
            return hidden_state[:, 0]
        elif self.sentence_pooling_method == "last":
            return self.last_token_pool(hidden_state, mask)

    def encode(self, features):
        if features is None:
            return None
        
        features = {k: v.to(self.model.device) for k, v in features.items()}
        psg_out = self.model(
            input_ids=features["input_ids"],
            attention_mask=features["attention_mask"],
            return_dict=True,
        )
        p_reps = self.sentence_embedding(
            psg_out.last_hidden_state, features["attention_mask"]
        )
        if self.normlized:
            p_reps = torch.nn.functional.normalize(p_reps, dim=-1)
        return p_reps.contiguous()

    def compute_similarity(self, q_reps, p_reps):
        if len(p_reps.size()) == 2:
            return torch.matmul(q_reps, p_reps.transpose(0, 1))
        return torch.matmul(q_reps, p_reps.transpose(-2, -1))

    def forward(self, query, passage):
        q_reps = self.encode(query)
        p_reps = self.encode(passage)
        if self.training:
            if self.negatives_cross_device and self.use_inbatch_neg:
                q_reps = self._dist_gather_tensor(q_reps)
                p_reps = self._dist_gather_tensor(p_reps)

            group_size = p_reps.size(0) // q_reps.size(0)
            if self.use_inbatch_neg:
                scores = (
                    self.compute_similarity(q_reps, p_reps) / self.temperature
                )  
                scores = scores.view(q_reps.size(0), -1)

                target = torch.arange(
                    scores.size(0), device=scores.device, dtype=torch.long
                )
                target = target * group_size
                loss = self.compute_loss(scores, target)
            else:
                scores = (
                    self.compute_similarity(
                        q_reps[:, None, :, ],
                        p_reps.view(q_reps.size(0), group_size, -1),
                    ).squeeze(1)
                    / self.temperature
                ) 

                scores = scores.view(q_reps.size(0), -1)
                target = torch.zeros(
                    scores.size(0), device=scores.device, dtype=torch.long
                )
                loss = self.compute_loss(scores, target)

        else:
            scores = self.compute_similarity(q_reps, p_reps)
            loss = None
        return dict(
            loss=loss,
            scores=scores,
            q_reps=q_reps,
            p_reps=p_reps,
        )

    def compute_loss(self, scores, target):
        return self.cross_entropy(scores, target)

    def _dist_gather_tensor(self, t: Optional[torch.Tensor]):
        if t is None:
            return None
        t = t.contiguous()

        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)

        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)

        return all_tensors