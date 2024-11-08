from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn, Tensor
from transformers import PreTrainedModel
from transformers.file_utils import ModelOutput


@dataclass
class RerankerOutput(ModelOutput):
    loss: Optional[Tensor] = None
    scores: Optional[Tensor] = None


class BiEncoderModel(nn.Module):
    def __init__(
        self, 
        model: PreTrainedModel,
        per_device_train_batch_size: int,
        train_group_size: int,
        yes_loc: int,
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.per_device_train_batch_size = per_device_train_batch_size
        self.train_group_size = train_group_size
        self.yes_loc = yes_loc

        # https://github.com/FlagOpen/FlagEmbedding/issues/1112
        self.register_buffer(
            'target_label',
            torch.zeros(per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def enable_input_require_grads(self, **kwargs):
        self.model.enable_input_require_grads(**kwargs)

    def encode(self, inputs):
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True
        )
        _, max_indices = torch.max(inputs["labels"], dim=1)
        # Output the probability distribution of the next token at the current position.
        # Move right one unit.
        predict_indices = max_indices - 1
        logits = [outputs.logits[i, predict_indices[i], :] for i in range(outputs.logits.shape[0])]
        logits = torch.stack(logits, dim=0)
        scores = logits[:, self.yes_loc]
        return scores.contiguous()

    def forward(self, inputs):
        logits = self.encode(inputs)

        scores = logits.view(
            self.per_device_train_batch_size,
            self.train_group_size
        )
        loss = self.cross_entropy(scores, self.target_label)

        return RerankerOutput(
            loss=loss,
            scores=logits,
        )