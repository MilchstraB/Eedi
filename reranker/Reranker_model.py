import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput


class CrossEncoder(nn.Module):
    def __init__(
        self, 
        model: PreTrainedModel,
        per_device_train_batch_size: int,
        train_group_size: int,
    ):
        super().__init__()
        self.model = model
        self.config = self.model.config
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.per_device_train_batch_size = per_device_train_batch_size
        self.train_group_size = train_group_size

        # https://github.com/FlagOpen/FlagEmbedding/issues/1112
        self.register_buffer(
            'target_label',
            torch.zeros(per_device_train_batch_size, dtype=torch.long)
        )

    def gradient_checkpointing_enable(self, **kwargs):
        self.model.gradient_checkpointing_enable(**kwargs)

    def forward(self, **kwargs):
        ranker_out: SequenceClassifierOutput = self.model(**kwargs, return_dict=True)
        logits = ranker_out.logits

        scores = logits.view(
            self.per_device_train_batch_size,
            self.train_group_size
        )
        loss = self.cross_entropy(scores, self.target_label)

        return SequenceClassifierOutput(
            loss=loss,
            **ranker_out,
        )