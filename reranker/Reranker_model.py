import torch
from torch import nn
from transformers import AutoModelForSequenceClassification, PreTrainedModel
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

    def forward(self, batch):
        ranker_out: SequenceClassifierOutput = self.model(**batch, return_dict=True)
        logits = ranker_out.logits

        if self.training:
            scores = logits.view(
                self.per_device_train_batch_size,
                self.train_group_size
            )
            loss = self.cross_entropy(scores, self.target_label)

            return SequenceClassifierOutput(
                loss=loss,
                **ranker_out,
            )
        else:
            return ranker_out

    @classmethod
    def from_pretrained(
        cls, 
        per_device_train_batch_size: int, 
        train_group_size: int,
        *args, 
        **kwargs
    ):
        model = AutoModelForSequenceClassification.from_pretrained(*args, **kwargs)
        reranker = cls(model, per_device_train_batch_size, train_group_size)
        return reranker