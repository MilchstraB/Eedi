import torch
import numpy as np


def apk(actual, predicted, k=25):
    """ Computes the average precision at k.

    Parameters
    ----------
    actual : A list of elements that are to be predicted (order doesn't matter)
    predicted : A list of predicted elements (order does matter)

    Returns
    -------
    score : The average precision at k over the input lists
    """

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        # first condition checks whether it is valid prediction
        # second condition checks if prediction is not repeated
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    return score / min(len(actual), k)

def mapk(actual, predicted, k=25):
    """ Computes the mean average precision at k.

    Parameters
    ----------
    actual : A list of lists of elements that are to be predicted (order doesn't matter)
    predicted : list of lists of predicted elements (order matters in the lists)
    k : The maximum number of predicted elements

    Returns
    -------
    score : The mean average precision at k over the input lists
    """

    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])


def print_rank_0(message):
    """If distributed is initialized, print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


def get_optimizer_grouped_parameters(model, base_lr, score_lr, weight_decay):
    no_decay = ["bias", "layernorm"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "score" in n and not any(nd in n for nd in no_decay)],
            "lr": score_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "score" in n and any(nd in n for nd in no_decay)],
            "lr": score_lr,
            "weight_decay": 0.0,
        },
        {
            "params": [p for n, p in model.named_parameters() if "score" not in n and not any(nd in n for nd in no_decay)],
            "lr": base_lr,
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() if "score" not in n and any(nd in n for nd in no_decay)],
            "lr": base_lr,
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters
