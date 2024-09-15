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
