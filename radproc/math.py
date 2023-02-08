import numpy as np


def weighted_median(arr, w):
    """general weighted median"""
    isort = np.argsort(arr)
    cs = w[isort].cumsum()
    cutoff = w.sum()/2
    try:
        return arr[isort][cs >= cutoff][0]
    except IndexError:
        return np.nan
