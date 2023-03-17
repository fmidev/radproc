import numpy as np


def db2lin(db):
    """decibels to linear scale"""
    return np.power(10, db/10)


def lin2db(lin):
    """linear to decibel scale"""
    return 10*np.log10(lin)


def interp_mba(xys, zs, m0=2, lo=-100, hi=100, resolution=50, margin=1):
    """scattered data interpolation with multilevel B-splines"""
    from mba import mba2 # optional dependency
    x = _square_grid(lo, hi, resolution)
    lom = lo-margin
    him = hi+margin
    interp = mba2([lom, lom], [him, him], [m0, m0], xys, zs)
    return interp(x)


def _square_grid(*args):
    """Generate a square cartesian meshgrid."""
    s = np.linspace(*args)
    return np.array(np.meshgrid(s, s)).transpose([1, 2, 0]).copy()


def weighted_median(arr, w):
    """general weighted median"""
    isort = np.argsort(arr)
    cs = w[isort].cumsum()
    cutoff = w.sum()/2
    try:
        return arr[isort][cs >= cutoff][0]
    except IndexError:
        return np.nan
