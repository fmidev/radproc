"""
"""
import numpy as np


def find(arr, value):
    """find closest value using argmin"""
    return abs(arr-value).argmin()


def m2km(m, pos):
    '''formatting m in km'''
    return '{:.0f}'.format(m*1e-3)


def echo_top_h(z, zmin=-8):
    top = (z>zmin).loc[::-1].idxmax()
    top[top==z.index[-1]] = np.nan
    return top


def source2dict(src):
    """radar source metadata as dictionary"""
    l = src.split(',')
    d = {}
    for pair in l:
        key, value = pair.split(':')
        d[key] = value
    return d
