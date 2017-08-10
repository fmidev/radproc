#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Vertical profile classification
@author: Jussi Tiira
"""
import numpy as np


def m2km(m, pos):
    '''formatting m in km'''
    return '{:.0f}'.format(m*1e-3)


def cloud_top_h(z, zmin=0):
    top = (z>zmin).loc[::-1].idxmax()
    top[top==z.index[-1]] = np.nan
    return top
