#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2022/8/2
"""

import numpy as np


def gencoords_base(N, d):
    x = np.arange((-N//2), (N//2), dtype=np.float32)
    c = x.copy()
    for i in range(1,d):
        c = np.column_stack([np.tile(x, N**i), np.repeat(c, N, axis=0)])
    return c


def genmask(N, d, radwn_range, exclude_dc=False, herm=False):
    c = gencoords_base(N, d)
    if isinstance(radwn_range,tuple):
        radwn_min = radwn_range[0]
        radwn_max = radwn_range[1]
    else:
        radwn_min = 0
        radwn_max = radwn_range
    if exclude_dc:
        radwn_min = max(1e-8,radwn_min)

    r2 = np.sum(c**2, axis=1)
    ret = np.logical_and(r2 < radwn_max**2, r2 >= radwn_min**2)
    if herm:
        assert N % 2 == 0, "Even sizes only"
        ret = np.logical_and(ret, np.arange(N**d) > np.ravel_multi_index((N//2,) * d, (N,) * d))
    return ret


def gencoords (N, d, radwn=None, exclude_dc=False, herm=False):
    """
    generate coordinates of all points in an NxN..xN grid with d dimensions
    coords in each dimension are [-N/2, N/2)
    N should be even
    radwn is the cutoff radius (in units of wavenumber)
    exclude_dc=False will omit the DC component. Only works if radwn is not None.
    Note: You can't use keyword args with a memoize function, only positional.
    Defaults still work though.
    Note: Now switched to x,y,z with x being fast order
    """
    c = gencoords_base(N, d)
    if radwn is not None:
        truncmask = genmask(N, d, radwn, exclude_dc, herm)
        c = c[truncmask, :]
    return c


def butterworth(N, d, cutoffwn, order=1):
    """
    巴特沃斯滤波器
    """
    xy = gencoords(N, d)
    r2 = (xy**2).sum(axis=1).reshape((N,)*d)
    f = 1.0 / (1.0 + (r2 / cutoffwn ** 2)**order)
    return f


def lowpass_mask(N, d, radwn=None, lowpasstype='butterworth', lowpassorder=1):
    """
    低通滤波的Mask
    """
    if radwn is None:
        return 1.0
    if lowpasstype == 'butterworth':
        mask = butterworth(N, d, radwn, lowpassorder)
    return mask
