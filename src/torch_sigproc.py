#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2022/8/2
"""

import torch


def gencoords_base(N, d,device=None):
    x = torch.arange((-N//2), (N//2), dtype=torch.float32,device=device)
    c = x.clone()
    for i in range(1,d):
        c = torch.column_stack([x.repeat(N**i), c.repeat(N)])
    return c


def genmask(N, d, radwn_range, device=None,exclude_dc=False, herm=False):
    c = gencoords_base(N, d,device=device)
    print(c.size())
    if isinstance(radwn_range,tuple):
        radwn_min = radwn_range[0]
        radwn_max = radwn_range[1]
    else:
        radwn_min = 0
        radwn_max = radwn_range
    if exclude_dc:
        radwn_min = max(1e-8,radwn_min)

    r2 = torch.sum(c**2, axis=1)
    ret = torch.logical_and(r2 < radwn_max**2, r2 >= radwn_min**2)
    if herm:
        assert N % 2 == 0, "Even sizes only"
        ret = torch.logical_and(ret, torch.arange(N**d) > torch.ravel_multi_index((N//2,) * d, (N,) * d))
    return ret


def gencoords (N, d, radwn=None, device=None,exclude_dc=False, herm=False):
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
        truncmask = genmask(N, d, radwn,device, exclude_dc, herm)
        c = c[truncmask, :]
    return c


def butterworth(N, d, cutoffwn, order=1,device=None):
    """
    巴特沃斯滤波器
    """
    xy = gencoords(N, d,device)
    r2 = (xy**2).sum(axis=1).reshape((N,)*d)
    f = 1.0 / (1.0 + (r2 / cutoffwn ** 2)**order)
    return f


def lowpass_mask(N, d, radwn=None, lowpasstype='butterworth', lowpassorder=1,device=None):
    """
    低通滤波的Mask
    """
    if radwn is None:
        return 1.0
    if lowpasstype == 'butterworth':
        mask = butterworth(N, d, radwn, lowpassorder,device)
    return mask
