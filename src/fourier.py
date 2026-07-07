#!/usr/bin/env python
# -- coding: utf-8 --
"""
Copyright (c) 2022. All rights reserved.
Created by C. L. Wang on 2022/7/26
"""


import warnings
from builtins import range
from threading import Lock

import numpy as n

fourier_lock = Lock()

import pyfftw
fftmod = pyfftw.interfaces.numpy_fft
pyfftw.interfaces.cache.enable()

import multiprocessing
# fft_threads = multiprocessing.cpu_count()
fft_threads = min(multiprocessing.cpu_count(), 4)


from numba import jit as autojit
@autojit
def alternatesign2(v, nrm=1.0):
    for p in range(v.shape[0]):
        for r in range(v.shape[1]):
            for c in range(v.shape[2]):
                v[p,r,c] *= nrm
                if (r+c)%2:
                    v[p,r,c] *= -1.0

@autojit
def alternatesign3(v, nrm=1.0):
    for p in range(v.shape[0]):
        for r in range(v.shape[1]):
            for c in range(v.shape[2]):
                v[p,r,c] *= nrm
                if (r+c+p)%2:
                    v[p,r,c] *= -1.0

def fftcenter2(ims, th=1):
    assert all([d%2 == 0 for d in ims.shape[-2:0]])
    assert all([d == ims.shape[-1] for d in ims.shape[-2:0]])
    tmp = ims.copy(order='c').reshape((-1,)+ ims.shape[-2:]) # make 3D if 2D
    alternatesign2(tmp)
    fourier_lock.acquire()
    fim = fftmod.fftn(tmp, axes=(-2,-1), threads=th)
    fourier_lock.release()
    nrm = 1.0/n.sqrt(n.prod(ims.shape[-2:]))
    alternatesign2(fim, nrm)
    del tmp
    return fim.reshape(ims.shape)

def ifftcenter2(fims, th=1):
    assert all([d%2 == 0 for d in fims.shape[-2:0]])
    assert all([d == fims.shape[-1] for d in fims.shape[-2:0]])
    tmp = fims.copy(order='c').reshape((-1,)+ fims.shape[-2:]) # make 3D if 2D
    alternatesign2(tmp)
    fourier_lock.acquire()
    ims = fftmod.ifftn(tmp, axes=(-2,-1), threads=th).real
    fourier_lock.release()
    nrm = n.sqrt(n.prod(fims.shape[-2:]))
    alternatesign2(ims, nrm)
    del tmp
    return ims.reshape(fims.shape)

def fftcenter3(vol, th=1):
    N = vol.shape[-1]
    assert all([d%2 == 0 for d in vol.shape])
    assert all([d == N for d in vol.shape])
    tmp = vol.copy(order='c')
    alternatesign3(tmp)
    fourier_lock.acquire()
    fv = fftmod.fftn(tmp, threads=th)
    fourier_lock.release()
    nrm = 1.0/n.sqrt(n.prod(vol.shape))
    ### HACK ALERT:
    if N%4 != 0:
        nrm *= -1.0
    alternatesign3(fv, nrm)
    del tmp
    return fv

def ifftcenter3(fv, th=1):
    N = fv.shape[-1]
    assert all([d%2 == 0 for d in fv.shape])
    assert all([d == N for d in fv.shape])
    tmp = fv[:,:,:N//2+1].copy(order='c')
    alternatesign3(tmp)
    fourier_lock.acquire()
    v = fftmod.irfftn(tmp, threads=th)
    fourier_lock.release()
    nrm = n.sqrt(n.prod(fv.shape))
    ### HACK ALERT:
    if N%4 != 0:
        nrm *= -1.0
    alternatesign3(v, nrm)
    del tmp
    return v

def fft(x: n.array, stack=False):
    """Forward Fourier Transform.
    2D (last two axes) if stack is True or x.ndim==2 otherwise 3D."""
    assert x.ndim in [2,3] , "fourier.F only supports 2D or 3D transforms!"
    if stack or x.ndim==2:
        return fftcenter2(x, fft_threads)
    else:
        if not n.all(n.array(x.shape) == x.shape[0]):
            warnings.warn('Taking a 3D Fourier transform of a non-cubic volume.. probably a mistake!')
        return fftcenter3(x, fft_threads)

def ifft(X, stack=False):
    """Inverse Fourier Transform.
    2D (last two axes) if stack is True or X.ndim==2 otherwise 3D."""
    assert X.ndim in [2,3] , "fourier.F only supports 2D or 3D transforms!"
    if stack or X.ndim==2:
        return ifftcenter2(X, fft_threads)
    else:
        if not n.all(n.array(X.shape) == X.shape[0]):
            warnings.warn('Taking a 3D Fourier transform of a non-cubic volume.. probably a mistake!')
        return ifftcenter3(X, fft_threads)

def truncate(X, N, stack=False):
    if n.array(X).size == 1: return X
    assert X.ndim in [2,3] , "fourier.truncate only supports 2D or 3D transforms!"
    old_N = X.shape[-1]
    a = (old_N//2) - (N//2)
    b = (old_N//2) + (N//2)
    if stack:
        return X[:,a:b,a:b]
    if X.ndim==2:
        return X[a:b,a:b]
    else:
        if not n.all(n.array(X.shape) == X.shape[0]):
            warnings.warn('Non-cubic volume.. probably a mistake!')
        return X[a:b,a:b,a:b]
def zeropad(X, N, stack=False):
    assert X.ndim in [2,3] , "fourier.zeropad only supports 2D or 3D transforms!"
    old_N = X.shape[-1]
    a = (N//2) - (old_N//2)
    b = (N//2) + (old_N//2)
    if stack:
        new_X = n.zeros((X.shape[0], N, N), dtype=X.dtype)
        new_X[:,a:b,a:b] = X
    elif X.ndim==2:
        new_X = n.zeros((N, N), dtype=X.dtype)
        new_X[a:b,a:b] = X
    else:
        if not n.all(n.array(X.shape) == X.shape[0]):
            warnings.warn('Non-cubic volume.. probably a mistake!')
        new_X = n.zeros((N, N, N), dtype=X.dtype)
        new_X[a:b,a:b,a:b] = X
    return new_X

def resample_fspace(X, N, rescale = False, stack = False):
    """Resample X to N x N [x N]
    if rescale: result preserves max
    if not rescale: result preserves sum
    """
    old_N = X.shape[-1]
    nrm = (n.sqrt(N**3) / n.sqrt(old_N**3)) if rescale else 1.0 # incorrect for 2D
    if old_N > N:
        return truncate(X, N, stack=stack) * nrm
    elif old_N == N:
        return X
    elif old_N < N:
        return zeropad(X, N, stack=stack) * nrm

def resample_real(x, N, rescale = False, stack=False):
    """Resample x to N x N [x N]
    if rescale: result preserves max
    if not rescale: result preserves sum
    """
    return ifft(resample_fspace(
           fft(x, stack=stack), N, rescale, stack), stack=stack)

def make_hermitian(fM, div_by_2=True):
    N = fM.shape[0]
    startFreq = 1-(N%2)
    if fM.ndim == 3:
        if startFreq:
            fM += n.roll(n.roll(n.roll(fM[::-1, ::-1, ::-1], 1, axis=0), 1, axis=1), 1, axis=2).conj()
        else:
            fM += fM[::-1, ::-1, ::-1].conj()
    elif fM.ndim == 2:
        if startFreq:
            fM += n.roll(n.roll(fM[::-1, ::-1], 1, axis=0), 1, axis=1).conj()
        else:
            fM += fM[::-1, ::-1].conj()
    else:
        assert False, "fourier.make_hermitian only support 2D or 3D transforms!"
    if div_by_2:
        fM *= 0.5
    return fM

def check_hermitian(fM):
    N = fM.shape[0]
    startFreq = 1-(N%2)
    if startFreq:
        E = fM - n.roll(n.roll(n.roll(fM[::-1, ::-1, ::-1], 1, axis=0), 1, axis=1), 1, axis=2).conj()
    else:
        E = fM - fM[::-1, ::-1, ::-1].conj()

    return n.linalg.norm(n.absolute(E))
