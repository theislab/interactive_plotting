#!/usr/bin/env python3

import numpy as np
import warnings
import panel as pn

from functools import wraps
from collections import Iterable


NO_SUBSAMPLE = (None, 'none')
SUBSAMPLING_STRATEGIES = ('datashade', 'decimate', 'sample_density', 'sample_unif')
ALL_SUBSAMPLING_STRATEGIES = NO_SUBSAMPLE + SUBSAMPLING_STRATEGIES

SUBSAMPLE_THRESH = 30_000
HOLOMAP_THRESH = 50
OBSM_SEP = ':'


def iterable(obj):
    return not isinstance(obj, str) and isinstance(obj, Iterable)


def istype(obj):
    return isinstance(obj, type) or (isinstance(obj, tuple) and isinstance(obj[0], type))


def is_categorical(obj):
    return obj.dtype.name == 'category'


def minmax(component, is_sorted=False):
    return (np.nanmin(component), np.nanmax(component)) if not is_sorted else (component[0], component[-1])


def skip_or_filter(adata, needles, haystack, where='', dtype=None,
                   skip=False, warn=True, ignore_after=None):
    needles_f = map(lambda n: n[:n.find(ignore_after)], needles) if ignore_after is not None else needles
    res = []

    for n, nf in zip(needles, needles_f):
        if nf not in haystack:
            msg = f'`{nf}` not found in `adata.{where}.keys()`.'
            if not skip:
                assert False, msg
            warnings.warn(msg + ' Skipping.')
            continue

        col = getattr(adata, where)[nf]
        val = col[0]
        if n != nf:
            assert where == 'obsm', f'Indexing is only supported for `adata.obsm`, found {nf} in adata.`{where}`.'
            _, ix = n.split(ignore_after)
            assert nf == _, 'Sanity check failed'
            val = val[int(ix)]

        msg = None
        if isinstance(dtype, type):
            if not isinstance(val, dtype):
                msg = f'Expected `{nf}` to be of type `{dtype.__name__}`, found  `{type(nf).__name__}`.'
        else:
            assert isinstance(dtype, str)
            if not dtype == col.dtype.name:
                msg = f'Expected `{nf}` to be of type `{dtype}`, found  `{col.dtype.name}`.'

        if msg is not None:
            if not skip:
                assert False, msg
            warnings.warn(msg + ' Skipping.')
            continue

        res.append(n)

    return res


def has_attributes(*args, **kwargs):

    def inner(fn):

        @wraps(fn)
        def inner2(*fargs, **fkwargs):
            bound = sig.bind(*fargs, **fkwargs)
            bound.apply_defaults()

            for k, v in kwargs.items():
                if isinstance(v, type) or istype(v):
                    assert isinstance(bound.arguments[k], v), f'Argument: `{k}` must be of type: `{v}`.'
                elif iterable(v):
                    if not iterable(v[0]):
                        v = [v]

                    for vals in v:
                        typp = None
                        if vals[0] is None or istype(vals[0]):
                            typp, *vals = vals
                        if not vals[0].startswith('a:'):
                            raise ValueError('The first element must be an attribute '
                                             f'annotated with: `a:`, found: `{vals[0]}`. '
                                             f'Consider using: `a:{vals[0]}`.')

                        obj = None
                        for val in vals:
                            if val.startswith('a:'):
                                obj = getattr(obj if obj is not None else bound.arguments[k], val[2:])
                            else:
                                assert obj is not None
                                obj = obj[val]
                        if typp is not None:
                            assert isinstance(obj, typp)
                else:
                    raise RuntimeError(f'Unable to decode invariant: `{k}={v}`.')

            return fn(*fargs, **fkwargs)

        sig = signature(fn)
        for param in tuple(kwargs.keys()) + args:
            if not param in sig.parameters.keys():
                raise ValueError(f'Parameter `{param}` not found in the signature.')

        return inner2

    return inner


def wrap_as_panel(fn):

    @wraps(fn)
    def inner(*args, **kwargs):
        reverse = kwargs.pop('reverse', True)
        res = fn(*args, **kwargs)
        if res is None:
            return None
        res = pn.panel(res)
        if reverse:
            res.reverse()
        return res

    return inner


def wrap_as_col(fn):

    @wraps(fn)
    def inner(*args, **kwargs):
        reverse = kwargs.pop('reverse', True)
        res = fn(*args, **kwargs)
        if res is None:
            return None

        res = pn.panel(res)
        if reverse:
            res.reverse()

        return pn.Column(pn.Row(*res[0]), res[1])

    return inner


def get_data(adata, key, haystacks=['obs', 'obsm']):
    for haystack in haystacks:
        obj = getattr(adata, haystack)
        if ':' in key and haystack == 'obsm':
            k, ix = key.split(OBSM_SEP)
            ix = int(ix)
        else:
            k, ix = key, None

        if k in obj:
            res = obj[k]
            if ix is not None:
                res = res[:, ix]
            if res.shape != (adata.n_obs, ):
                msg = f'`{key}` in `adata.{haystack}` has wrong shape of `{res.shape}`.'
                if haystack == 'obsm':
                    msg += f' Try using `{key}{OBSM_SEP}ix`, `ix` in [0, {res.shape[-1]}).'
                raise ValueError(msg)

            return res, is_categorical(res)

    raise ValueError(f'Unable to find `{key}` in `adata.{haystacks}`.')
