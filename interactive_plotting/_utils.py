#!/usr/bin/env python3

from functools import wraps
from collections import Iterable
from inspect import signature

import numpy as np
import warnings
import panel as pn


NO_SUBSAMPLE = (None, 'none')
SUBSAMPLING_STRATEGIES = ('datashade', 'decimate', 'sample_density', 'sample_unif')
ALL_SUBSAMPLING_STRATEGIES = NO_SUBSAMPLE + SUBSAMPLING_STRATEGIES

SUBSAMPLE_THRESH = 30_000
HOLOMAP_THRESH = 50
OBSM_SEP = ':'


def iterable(obj):
    '''
    Checks whether the object is iterable non-string.
    
    Params
    --------
    obj: Object
        Python object

    Returns
    --------
    is_iterable: Bool
        whether the object is not `str` and is
        instance of class `Iterable`
    '''

    return not isinstance(obj, str) and isinstance(obj, Iterable)


def istype(obj):
    '''
    Checks whether the object is of class `type`.

    Params
    --------
    obj: Union[Object, Tuple]
        Python object or a tuple

    Returns
    --------
    is_type: Bool
        `True` if the objects is instance of class `type` or
        all the element of the tuple is of class `type`
    '''

    return isinstance(obj, type) or (isinstance(obj, tuple) and all(map(lambda o: isinstance(o, type), obj)))

def is_numeric(obj):
    '''
    Params
    obj: Object
        Python object

    --------
    Returns
    is_numeric: Bool
        `True` if the object is numeric, else `False`
    --------
    '''
    return all(hasattr(obj, attr)
               for attr in ('__add__', '__sub__', '__mul__', '__truediv__', '__pow__'))


def is_categorical(obj):
    '''
    Is the object categorical?

    Params
    --------
    obj: Python object
        object that has attribute `'dtype'`,

    Returns
    --------
    is_categorical: Bool
        `True` if it's categorical else `False`
    '''

    return obj.dtype.name == 'category'


def minmax(component, is_sorted=False):
    '''
    Get the minimum and maximum value of an array.

    Params
    --------
    component: Union[np.ndarray, List, Tuple]
        1-D array
    is_sorted: Bool, optional (default: `False`)
        whether the component is already sorted,
        if `True`, min and max are the first and last
        elements respectively

    Returns
    --------
    min_max: Tuple[Float, Float]
        minimum and maximum values that are not NaN
    '''

    return (np.nanmin(component), np.nanmax(component)) if not is_sorted else (component[0], component[-1])


def skip_or_filter(adata, needles, haystack, where='', dtype=None,
                   skip=False, warn=True, ignore_after=None):
    '''
    Find all the needles in a given haystack.

    Params
    --------
    adata: anndata.AnnData
        anndata object
    needles: List[Str]
        keys to search for
    haystack: Iterable
        collection to search, e.g. `adata.obs.keys()`
    where: Str, optional, default (`''`)
        attribute of `anndata.AnnData` where to look, e. g. `'obsm'`
    dtype: Union[Callable, Type]
        expected datatype of the needles
    skip: Bool, optional (default: `False`)
        whether to skip the needles which do not have
        the expected `dtype`
    warn: Bool
        whether to issue a warning if `skip=True`
    ignore_after: Str, optional (default: `None`)
        token used for extracting the actual key name
        from the needle of form `KeyTokenIndex`, neeeded when
        `where='obsm'`, useful e.g. for extracting specific components

    Returns
    --------
    found_needles: List[Str]
        list of all the needles in haystack
        if `skip=False`, will throw a `RuntimeError`
        if the needle's type differs from `dtype`
    '''

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
        val = col.iloc[0]
        if n != nf:
            assert where == 'obsm', f'Indexing is only supported for `adata.obsm`, found {nf} in adata.`{where}`.'
            _, ix = n.split(ignore_after)
            assert nf == _, 'Unable to parse input.'
            val = val[int(ix)]

        msg = None
        is_tup = isinstance(dtype, tuple)
        if isinstance(dtype, type) or is_tup:
            if not isinstance(val, dtype):
                types = dtype.__name__ if not is_tup else f"Union[{', '.join(map(lambda d: d.__name__, dtype))}]"
                msg = f'Expected `{nf}` to be of type `{types}`, found  `{type(val).__name__}`.'
        elif callable(dtype):
            if not dtype(val):
                msg = f'`{nf}` did not pass the type checking of `{callable.__name__}`.'
        else:
            assert isinstance(dtype, str)
            if not dtype == col.dtype.name:
                msg = f'Expected `{nf}` to be of type `{dtype}`, found  `{col.dtype.name}`.'

        if msg is not None:
            if not skip:
                raise RuntimeError(msg)
            warnings.warn(msg + ' Skipping.')
            continue

        res.append(n)

    return res


def has_attributes(*args, **kwargs):
    '''
    Params
    --------
    *args: variable length arguments
        key to check in for 
    **kwargs: keyword arguments
        keys to check

    Returns
    wrapped: callable
        function, which does the checks at runtime
    --------
    '''

    def inner(fn):
        '''
        Params
        --------
        fn: callable
            function to wrap

        Returns
        --------
        wrapped: callable
            the wrapped function
        '''

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
    '''
    Wrap the widget inside a panel.

    Params
    --------
    fn: callable
        funtion that returns a plot, such as `scatter`
    
    Returns
    --------
    wrapper: callable
        function which return object of type `pn.panel`
    '''

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
    '''
    Wrap the widget in a column, having it's
    input in one row.

    Params
    --------
    fn: callable
        funtion that returns a plot, such as `dpt`
    
    Returns
    --------
    wrapped: callable 
        function which return object of type `pn.Column`
    '''

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


def get_data(adata, needle, ignore_after=OBSM_SEP, haystacks=['obs', 'obsm']):
    f'''
    Search for a needle in multiple haystacks.

    Params
    --------
    adata: anndata.AnnData
        anndata object
    needle: Str
        needle to search for
    ignore_after: Str, optional (default: `{OBSM_SEP}`)
        token used for extracting the actual key name
        from the needle of form `KeyTokenIndex`, neeeded
        when `'obsm' in haystacks`, useful e.g. for extracting specific components
    haystack: List[Str], optional (default: `['obs', 'obsm']`)
        attributes of `anndata.AnnData`

    Returns
    --------
    (result, is_categorical): Tuple[Object, Bool]
        the found object and whether it's categorical
    '''

    for haystack in haystacks:
        obj = getattr(adata, haystack)
        if ignore_after in needle and haystack == 'obsm':
            k, ix = needle.split(ignore_after)
            ix = int(ix)
        else:
            k, ix = needle, None

        if k in obj:
            res = obj[k]
            if ix is not None:
                res = res[:, ix]
            if res.shape != (adata.n_obs, ):
                msg = f'`{needle}` in `adata.{haystack}` has wrong shape of `{res.shape}`.'
                if haystack == 'obsm':
                    msg += f' Try using `{needle}{OBSM_SEP}ix`, `ix` in [0, {res.shape[-1]}).'
                raise ValueError(msg)

            return res, is_categorical(res)

    raise ValueError(f'Unable to find `{needle}` in `adata.{haystacks}`.')
