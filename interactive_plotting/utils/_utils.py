#!/usr/bin/env python3

from functools import wraps
from collections import Iterable
from inspect import signature
from sklearn.neighbors import NearestNeighbors

import matplotlib.colors as colors
import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx
import panel as pn
import re
import itertools
import warnings


NO_SUBSAMPLE = (None, 'none')
SUBSAMPLING_STRATEGIES = ('datashade', 'decimate', 'density', 'uniform')
ALL_SUBSAMPLING_STRATEGIES = NO_SUBSAMPLE + SUBSAMPLING_STRATEGIES

SUBSAMPLE_THRESH = 30_000
HOLOMAP_THRESH = 50
OBSM_SEP = ':'

CBW = 10  # colorbar width
BS_PAT = re.compile('^X_(.+)')

# for graph
DEFAULT_LAYOUTS = {l.split('_layout')[0]:getattr(nx.layout, l)
                   for l in dir(nx.layout) if l.endswith('_layout')}
DEFAULT_LAYOUTS.pop('bipartite')
DEFAULT_LAYOUTS.pop('rescale')
DEFAULT_LAYOUTS.pop('spectral')


class SamplingLazyDict(dict):

    def __init__(self, adata, subsample, *args, callback_kwargs={}, **kwargs):
        super().__init__(*args, **kwargs)
        self.adata = adata
        self.callback_kwargs = callback_kwargs

        if subsample == 'uniform':
            self.callback = sample_unif
        elif subsample == 'density':
            self.callback = sample_density
        else:
            ixs = list(range(adata.n_obs))
            self.callback = lambda *args, **kwargs: (adata, ixs)

    def __getitem__(self, key):
        if key not in self:
            bs, comps = key
            rev_comps = comps[::-1]

            if (bs, rev_comps) in self.keys():
                res, ixs = self[bs, rev_comps]
            else:
                res, ixs = self.callback(self.adata, bs=bs, components=comps, **self.callback_kwargs)

            self[key] = res, ixs

            return res, ixs

        return super().__getitem__(key)


def to_hex_palette(palette, normalize=True):
    """
    Converts matplotlib color array to hex strings
    """
    if not isinstance(palette, np.ndarray):
        palette = np.array(palette)

    if isinstance(palette[0], str):
        assert all(map(colors.is_color_like, palette)), 'Not all strings are color like.'
        return palette

    if normalize:
        minn = np.min(palette)
        # normalize to [0, 1]
        palette = (palette - minn) / (np.max(palette) - minn)

    return [colors.to_hex(c) if colors.is_color_like(c) else c for c in palette]


def pad(minn, maxx, padding=0.05):
    if minn > maxx:
        maxx, minn = minn, maxx
    return minn - padding, maxx + padding


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


def minmax(component, perc=None, is_sorted=False):
    '''
    Get the minimum and maximum value of an array.

    Params
    --------
    component: Union[np.ndarray, List, Tuple]
        1-D array
    perc: Union[List[Float], Tuple[Float]]
        clip the values by the percentiles
    is_sorted: Bool, optional (default: `False`)
        whether the component is already sorted,
        if `True`, min and max are the first and last
        elements respectively

    Returns
    --------
    min_max: Tuple[Float, Float]
        minimum and maximum values that are not NaN
    '''
    if perc is not None:
        assert len(perc) == 2, 'Percentile must be of length 2.'
        component = np.clip(component, *np.percentile(component, sorted(perc)))

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

    needles_f = list(map(lambda n: n[:n.find(ignore_after)], needles)) if ignore_after is not None else needles
    res = []

    for n, nf in zip(needles, needles_f):
        if nf not in haystack:
            msg = f'`{nf}` not found in `adata.{where}.keys()`.'
            if not skip:
                assert False, msg
            if warn:
                warnings.warn(msg + ' Skipping.')
            continue

        col = getattr(adata, where)[nf]
        val = col[0] if isinstance(col, np.ndarray) else col.iloc[0]  # np.ndarray of pd.DataFrame
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
            if warn:
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
        attributes to check, keys will be interpreted
        as arguments in function signature to check
        and values are lists annotated as follows:
        `[<optional_dtype>, 'a:<attribute1>', '<key1>', '<key2>', 'a:<attribute2>', ...]`
        using type `None` will result in no type checking

    Returns
    --------
    wrapped: Callable
        function, which does the checks at runtime
    --------
    '''

    def inner(fn):
        '''
        Binds the arguments of the function and checks the types.

        Params
        --------
        fn: Callable
            function to wrap

        Returns
        --------
        wrapped: Callable
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
    fn: Callable
        funtion that returns a plot, such as `scatter`
    
    Returns
    --------
    wrapper: Callable
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
    fn: Callable
        funtion that returns a plot, such as `dpt`
    
    Returns
    --------
    wrapped: Callable 
        function which return object of type `pn.Column`
    '''

    def chunkify(l, n):
        for i in range(0, len(l), n):
            yield l[i: i + n]

    @wraps(fn)
    def inner(*args, **kwargs):
        reverse = kwargs.pop('reverse', True)
        res = fn(*args, **kwargs)
        if res is None:
            return None

        res = pn.panel(res)
        if reverse:
            res.reverse()

        widgets = list(map(lambda w: pn.Row(*w), filter(len, chunkify(res[0], 3))))
        return pn.Column(*(widgets + [res[1]]))

    return inner


def get_data(adata, needle, ignore_after=OBSM_SEP, haystacks=['obs', 'obsm', 'var_names']):
    f'''
    Search for a needle in multiple haystacks.

    Params
    --------
    adata: anndata.AnnData
        anndata object
    needle: Str
        needle to search for
    ignore_after: Str, optional (default: `'{OBSM_SEP}'`)
        token used for extracting the actual key name
        from the needle of form `KeyTokenIndex`, neeeded
        when `'obsm' in haystacks`, useful e.g. for extracting specific components
    haystack: List[Str], optional (default: `['obs', 'obsm', 'var_names']`)
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
            res = obj[k] if haystack != 'var_names' else adata.obs_vector(k)
            if ix is not None:
                assert res.ndim == 2, f'`adata.{haystack}[{k}]` must have a dimension of 2, found `{res.dim}`.'
                assert res.shape[-1] > ix, f'Index `{ix}` out of bounds for `adata.{haystack}[{k}]` of shape `{res.shape}`.'
                res = res[:, ix]
            if res.shape != (adata.n_obs, ):
                msg = f'`{needle}` in `adata.{haystack}` has wrong shape of `{res.shape}`.'
                if haystack == 'obsm':
                    msg += f' Try using `{needle}{OBSM_SEP}ix`, 0 <= `ix` < {res.shape[-1]}.'
                raise ValueError(msg)

            return res, is_categorical(res)

    raise ValueError(f'Unable to find `{needle}` in `adata.{haystacks}`.')


def get_all_obsm_keys(adata, ixs):
    if not isinstance(ixs, (tuple, list)):
        ixs = [ixs]

    assert all(map(lambda ix: ix >= 0, ixs)), f'All indices must be non-negative.'

    return list(itertools.chain.from_iterable((f'{key}{OBSM_SEP}{ix}'
                                          for key in adata.obsm.keys() if isinstance(adata.obsm[key], np.ndarray) and adata.obsm[key].ndim == 2 and adata.obsm[key].shape[-1] > ix)
                                          for ix in ixs))

# based on:
# https://github.com/velocyto-team/velocyto-notebooks/blob/master/python/DentateGyrus.ipynb
def sample_unif(adata, steps, bs='umap', components=[0, 1]):
    if not isinstance(steps, (tuple, list)):
        steps = (steps, steps)

    assert len(components)
    assert min(components) >= 0

    embedding = adata.obsm[f'X_{bs}'][:, components]

    grs = []
    for i in range(embedding.shape[1]):
        m, M = np.min(embedding[:, i]), np.max(embedding[:, i])
        m = m - 0.025 * np.abs(M - m)
        M = M + 0.025 * np.abs(M - m)
        gr = np.linspace(m, M, num=steps[i])
        grs.append(gr)

    meshes_tuple = np.meshgrid(*grs)
    gridpoints_coordinates = np.vstack([i.flat for i in meshes_tuple]).T

    nn = NearestNeighbors()
    nn.fit(embedding)
    dist, ixs = nn.kneighbors(gridpoints_coordinates, 1)

    min_dist = np.sqrt((meshes_tuple[0][0, 0] - meshes_tuple[0][0, 1]) ** 2 +
                             (meshes_tuple[1][0, 0] - meshes_tuple[1][1, 0]) ** 2) / 2

    ixs = ixs[dist < min_dist]
    ixs = np.unique(ixs)

    return adata[ixs].copy(), ixs


def sample_density(adata, size, bs='umap', seed=None, components=[0, 1]):
    if size >= adata.n_obs:
        return adata

    if components[0] == components[1]:
        tmp = pd.DataFrame(np.ones(adata.n_obs) / adata.n_obs, columns=['prob_density'])
    else:
        # should be unique, using it only once since we cache the results
        # we don't need to add the components
        key_added =  f'{bs}_density_ipl_tmp'
        remove_key = False  # we may be operating on original object, keep it clean
        if key_added not in adata.obs.keys():
            sc.tl.embedding_density(adata, bs, key_added=key_added)
            remove_key = True
        tmp = pd.DataFrame(np.exp(adata.obs[key_added]) / np.sum(np.exp(adata.obs[key_added])))
        tmp.rename(columns={key_added: 'prob_density'}, inplace=True)
        if remove_key:
            del adata.obs[key_added]

    state = np.random.RandomState(seed)
    ixs = sorted(state.choice(range(adata.n_obs), size=size, p=tmp['prob_density'], replace=False))

    return adata[ixs].copy(), ixs
