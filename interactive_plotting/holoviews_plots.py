#!/usr/bin/env python3

from .utils import *

from collections import Iterable, ChainMap, defaultdict, OrderedDict as odict

from pandas.api.types import is_categorical_dtype, is_string_dtype, infer_dtype
from scipy.sparse import issparse
from functools import partial
from bokeh.palettes import Viridis256
from datashader.colors import Sets1to3
from pandas.core.indexes.base import Index
from holoviews.operation.datashader import datashade, bundle_graph, shade, dynspread, rasterize, spread
from holoviews.operation import decimate
from bokeh.transform import linear_cmap
from bokeh.models import HoverTool

import scanpy as sc
import numpy as np
import pandas as pd
import networkx as nx
import holoviews as hv
import datashader as ds
import warnings


try:
    assert callable(sc.tl.dpt)
    dpt_fn = sc.tl.dpt
except AssertionError:
    from scanpy.api.tl import dpt as dpt_fn

#TODO: DRY

@wrap_as_panel
def scatter(adata, genes=None, basis=None, components=[1, 2], obs_keys=None,
            obsm_keys=None, use_raw=False, subsample='datashade', steps=40, keep_frac=None, lazy_loading=True,
            default_obsm_ixs=[0], sort=True, skip=True, seed=None, cols=None, size=4,
            perc=None, show_perc=True, cmap=None, plot_height=400, plot_width=400, save=None):
    '''
    Scatter plot for continuous observations.

    Params
    --------
    adata: anndata.Anndata
        anndata object
    genes: List[Str], optional (default: `None`)
        list of genes to add for visualization
        if `None`, use `adata.var_names`
    basis: Union[Str, List[Str]], optional (default: `None`)
        basis in `adata.obsm`, if `None`, get all available
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `basis`
        if it's of type `List[Int]`, all the basis have use the same components
    obs_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obs`
        if `None`, get all available
    obsm_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obsm`
        if `None`, get all available
    use_raw: Bool, optional (default: `False`)
        use `adata.raw` for gene expression levels
    subsample: Str, optional (default: `'datashade'`)
        subsampling strategy for large data
        possible values are `None, 'none', 'datashade', 'decimate', 'density', 'uniform'`
        using `subsample='datashade'` is preferred over other options since it does not subset
        when using `subsample='datashade'`, colorbar is not visible
        `'density'` and `'uniform'` use first element of `basis` for their computation
    steps: Union[Int, Tuple[Int, Int]], optional (default: `40`)
        step size when the embedding directions
        larger step size corresponds to higher density of points
    keep_frac: Float, optional (default: `adata.n_obs / 5`)
        number of observations to keep when `subsample='decimate'`
    lazy_loading: Bool, optional (default: `False`)
        only visualize when necessary
        for notebook sharing, consider using `lazy_loading=False`
    default_obsm_ixs: List[Int], optional (default: `[0]`)
        indices of 2-D elements in `adata.obsm` to add
        when `obsm_keys=None`
        by default adds only 1st column
    sort: Bool, optional (default: `True`)
        whether sort the `genes`, `obs_keys` and `obsm_keys`
        in ascending order
    skip: Bool, optional (default: `True`)
        skip all the keys not found in the corresponding collections
    seed: Int, optional (default: `None`)
        random seed, used when `subsample='decimate'``
    cols: Int, optional (default: `2`)
        number of columns when plotting basis
        if `None`, use togglebar
    size: Int, optional (default: `4`)
        size of the glyphs
        works only when `subsample != 'datashade'`
    perc: List[Float], optional (default: `None`)
        percentile for colors 
        useful when `lazy_loading = False`
        works only when `subsample != 'datashade'`
    show_perc: Bool, optional (default: `True`)
        show percentile slider when `lazy_loading = True`
        works only when `subsample != 'datashade'`
    cmap: List[Str], optional (default: `bokeh.palettes.Viridis256`)
        continuous colormap in hex format
    plot_height: Int, optional (default: `400`)
        height of the plot in pixels
    plot_width: Int, optional (default: `400`)
        width of the plot in pixels
    save: Union[os.PathLike, Str, NoneType], optional (default: `None`)
        path where to save the plot

    Returns
    --------
    plot: panel.panel
        holoviews plot wrapped in `panel.panel`
    '''

    def create_scatterplot(gene, perc_low, perc_high, *args, bs=None):
        ixs = np.where(basis == bs)[0][0]
        is_diffmap = bs == 'diffmap'

        if len(args) > 0:
            ixs = np.where(basis == bs)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (not is_diffmap)) % adata.obsm[f'X_{bs}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        if perc_low is not None and perc_high is not None:
            if perc_low > perc_high:
                perc_low, perc_high = perc_high, perc_low
            perc = [perc_low, perc_high]
        else:
            perc = None

        ad, _ = alazy[bs, tuple(comp)]
        ad_mraw = ad.raw if use_raw else ad

        # because diffmap has small range, it iterferes with
        # the legend created
        emb = ad.obsm[f'X_{bs}'][:, comp] * (1000 if is_diffmap else 1)
        comp += not is_diffmap  # naming consistence

        bsu = bs.upper()
        x = hv.Dimension('x', label=f'{bsu}{comp[0]}')
        y = hv.Dimension('y', label=f'{bsu}{comp[1]}')

        #if ignore_after is not None and ignore_after in gene:
        if gene in ad.obsm.keys():
            data = ad.obsm[gene][:, 0]
        elif gene in ad.obs.keys():
            data = ad.obs[gene].values
        elif gene in ad_mraw.var_names:
            data = ad_mraw.obs_vector(gene)
        else:
            gene, ix = gene.split(ignore_after)
            ix = int(ix)
            data = ad.obsm[gene][:, ix]

        data = np.array(data, dtype=np.float64)

        # we need to clip the data as well
        scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'gene': data},
                             kdims=[x, y], vdims='gene')

        return scatter.opts(cmap=cmap, color='gene',
                            colorbar=True,
                            colorbar_opts={'width': CBW},
                            size=size,
                            clim=minmax(data, perc=perc),
                            xlim=minmax(emb[:, 0]),
                            ylim=minmax(emb[:, 1]),
                            xlabel=f'{bsu}{comp[0]}',
                            ylabel=f'{bsu}{comp[1]}')

    def _create_scatterplot_nl(bs, gene, perc_low, perc_high, *args):
        # arg switching
        return create_scatterplot(gene, perc_low, perc_high, *args, bs=bs)

    if perc is None:
        perc = [None, None]
    assert len(perc) == 2, f'Percentile must be of length 2, found `{len(perc)}`.'
    if all(map(lambda p: p is not None, perc)):
        perc = sorted(perc)

    if keep_frac is None:
        keep_frac = 0.2

    if basis is None:
        basis = np.ravel(sorted(filter(len, map(BS_PAT.findall, adata.obsm.keys()))))
    elif isinstance(basis, str):
        basis = np.array([basis])
    elif not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        cb_kwargs = {'steps': steps}
    elif subsample == 'density':
        cb_kwargs = {'size': int(keep_frac * adata.n_obs), 'seed': seed}
    else:
        cb_kwargs = {}
    alazy = SamplingLazyDict(adata, subsample, callback_kwargs=cb_kwargs)
    adata_mraw = adata.raw if use_raw else adata  # maybe raw

    if obs_keys is None:
        obs_keys = skip_or_filter(adata, adata.obs.keys(), adata.obs.keys(), dtype=is_numeric,
                                  where='obs', skip=True, warn=False)
    else:
        if not iterable(obs_keys):
            obs_keys = [obs_keys]
        obs_keys = skip_or_filter(adata, obs_keys, adata.obs.keys(), dtype=is_numeric,
                                  where='obs', skip=skip)

    if obsm_keys is None:
        obsm_keys = get_all_obsm_keys(adata, default_obsm_ixs)
        ignore_after = OBSM_SEP
        obsm_keys = skip_or_filter(adata, obsm_keys, adata.obsm.keys(), where='obsm',
                                   dtype=is_numeric, skip=True, warn=False, ignore_after=ignore_after)
    else:
        if not iterable(obsm_keys):
            obsm_keys = [obsm_keys]

        ignore_after = OBSM_SEP if any((OBSM_SEP in obs_key for obs_key in obsm_keys)) else None
        obsm_keys = skip_or_filter(adata, obsm_keys, adata.obsm.keys(), where='obsm',
                                   dtype=is_numeric, skip=skip, ignore_after=ignore_after)

    if genes is None:
        genes = adata_mraw.var_names
    elif not iterable(genes):
        genes = [genes]
        genes = skip_or_filter(adata_mraw, genes, adata_mraw.var_names, where='adata.var_names', skip=skip)

    if isinstance(genes, Index):
        genes = list(genes)

    if sort:
        if any(genes[i] > genes[i + 1] for i in range(len(genes) - 1)):
            genes = sorted(genes)
        if any(obs_keys[i] > obs_keys[i + 1] for i in range(len(obs_keys) - 1)):
            obs_keys = sorted(obs_keys)
        if any(obsm_keys[i] > obsm_keys[i + 1] for i in range(len(obsm_keys) - 1)):
            obsm_keys = sorted(obsm_keys)

    conditions = obs_keys + obsm_keys + genes
    if len(conditions) == 0:
        warnings.warn(f'Nothing to plot, no conditions found.')
        return

    if not isinstance(components, np.ndarray):
        components = np.array(components)
    if components.ndim == 1:
        components = np.repeat(components[np.newaxis, :], len(basis), axis=0)

    assert components.ndim == 2, f'Only `2` dimensional components are supported, got `{components.ndim}`.'
    assert components.shape[-1] == 2, f'Components\' second dimension must be of size `2`, got `{components.shape[-1]}`.'
    if not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    assert components.shape[0] == len(basis), f'Expected #components == `{len(basis)}`, got `{components.shape[0]}`.'
    assert np.all(components >= 0), f'Currently, only positive indices are supported, found `{list(map(list, components))}`.'

    diffmap_ix = np.where(basis != 'diffmap')[0]
    components[diffmap_ix, :] -= 1

    for bs, comp in zip(basis, components):
        shape = adata.obsm[f'X_{bs}'].shape
        assert f'X_{bs}' in adata.obsm.keys(), f'`X_{bs}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{bs}` with shape `{shape}`.'

    if adata.n_obs > SUBSAMPLE_THRESH and subsample in NO_SUBSAMPLE:
        warnings.warn(f'Number of cells `{adata.n_obs}` > `{SUBSAMPLE_THRESH}`. Consider specifying `subsample={SUBSAMPLING_STRATEGIES}`.')

    if len(conditions) > HOLOMAP_THRESH and not lazy_loading:
        warnings.warn(f'Number of conditions `{len(conditions)}` > `{HOLOMAP_THRESH}`. Consider specifying `lazy_loading=True`.')

    if cmap is None:
        cmap = Viridis256

    kdims = [hv.Dimension('Basis', values=basis),
             hv.Dimension('Condition', values=conditions),
             hv.Dimension('Percentile (lower)', range=(0, 100), step=0.1, type=float, default=0 if perc[0] is None else perc[0]),
             hv.Dimension('Percentile (upper)', range=(0, 100), step=0.1, type=float, default=100 if perc[1] is None else perc[1])]

    cs = create_scatterplot
    _cs = _create_scatterplot_nl
    if not show_perc or subsample == 'datashade' or not lazy_loading:
        kdims = kdims[:2]
        cs = lambda gene, *args, **kwargs: create_scatterplot(gene, perc[0], perc[1], *args, **kwargs)
        _cs = lambda bs, gene, *args, **kwargs: _create_scatterplot_nl(bs, gene, perc[0], perc[1], *args, **kwargs)

    if not lazy_loading:
        dynmaps = [hv.HoloMap({(g, b):cs(g, bs=b) for g in conditions for b in basis}, kdims=kdims[::-1])]
    else:
        for bs, comp in zip(basis, components):
            kdims.append(hv.Dimension(f'{bs.upper()}[X]',
                                      type=int, default=1, step=1,
                                      range=(1, adata.obsm[f'X_{bs}'].shape[-1])))
            kdims.append(hv.Dimension(f'{bs.upper()}[Y]',
                                      type=int, default=2, step=1,
                                      range=(1, adata.obsm[f'X_{bs}'].shape[-1])))
        if cols is None:
            dynmaps = [hv.DynamicMap(_cs, kdims=kdims)]
        else:
            dynmaps = [hv.DynamicMap(partial(cs, bs=bs), kdims=kdims[1:]) for bs in basis]

    if subsample == 'datashade':
        dynmaps = [dynspread(datashade(d, aggregator=ds.mean('gene'), color_key='gene',
                                       cmap=cmap, streams=[hv.streams.RangeXY(transient=True)]),
                             threshold=0.8, max_px=5)
                   for d in dynmaps]
    elif subsample == 'decimate':
        dynmaps = [decimate(d, max_samples=int(adata.n_obs * keep_frac),
                            streams=[hv.streams.RangeXY(transient=True)], random_seed=seed) for d in dynmaps]

    dynmaps = [d.opts(framewise=True, axiswise=True, frame_height=plot_height, frame_width=plot_width) for d in dynmaps]

    if cols is None:
        plot = dynmaps[0].opts(title='', frame_height=plot_height, frame_width=plot_width)
    else:
        plot = hv.Layout(dynmaps).opts(title='', height=plot_height, width=plot_width).cols(cols)

    if save is not None:
        hv.renderer('bokeh').save(plot, save)

    return plot


@wrap_as_panel
def scatterc(adata, basis=None, components=[1, 2], obs_keys=None,
             obsm_keys=None, subsample='datashade', steps=40, keep_frac=None, hover=False, lazy_loading=True,
             default_obsm_ixs=[0], sort=True, skip=True, seed=None, legend_loc='top_right', cols=None, size=4,
             cmap=None, show_legend=True, plot_height=400, plot_width=400, save=None):
    '''
    Scatter plot for categorical observations.

    Params
    --------
    adata: anndata.Anndata
        anndata object
    basis: Union[Str, List[Str]], optional (default: `None`)
        basis in `adata.obsm`, if `None`, get all available
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `basis`
        if it's of type `List[Int]`, all the basis have use the same components
    obs_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obs`
        if `None`, get all available
    obsm_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obsm`
        if `None`, get all available
    subsample: Str, optional (default: `'datashade'`)
        subsampling strategy for large data
        possible values are `None, 'none', 'datashade', 'decimate', 'density', 'uniform'`
        using `subsample='datashade'` is preferred over other options since it does not subset
        when using `subsample='datashade'`, colorbar is not visible
        `'density'` and `'uniform'` use first element of `basis` for their computation
    steps: Union[Int, Tuple[Int, Int]], optional (default: `40`)
        step size when the embedding directions
        larger step size corresponds to higher density of points
    keep_frac: Float, optional (default: `adata.n_obs / 5`)
        number of observations to keep when `subsample='decimate'`
    hover: Union[Bool, Int], optional (default: `False`)
        whether to display cell index when hovering over a block
        if integer, it specifies the number of rows/columns (defualt: `10`)
    lazy_loading: Bool, optional (default: `False`)
        only visualize when necessary
        for notebook sharing, consider using `lazy_loading=False`
    sort: Bool, optional (default: `True`)
        whether sort the `genes`, `obs_keys` and `obsm_keys`
        in ascending order
    skip: Bool, optional (default: `True`)
        skip all the keys not found in the corresponding collections
    seed: Int, optional (default: `None`)
        random seed, used when `subsample='decimate'``
    legend_loc: Str, optional (default: `top_right`)
        position of the legend
    cols: Int, optional (default: `None`)
        number of columns when plotting basis
        if `None`, use togglebar
    size: Int, optional (default: `4`)
        size of the glyphs
        works only when `subsample!='datashade'`
    cmap: List[Str], optional (default: `datashader.colors.Sets1to3`)
        categorical colormap in hex format
    plot_height: Int, optional (default: `400`)
        height of the plot in pixels
    plot_width: Int, optional (default: `400`)
        width of the plot in pixels
    save: Union[os.PathLike, Str, NoneType], optional (default: `None`)
        path where to save the plot

    Returns
    --------
    plot: panel.panel
        holoviews plot wrapped in `panel.panel`
    '''

    def create_legend(condition, bs):
        # slightly hacky solution to get the correct initial limits
        xlim = lims['x'][bs]
        ylim = lims['y'][bs]

        return hv.NdOverlay({k: hv.Points([0, 0], label=str(k)).opts(size=0, color=v, xlim=xlim, ylim=ylim)  # alpha affects legend
                             for k, v in cmaps[condition].items()})

    def add_hover(subsampled, dynmaps=None, by_block=True):
        hovertool = HoverTool(tooltips=[('Cell Index', '@index')])
        hover_width, hover_height = (10, 10) if isinstance(hover, bool) else (hover, hover)

        if by_block:
            if dynmaps is None:
                dynmaps = subsampled

            return [s * hv.util.Dynamic(rasterize(d, width=hover_width, height=hover_height, streams=[hv.streams.RangeXY],
                                                  aggregator=ds.reductions.min('index')), operation=hv.QuadMesh)\
                                                          .opts(tools=[hovertool], axiswise=True, framewise=True, alpha=0, hover_alpha=0.25,
                                                                height=plot_height, width=plot_width)
                    for s, d in zip(subsampled, dynmaps)]

        return [s.opts(tools=[hovertool]) for s in subsampled]

    def create_scatterplot(cond, *args, bs=None):
        ixs = np.where(basis == bs)[0][0]
        is_diffmap = bs == 'diffmap'

        if len(args) > 0:
            ixs = np.where(basis == bs)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (not is_diffmap)) % adata.obsm[f'X_{bs}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        # subsample is uniform or density
        ad, ixs = alazy[bs, tuple(comp)]
        # because diffmap has small range, it interferes with the legend
        emb = ad.obsm[f'X_{bs}'][:, comp] * (1000 if is_diffmap else 1)
        comp += not is_diffmap  # naming consistence

        bsu = bs.upper()
        x = hv.Dimension('x', label=f'{bsu}{comp[0]}')
        y = hv.Dimension('y', label=f'{bsu}{comp[1]}')

        #if ignore_after is not None and ignore_after in gene:
        if cond in ad.obsm.keys():
            data = ad.obsm[cond][:, 0]
        elif cond in ad.obs.keys():
            data = ad.obs[cond]
        else:
            cond, ix = cond.split(ignore_after)
            ix = int(ix)
            data = ad.obsm[cond][:, ix]

        data = pd.Categorical(data).as_ordered()
        scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'cond': data, 'index': ixs},
                             kdims=[x, y], vdims=['cond', 'index']).sort('cond')

        return scatter.opts(color_index='cond', cmap=cmaps[cond],
                            show_legend=show_legend,
                            legend_position=legend_loc,
                            size=size,
                            xlim=minmax(emb[:, 0]),
                            ylim=minmax(emb[:, 1]),
                            xlabel=f'{bsu}{comp[0]}',
                            ylabel=f'{bsu}{comp[1]}')

    def _cs(bs, cond, *args):
        return create_scatterplot(cond, *args, bs=bs)

    if keep_frac is None:
        keep_frac = 0.2

    if basis is None:
        basis = np.ravel(sorted(filter(len, map(BS_PAT.findall, adata.obsm.keys()))))
    elif isinstance(basis, str):
        basis = np.array([basis])
    elif not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    if not isinstance(hover, bool):
        assert hover > 1, f'Expected `hover` to be `> 1` when being an integer, found: `{hover}`.'

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        cb_kwargs = {'steps': steps}
    elif subsample == 'density':
        cb_kwargs = {'size': int(keep_frac * adata.n_obs), 'seed': seed}
    else:
        cb_kwargs = {}
    alazy = SamplingLazyDict(adata, subsample, callback_kwargs=cb_kwargs)

    if obs_keys is None:
        obs_keys = skip_or_filter(adata, adata.obs.keys(), adata.obs.keys(),
                                  dtype='category', where='obs', skip=True, warn=False)
    else:
        if not iterable(obs_keys):
            obs_keys = [obs_keys]

        obs_keys = skip_or_filter(adata, obs_keys, adata.obs.keys(),
                                  dtype='category', where='obs', skip=skip)

    if obsm_keys is None:
        obsm_keys = get_all_obsm_keys(adata, default_obsm_ixs)
        obsm_keys = skip_or_filter(adata, obsm_keys, adata.obsm.keys(), where='obsm',
                                   dtype='category', skip=True, warn=False, ignore_after=OBSM_SEP)
    else:
        if not iterable(obsm_keys):
            obsm_keys = [obsm_keys]

        ignore_after = OBSM_SEP if any((OBSM_SEP in obs_key for obs_key in obsm_keys)) else None
        obsm_keys = skip_or_filter(adata, obsm_keys, adata.obsm.keys(), where='obsm',
                                   dtype='category', skip=skip, ignore_after=ignore_after)

    if sort:
        if any(obs_keys[i] > obs_keys[i + 1] for i in range(len(obs_keys) - 1)):
            obs_keys = sorted(obs_keys)
        if any(obsm_keys[i] > obsm_keys[i + 1] for i in range(len(obsm_keys) - 1)):
            obsm_keys = sorted(obsm_keys)

    conditions = obs_keys + obsm_keys

    if len(conditions) == 0:
        warnings.warn('Nothing to plot, no conditions found.')
        return

    if not isinstance(components, np.ndarray):
        components = np.array(components)
    if components.ndim == 1:
        components = np.repeat(components[np.newaxis, :], len(basis), axis=0)

    assert components.ndim == 2, f'Only `2` dimensional components are supported, got `{components.ndim}`.'
    assert components.shape[-1] == 2, f'Components\' second dimension must be of size `2`, got `{components.shape[-1]}`.'

    assert components.shape[0] == len(basis), f'Expected #components == `{len(basis)}`, got `{components.shape[0]}`.'
    assert np.all(components >= 0), f'Currently, only positive indices are supported, found `{list(map(list, components))}`.'

    diffmap_ix = np.where(basis != 'diffmap')[0]
    components[diffmap_ix, :] -= 1

    for bs, comp in zip(basis, components):
        shape = adata.obsm[f'X_{bs}'].shape
        assert f'X_{bs}' in adata.obsm.keys(), f'`X_{bs}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{bs}` with shape `{shape}`.'

    if adata.n_obs > SUBSAMPLE_THRESH and subsample in NO_SUBSAMPLE:
        warnings.warn(f'Number of cells `{adata.n_obs}` > `{SUBSAMPLE_THRESH}`. Consider specifying `subsample={SUBSAMPLING_STRATEGIES}`.')

    if len(conditions) > HOLOMAP_THRESH and not lazy_loading:
        warnings.warn(f'Number of  conditions `{len(conditions)}` > `{HOLOMAP_THRESH}`. Consider specifying `lazy_loading=True`.')

    if cmap is None:
        cmap = Sets1to3

    lims = dict(x=dict(), y=dict())
    for bs in basis:
        emb = adata.obsm[f'X_{bs}']
        is_diffmap = bs == 'diffmap'
        if is_diffmap:
            emb = (emb * 1000).copy()
        lims['x'][bs] = minmax(emb[:, 0 + is_diffmap])
        lims['y'][bs] = minmax(emb[:, 1 + is_diffmap])

    kdims = [hv.Dimension('Basis', values=basis),
             hv.Dimension('Condition', values=conditions)]

    cmaps = dict()
    for cond in conditions:
        color_key = f'{cond}_colors'
        # use the datashader default cmap since setting it doesn't work (for multiple conditions)
        cmaps[cond] = odict(zip(adata.obs[cond].cat.categories, # adata.uns.get(color_key, cmap)))
                                cmap if subsample == 'datashade' else adata.uns.get(color_key, cmap)))
    # this approach (for datashader) does not really work - the legend gets mixed up
    # cmap = dict(ChainMap(*[c.copy() for c in cmaps.values()]))
    # if len(cmap.keys()) != len([k for c in conditions for k in cmaps[c].keys()]):
    #     warnings.warn('Found same key across multiple conditions. The colormap/legend may not accurately display the colors.')

    if not lazy_loading:
        # have to wrap because of the *args
        dynmaps = [hv.HoloMap({(c, b):create_scatterplot(c, bs=b) for c in conditions for b in basis}, kdims=kdims[::-1])]
    else:
        for bs, comp in zip(basis, components):
            kdims.append(hv.Dimension(f'{bs.upper()}[X]',
                                      type=int, default=1, step=1,
                                      range=(1, adata.obsm[f'X_{bs}'].shape[-1])))
            kdims.append(hv.Dimension(f'{bs.upper()}[Y]',
                                      type=int, default=2, step=1,
                                      range=(1, adata.obsm[f'X_{bs}'].shape[-1])))

        if cols is None:
            dynmaps = [hv.DynamicMap(_cs, kdims=kdims)]
        else:
            dynmaps = [hv.DynamicMap(partial(create_scatterplot, bs=bs), kdims=kdims[1:]) for bs in basis]

    legend = None
    if subsample == 'datashade':
        subsampled = [dynspread(datashade(d, aggregator=ds.count_cat('cond'), color_key=cmap,
                                          streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                          min_alpha=255).opts(axiswise=True, framewise=True), threshold=0.8, max_px=5)
                      for d in dynmaps]
        dynmaps = add_hover(subsampled, dynmaps) if hover else subsampled

        if show_legend:
            warnings.warn('Automatic adjustment of axes is currently not working when '
                          '`show_legend=True` and `subsample=\'datashade\'`.')
            legend = hv.DynamicMap(create_legend, kdims=kdims[:2][::-1])
        elif hover:
            warnings.warn('Automatic adjustment of axes is currently not working when hovering is enabled.')

    elif subsample == 'decimate':
        subsampled = [decimate(d, max_samples=int(adata.n_obs * keep_frac),
                               streams=[hv.streams.RangeXY(transient=True)], random_seed=seed) for d in dynmaps]
        dynmaps = add_hover(subsampled, by_block=False) if hover else subsampled
    elif hover:
        dynmaps = add_hover(dynmaps, by_block=False)

    if cols is None:
        dynmap = dynmaps[0].opts(title='', frame_height=plot_height, frame_width=plot_width, axiswise=True, framewise=True)
        if legend is not None:
            dynmap = (dynmap * legend).opts(legend_position=legend_loc)
    else:
        if legend is not None:
            dynmaps = [(d * l).opts(legend_position=legend_loc)
                       for d, l in zip(dynmaps, legend.layout('bs') if lazy_loading and show_legend else [legend] * len(dynmaps))]

        dynmap = hv.Layout([d.opts(axiswise=True, framewise=True,
                                   frame_height=plot_height, frame_width=plot_width) for d in dynmaps]).cols(cols)

    plot = dynmap.cols(cols).opts(title='', height=plot_height, width=plot_width) if cols is not None else dynmap
    if save is not None:
        hv.renderer('bokeh').save(plot, save)

    return plot


@wrap_as_col
def dpt(adata, key, genes=None, basis=None, components=[1, 2],
        subsample='datashade', steps=40, use_raw=False, keep_frac=None,
        sort=True, skip=True, seed=None, show_legend=True, root_cell_all=False,
        root_cell_hl=True, root_cell_bbox=True, root_cell_size=None, root_cell_color='orange',
        legend_loc='top_right', size=4, perc=None, show_perc=True, cat_cmap=None, cont_cmap=None,
        plot_height=400, plot_width=400, *args, **kwargs):
    '''
    Scatter plot for categorical observations.

    Params
    --------
    adata: anndata.Anndata
        anndata object
    key: Str
        key in `adata.obs`, `adata.obsm` or `adata.var_names`
        to be visualized in top right plot
        can be categorical or continuous
    genes: List[Str], optional (default: `None`)
        list of genes to add for visualization
        if `None`, use `adata.var_names`
    basis: Union[Str, List[Str]], optional (default: `None`)
        basis in `adata.obsm`, if `None`, get all available
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `basis`
        if it's of type `List[Int]`, all the basis have use the same components
    use_raw: Bool, optional (default: `False`)
        use `adata.raw` for gene expression levels
    subsample: Str, optional (default: `'datashade'`)
        subsampling strategy for large data
        possible values are `None, 'none', 'datashade', 'decimate', 'density', 'uniform'`
        using `subsample='datashade'` is preferred over other options since it does not subset
        when using `subsample='datashade'`, colorbar is not visible
        `'density'` and `'uniform'` use first element of `basis` for their computation
    steps: Union[Int, Tuple[Int, Int]], optional (default: `40`)
        step size when the embedding directions
        larger step size corresponds to higher density of points
    keep_frac: Float, optional (default: `adata.n_obs / 5`)
        number of observations to keep when `subsample='decimate'`
    sort: Bool, optional (default: `True`)
        whether sort the `genes`, `obs_keys` and `obsm_keys`
        in ascending order
    skip: Bool, optional (default: `True`)
        skip all the keys not found in the corresponding collections
    seed: Int, optional (default: `None`)
        random seed, used when `subsample='decimate'``
    show_legend: Bool, optional (default: `True`)
        whether to show legend
    legend_loc: Str, optional (default: `top_right`)
        position of the legend
    cols: Int, optional (default: `None`)
        number of columns when plotting basis
        if `None`, use togglebar
    size: Int, optional (default: `4`)
        size of the glyphs
        works only when `subsample!='datashade'`
    perc: List[Float], optional (default: `None`)
        percentile for colors when `key` refers to continous observation
        works only when `subsample != 'datashade'`
    show_perc: Bool, optional (default: `True`)
        show percentile slider when `key` refers to continous observation
        works only when `subsample != 'datashade'`
    cat_cmap: List[Str], optional (default: `datashader.colors.Sets1to3`)
        categorical colormap in hex format
        used when `key` is categorical variable
    cont_cmap: List[Str], optional (default: `bokeh.palettes.Viridis256`)
        continuous colormap in hex format
        used when `key` is continuous variable
    root_cell_all: Bool, optional (default: `False`)
        show all root cells, even though they might not be in the embedding
        (e.g. when subsample='uniform' or 'density')
        otherwise only show in the embedding (based on the data of the 1st `basis`)
    root_cell_hl: Bool, optional (default: `True`)
        highlight the root cell
    root_cell_bbox: Bool, optional (default: `True`)
        show bounding box around the root cell
    root_cell_size: Int, optional (default `None`)
        size of the root cell, if `None`, it's `size * 2`
    root_cell_color: Str, optional (default: `red`)
        color of the root cell, can be in hex format
    plot_height: Int, optional (default: `400`)
        height of the plot in pixels
    plot_width: Int, optional (default: `400`)
        width of the plot in pixels
    *args, **kwargs:
        additional arguments for `sc.tl.dpt`

    Returns
    --------
    plot: panel.Column
        holoviews plot wrapped in `panel.Column`
    '''

    def create_scatterplot(root_cell, gene, bs, perc_low, perc_high, *args, typp='expr', ret_hl=False):
        ixs = np.where(basis == bs)[0][0]
        is_diffmap = bs == 'diffmap'

        if len(args) > 0:
            ixs = np.where(basis == bs)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (not is_diffmap)) % adata.obsm[f'X_{bs}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        ad, _ = alazy[bs, tuple(comp)]
        ad_mraw = ad.raw if use_raw else ad

        if perc_low is not None and perc_high is not None:
            if perc_low > perc_high:
                perc_low, perc_high = perc_high, perc_low
            perc = [perc_low, perc_high]
        else:
            perc = None

        # because diffmap has small range, it iterferes with
        # the legend created
        emb = ad.obsm[f'X_{bs}'][:, comp] * (1000 if is_diffmap else 1)
        comp += not is_diffmap  # naming consistence

        bsu = bs.upper()
        x = hv.Dimension('x', label=f'{bsu}{comp[0]}')
        y = hv.Dimension('y', label=f'{bsu}{comp[1]}')
        xmin, xmax = minmax(emb[:, 0])
        ymin, ymax = minmax(emb[:, 1])

        # adata is the original, ad may be subsampled
        mask = np.in1d(adata.obs_names, ad.obs_names)

        if typp == 'emb_discrete':
            scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'condition': data[mask]},
                                 kdims=[x, y], vdims='condition').sort('condition')

            scatter = scatter.opts(title=key,
                                   color='condition',
                                   xlim=(xmin, xmax),
                                   ylim=(ymin, ymax),
                                   size=size,
                                   xlabel=f'{bsu}{comp[0]}',
                                   ylabel=f'{bsu}{comp[1]}')

            if is_cat:
                # we're manually creating legend (for datashade)
                return scatter.opts(cmap=cat_cmap, show_legend=False)

            return scatter.opts(colorbar=True, colorbar_opts={'width': CBW},
                                cmap=cont_cmap, clim=minmax(data, perc=perc))


        if typp == 'root_cell_hl':
            # find the index of the root cell in maybe subsampled data
            rid = np.where(ad.obs_names == root_cell)[0]
            if not len(rid):
                return hv.Scatter([]).opts(axiswise=True, framewise=True)

            rid = rid[0]
            dx, dy = (xmax - xmin) / 25, (ymax - ymin) / 25
            rx, ry = emb[rid, 0], emb[rid, 1]

            root_cell_scatter = hv.Scatter({'x': emb[rid, 0], 'y': emb[rid, 1]}).opts(color=root_cell_color, size=root_cell_size)
            if root_cell_bbox:
                root_cell_scatter *= hv.Bounds((rx - dx, ry - dy, rx + dx, ry + dy)).opts(line_width=4, color=root_cell_color).opts(axiswise=True, framewise=True)

            return root_cell_scatter


        adata.uns['iroot'] = np.where(adata.obs_names == root_cell)[0][0]
        dpt_fn(adata, *args, **kwargs)

        pseudotime = adata.obs['dpt_pseudotime'].values
        pseudotime = pseudotime[mask]
        pseudotime[pseudotime == np.inf] = 1
        pseudotime[pseudotime == -np.inf] = 0

        if typp == 'emb':

            scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'pseudotime': pseudotime},
                                 kdims=[x, y], vdims='pseudotime')

            return scatter.opts(title='Pseudotime',
                                cmap=cont_cmap, color='pseudotime',
                                colorbar=True,
                                colorbar_opts={'width': CBW},
                                size=size,
                                clim=minmax(pseudotime, perc=perc),
                                xlim=(xmin, xmax),
                                ylim=(ymin, ymax),
                                xlabel=f'{bsu}{comp[0]}',
                                ylabel=f'{bsu}{comp[1]}')# if not ret_hl else root_cell_scatter

        if typp == 'expr':
            expr = ad_mraw.obs_vector(gene)

            x = hv.Dimension('x', label='pseudotime')
            y = hv.Dimension('y', label='expression')
            # data is in outer scope
            scatter_expr = hv.Scatter({'x': pseudotime, 'y': expr, 'condition': data[mask]},
                                      kdims=[x, y], vdims='condition')

            scatter_expr = scatter_expr.opts(title=key,
                                             color='condition',
                                             size=size,
                                             xlim=minmax(pseudotime),
                                             ylim=minmax(expr))
            if is_cat:
                # we're manually creating legend (for datashade)
                return scatter_expr.opts(cmap=cat_cmap, show_legend=False)

            return scatter_expr.opts(colorbar=True, colorbar_opts={'width': CBW},
                                     cmap=cont_cmap, clim=minmax(data, perc=perc))

        if typp == 'hist':
            return hv.Histogram(np.histogram(pseudotime, bins=20)).opts(xlabel='pseudotime',
                                                                        ylabel='frequency',
                                                                        color='#f2f2f2')

        raise RuntimeError(f'Unknown type `{typp}` for `create_scatterplot`.')

    # we copy beforehand
    if kwargs.pop('copy', False):
        adata = adata.copy()

    if keep_frac is None:
        keep_frac = 0.2

    if root_cell_size is None:
        root_cell_size = size * 2

    if basis is None:
        basis = np.ravel(sorted(filter(len, map(BS_PAT.findall, adata.obsm.keys()))))
    elif isinstance(basis, str):
        basis = np.array([basis])
    elif not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    if perc is None:
        perc = [None, None]
    assert len(perc) == 2, f'Percentile must be of length 2, found `{len(perc)}`.'
    if all(map(lambda p: p is not None, perc)):
        perc = sorted(perc)

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        cb_kwargs = {'steps': steps}
    elif subsample == 'density':
        cb_kwargs = {'size': int(keep_frac * adata.n_obs), 'seed': seed}
    else:
        cb_kwargs = {}
    alazy = SamplingLazyDict(adata, subsample, callback_kwargs=cb_kwargs)
    adata_mraw = adata.raw if use_raw else adata

    if genes is None:
        genes = adata_mraw.var_names
    elif not iterable(genes):
        genes = [genes]
        genes = skip_or_filter(adata_mraw, genes, adata_mraw.var_names, where='adata.var_names', skip=skip)

    if sort:
        if any(genes[i] > genes[i + 1] for i in range(len(genes) - 1)):
            genes = sorted(genes)

    if len(genes) == 0:
        warnings.warn(f'No genes found. Consider speciying `skip=False`.')
        return

    if not isinstance(components, np.ndarray):
        components = np.array(components)
    if components.ndim == 1:
        components = np.repeat(components[np.newaxis, :], len(basis), axis=0)

    assert components.ndim == 2, f'Only `2` dimensional components are supported, got `{components.ndim}`.'
    assert components.shape[-1] == 2, f'Components\' second dimension must be of size `2`, got `{components.shape[-1]}`.'

    if not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    assert components.shape[0] == len(basis), f'Expected #components == `{len(basis)}`, got `{components.shape[0]}`.'
    assert np.all(components >= 0), f'Currently, only positive indices are supported, found `{list(map(list, components))}`.'

    diffmap_ix = np.where(basis != 'diffmap')[0]
    components[diffmap_ix, :] -= 1

    for bs, comp in zip(basis, components):
        shape = adata.obsm[f'X_{bs}'].shape
        assert f'X_{bs}' in adata.obsm.keys(), f'`X_{bs}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{bs}` with shape `{shape}`.'

    if adata.n_obs > SUBSAMPLE_THRESH and subsample in NO_SUBSAMPLE:
        warnings.warn(f'Number of cells `{adata.n_obs}` > `{SUBSAMPLE_THRESH}`. Consider specifying `subsample={SUBSAMPLING_STRATEGIES}`.')

    if cat_cmap is None:
        cat_cmap = Sets1to3

    if cont_cmap is None:
        cont_cmap = Viridis256

    kdims = [hv.Dimension('Root cell', values=(adata if root_cell_all else alazy[basis[0], tuple(components[0])][0]).obs_names),
             hv.Dimension('Gene', values=genes),
             hv.Dimension('Basis', values=basis)]
    cs = lambda cell, gene, bs, *args, **kwargs: create_scatterplot(cell, gene, bs, perc[0], perc[1], *args, **kwargs)

    data, is_cat = get_data(adata, key)
    if is_cat:
        data = pd.Categorical(data)
        aggregator = ds.count_cat
        cmap = cat_cmap
        legend = hv.NdOverlay({c: hv.Points([0, 0], label=str(c)).opts(size=0, color=color)
                               for c, color in zip(data.categories, cat_cmap)})
    else:
        data = np.array(data, dtype=np.float64)
        aggregator = ds.mean
        cmap = cont_cmap
        legend = None
        if show_perc and subsample != 'datashade':
            kdims += [
                hv.Dimension('Percentile (lower)', range=(0, 100), step=0.1, type=float, default=0 if perc[0] is None else perc[0]),
                hv.Dimension('Percentile (upper)', range=(0, 100), step=0.1, type=float, default=100 if perc[1] is None else perc[1])
            ]
            cs = create_scatterplot

    emb = hv.DynamicMap(partial(cs, typp='emb'), kdims=kdims)
    if root_cell_hl:
        root_cell = hv.DynamicMap(partial(cs, typp='root_cell_hl'), kdims=kdims)
    emb_d = hv.DynamicMap(partial(cs, typp='emb_discrete'), kdims=kdims)
    expr = hv.DynamicMap(partial(cs, typp='expr'), kdims=kdims)
    hist = hv.DynamicMap(partial(cs, typp='hist'), kdims=kdims)

    if subsample == 'datashade':
        emb = dynspread(datashade(emb, aggregator=ds.mean('pseudotime'), cmap=cont_cmap,
                                  streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                  min_alpha=255),
                        threshold=0.8, max_px=5)
        emb_d = dynspread(datashade(emb_d, aggregator=aggregator('condition'), cmap=cmap,
                                    streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                    min_alpha=255),
                        threshold=0.8, max_px=5)
        expr = dynspread(datashade(expr, aggregator=aggregator('condition'), cmap=cmap,
                                   streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                   min_alpha=255),
                        threshold=0.8, max_px=5)
    elif subsample == 'decimate':
        emb, emb_d, expr = (decimate(d, max_samples=int(adata.n_obs * keep_frac)) for d in (emb, emb_d, expr))

    if root_cell_hl:
        emb *= root_cell  # emb * root_cell.opts(axiswise=True, framewise=True)

    emb = emb.opts(axiswise=False, framewise=True, frame_height=plot_height, frame_width=plot_width)
    expr = expr.opts(axiswise=True, framewise=True, frame_height=plot_height, frame_width=plot_width)
    emb_d = emb_d.opts(axiswise=True, framewise=True, frame_height=plot_height, frame_width=plot_width)
    hist = hist.opts(axiswise=True, framewise=True, frame_height=plot_height, frame_width=plot_width)

    if show_legend and legend is not None:
        emb_d = (emb_d * legend).opts(legend_position=legend_loc, show_legend=True)

    return ((emb + emb_d)  + (hist + expr).opts(axiswise=True, framewise=True)).cols(2)


@wrap_as_col
def graph(adata, key, basis=None, components=[1, 2], obs_keys=[], color_key=None, color_key_reduction=np.sum,
          ixs=None, top_n_edges=None, filter_edges=None, directed=True, bundle=False, bundle_kwargs={},
          subsample=None, layouts=None, layout_kwargs={}, force_paga_indices=False,
          degree_by=None, legend_loc='top_right', node_size=12, edge_width=2, arrowhead_length=None,
          perc=None, color_edges_by='weight', hover_selection='nodes',
          node_cmap=None, edge_cmap=None, plot_height=600, plot_width=600):
    '''
    Params
    --------

    adata: anndata.Anndata
        anndata object
    key: Str
        key in `adata.uns`, `adata.uns[\'paga\'] or adata.uns[\'neighbors\']` which
        represents the graph as an adjacency matrix (can be sparse)
        use `'paga'` to access PAGA connectivies graph or (prefix `'p:...'`)
        to access `adata.uns['paga'][...]`
        for `adata.uns['neighbors'][...]`, use prefix `'n:...'`
    basis: Union[Str, List[Str]], optional (default: `None`)
        basis in `adata.obsm`, if `None`, get all of them
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `basis`
        if it's of type `List[Int]`, all the basis have use the same components
    color_key: Str, optional (default: `None`)
        variable in `adata.obs` with which to color in each node
        or `'incoming'`, `'outgoing'` for coloring values based on weights
    color_key_reduction: Callable, optional (default: `np.sum`)
        a numpy function, such as `np.mean`, `np.max`, ... when
        `color_key` is `'incoming'` or `'outgoing'`
    obs_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obs`
        if `None`, get all available, only visible when `hover_selection='nodes'`
    ixs: List[Int], optional (default: `None`)
        list of indices of nodes of graph to visualize
        if `None`, visualize all
    top_n_edges: Union[Int, Tuple[Int, Bool, Str]], optional (default: `None`)
        only for directed graph
        maximum number of outgoing edges per node to keep based on decreasing weight
        if a tuple, the second element specifies whether it's ascending or not
        the third one whether whether to consider outgoing ('out') or ('in') incoming edges
    filter_edges: Tuple[Float, Float], optional (default: `None`)
        min and max threshold values for edge visualization
        nodes without edges will *NOT* be removed
    directed: Bool, optional (default: `True`)
        whether the graph is directed or not
    subsample: Str, optional (default: `None`)
        subsampling strategies for edges
        possible values are `None, \'none\', \'datashade\'`
    bundle: Bool, optional (default: `False`)
        whether to bundle edges together (can be computationally expensive)
    bundle_kwargs: Dict, optional (defaul: `{}`)
        kwargs for bundler, e.g. `iterations=1` (default `4`)
        for more options, see `hv.operation.datashader.bundle_graph`
    layouts: List[Str], optional (default: `None`)
        layout names to use when drawing graph, e.g. `'umap'` in `adata.obsm`
        or `'kamada_kawai'` from `nx.layouts`
        if `None`, use all available layouts
    layout_kwargs: Dict[Str, Dict], optional (default: `{}`)
        kwargs for a given layout
    force_paga_indices: Bool, optional (default: `False`)
        by default, when `key='paga'`, all indices are used
        regardless of what was specified
    degree_by: Str, optional (default: `None`)
        if `'weights'`, use edge weights when calculating the degree
        only visible when `hover_selection='nodes'`
    legend_loc: Str, optional (default: `'top_right'`)
        locations of the legend, if `None`, do not show legend
    node_size: Float, optional (default: `12`)
        size of the graph nodes
    edge_width: Float, optional (default: `2`)
        width of the graph edges
    arrowhead_length: Float, optional (default: `None`)
        length of the arrow when `directed=True`
    perc: List[Float], optional (default: `None`)
        percentile for edge colors
        *WARNING* this can remove nodes and will be fixed in the future
    color_edges_by: Str, optional (default: `weight`)
        whether to color edges, if `None` do not color edges
    hover_selection: Str, optional (default: `'nodes'`)
        whether to define hover over `'nodes'` or `'edges'`
        if `subsample == 'datashade'`, it is always `'nodes'`
    node_cmap: List[Str], optional (default: `datashader.colors.Sets1to3`)
        colormap in hex format for `color_key`
    edge_cmap: List[Str], optional (default: `bokeh.palettes.Viridis256`)
        continuous colormap in hex format for edges
    plot_height: Int, optional (default: `600`)
        height of the plot in pixels
    plot_width: Int, optional (default: `600`)
        width of the plot in pixels

    Returns
    --------
    plot: panel.Column
        `hv.DynamicMap` wrapped in `panel.Column` that displays the graph in various layouts
    '''

    def normalize(emb):
        # TODO: to this once
        # normalize because of arrows...
        emb = emb.copy()
        x_min, y_min = np.min(emb[:, 0]), np.min(emb[:, 1])
        emb[:, 0] = (emb[:, 0] - x_min) / (np.max(emb[:, 0]) - x_min)
        emb[:, 1] = (emb[:, 1] - y_min) / (np.max(emb[:, 1]) - y_min)
        return emb

    def create_graph(adata, data):
        if perc is not None:
            data = percentile(data, perc)
        create_using = nx.DiGraph if directed else nx.Graph
        g = (nx.from_scipy_sparse_matrix if issparse(data)  else nx.from_numpy_array)(data, create_using=create_using)

        if  filter_edges is not None:
            minn, maxx = filter_edges
            minn = minn if minn is not None else -np.inf
            maxx = maxx if maxx is not None else np.inf
            for e, attr in list(g.edges.items()):
                if attr['weight'] < minn or attr['weight'] > maxx:
                    g.remove_edge(*e)

        to_keep = None
        if top_n_edges is not None:
            if isinstance(top_n_edges, (tuple, list)):
                to_keep, ascending, group_by = top_n_edges
            else:
                to_keep, ascending, group_by = top_n_edges, False, 'out'

            source, target = zip(*g.edges)
            weights = [v['weight'] for v in g.edges.values()]
            tmp = pd.DataFrame({'out': source, 'in': target, 'w': weights})

            to_keep = set(map(tuple, tmp.groupby(group_by).apply(lambda g: g.sort_values('w', ascending=ascending).take(range(min(to_keep, len(g)))))[['out', 'in']].values))

            for e in list(g.edges):
                if e not in to_keep:
                    g.remove_edge(*e)

        if not len(g.nodes):
            raise RuntimeError('Empty graph.')

        if not len(g.edges):
            msg = 'No edges to visualize.'
            if filter_edges is not None:
                msg += f' Consider altering the edge filtering thresholds `{filter_edges}`.'
            if top_n_edges is not None:
                msg += f' Perhaps use more top edges than `{to_keep}`.'
            raise RuntimeError(msg)

        if hover_selection == 'nodes':
            if directed:
                nx.set_node_attributes(g, values=dict(g.in_degree(weight=degree_by)),
                                       name='indegree')
                nx.set_node_attributes(g, values=dict(g.out_degree(weight=degree_by)),
                                       name='outdegree')
                nx.set_node_attributes(g, values=nx.in_degree_centrality(g),
                                       name='indegree centrality')
                nx.set_node_attributes(g, values=nx.out_degree_centrality(g),
                                       name='outdegree centrality')
            else:
                nx.set_node_attributes(g, values=dict(g.degree(weight=degree_by)),
                                       name='degree')
                nx.set_node_attributes(g, values=nx.degree_centrality(g),
                                       name='centrality')

        if not is_paga:
            nx.set_node_attributes(g, values=dict(zip(g.nodes.keys(), adata.obs.index)),
                                   name='name')
            for key in list(obs_keys):
                nx.set_node_attributes(g, values=dict(zip(g.nodes.keys(), adata.obs[key])),
                                       name=key)
            if color_key is not None:
                # color_vals has been set beforehand
                nx.set_node_attributes(g, values=dict(zip(g.nodes.keys(), adata.obs[color_key] if color_key in adata.obs.keys() else color_vals)),
                                       name=color_key)

        else:
            nx.set_node_attributes(g, values=dict(zip(g.nodes.keys(), adata.obs[color_key].cat.categories)),
                                   name=color_key)

        return g

    def embed_graph(layout_key, graph):
        bs_key = f'X_{layout_key}'
        if bs_key in adata.obsm.keys():
            emb = adata_ss.obsm[bs_key][:, get_component[layout_key]]
            emb = normalize(emb)
            layout = dict(zip(graph.nodes.keys(), emb))
            l_kwargs = {}
        elif layout_key == 'paga':
            layout = dict(zip(graph.nodes.keys(), paga_pos))
            l_kwargs = {}
        elif layout_key in DEFAULT_LAYOUTS:
            layout = DEFAULT_LAYOUTS[layout_key]
            l_kwargs = layout_kwargs.get(layout_key, {})

        g = hv.Graph.from_networkx(graph, positions=layout, **l_kwargs)
        g = g.opts(inspection_policy='nodes' if subsample == 'datashade' else hover_selection,
                      tools=['hover', 'box_select'],
                      edge_color=hv.dim(color_edges_by) if color_edges_by is not None else None,
                      edge_line_width=edge_width * (hv.dim('weight') if is_paga else 1),
                      edge_cmap=edge_cmap,
                      node_color=color_key,
                      node_cmap=node_cmap,
                      directed=directed,
                      colorbar=True,
                      show_legend=legend_loc is not None
        )

        return g if arrowhead_length is None else g.opts(arrowhead_length=arrowhead_length)

    def get_nodes(layout_key):  # DRY DRY DRY
        nodes = bundled[layout_key].nodes
        bs_key = f'X_{layout_key}'

        if bs_key in adata.obsm.keys():
            emb = adata_ss.obsm[bs_key][:, get_component[layout_key]]
            emb = normalize(emb)
            xlim = minmax(emb[:, 0])
            ylim = minmax(emb[:, 1])
        elif layout_key == 'paga':
            xlim = minmax(paga_pos[:, 0])
            ylim = minmax(paga_pos[:, 1])
        else:
            xlim, ylim = bundled[layout_key].range('x'), bundled[layout_key].range('y')

        xlim, ylim = pad(*xlim), pad(*ylim)  # for datashade

        # remove axes for datashade
        return nodes.opts(xlim=xlim, ylim=ylim, xaxis=None, yaxis=None, show_legend=legend_loc is not None)

    assert subsample in (None, 'none', 'datashade'), \
        f'Invalid subsampling strategy `{subsample}`. Possible values are None, \'none\', \'datashade\'.`'

    if top_n_edges is not None:
        assert directed, f'`n_top_edges` works only on directed graphs.`'
        if isinstance(top_n_edges, (tuple, list)):
            assert len(top_n_edges) == 3, f'`top_n_edges` must be of length 3, found `{len(top_n_edges)}`.'
            assert isinstance(top_n_edges[0], int), f'`top_n_edges[0]` must be an int, found `{type(top_n_edges[0])}`.'
            assert isinstance(top_n_edges[1], bool), f'`top_n_edges[1]` must be a bool, found `{type(top_n_edges[1])}`.'
            assert top_n_edges[2] in ('in', 'out') , '`top_n_edges[2]` must be either \'in\' or \'out\'.'
        else:
            assert isinstance(top_n_edges, int), f'`top_n_edges` must be an int, found `{type(top_n_edges)}`.'

    if edge_cmap is None:
        edge_cmap = Viridis256
    if node_cmap is None:
        node_cmap = Sets1to3

    if color_key is not None:
        assert color_key in adata.obs or color_key in ('incoming', 'outgoing'), f'Color key `{color_key}` not found in `adata.obs` and is not \'incoming\' or \'outgoing\'.'

    if obs_keys is None:
        obs_keys = adata.obs.keys()
    else:
        for obs_key in obs_keys:
            assert obs_key in adata.obs.keys(), f'Key `{obs_key}` not found in `adata.obs`.'

    if key.startswith('p:') or key.startswith('n:'):
        which, key = key.split(':')
    elif key == 'paga':  # QOL
        which, key = 'p', 'connectivities'
    else:
        which = None

    paga_pos = None
    is_paga = False
    if which is None and key in adata.uns.keys():
        data = adata.uns[key]
    elif which == 'n' and key in adata.uns['neighbors'].keys():
        data = adata.uns['neighbors'][key]
    elif which == 'p' and key in adata.uns['paga'].keys():
        data = adata.uns['paga'][key]
        is_paga = True
        directed = False
        if 'pos' in adata.uns['paga'].keys():
            paga_pos = adata.uns['paga']['pos']
    else:
        raise ValueError(f'Key `{key}` not found in `adata.uns` or '
                         '`adata.uns[\'neighbors\']` or `adata.uns[\'paga\']`. '
                         'To visualize the graphs in `uns[\'neighbors\']` or uns[\'paga\'] '
                         'prefix the key with `n:` or `p:`, respectively (e.g. `n:connectivities`).')
    assert data.ndim == 2, f'Adjacency matrix must be dimension of `2`, found `{adata.ndim}`.'
    assert data.shape[0] == data.shape[1], 'Adjacency matrix is not square, found shape `{data.shape}`.'

    if ixs is None or (is_paga and not force_paga_indices):
        ixs = np.arange(data.shape[0])
    else:
        assert np.min(ixs) >= 0
        assert np.max(ixs) < adata.shape[0]

    data = data[ixs, :][:, ixs]
    adata_ss = adata[ixs, :] if not is_paga or (len(ixs) != data.shape[0] and force_paga_indices) else adata

    if layouts is None:
        layouts = list(DEFAULT_LAYOUTS.keys())
    if isinstance(layouts, str):
        layouts = [layouts]
    for l in layouts:
        assert l in DEFAULT_LAYOUTS.keys(), f'Unknown layout `{l}`. Available layouts are `{list(DEFAULT_LAYOUTS.keys())}`.'

    if np.min(data) < 0 and 'kamada_kawai' in layouts:
        warnings.warn('`kamada_kawai` layout required non-negative edges, removing it from the list of possible layouts.')
        layouts.remove('kamada_kawai')

    if basis is None:
        basis = np.ravel(sorted(filter(len, map(BS_PAT.findall, adata.obsm.keys()))))
    elif not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    if not isinstance(components, np.ndarray):
        components = np.array(components)
    if components.ndim == 1:
        components = np.repeat(components[np.newaxis, :], len(basis), axis=0)
    if len(basis):
        components[np.where(basis != 'diffmap')] -= 1

    if is_paga:
        g_name = adata.uns['paga']['groups']
        if color_key is None or color_key != g_name:
            warnings.warn(f'Color key `{color_key}` differs from PAGA\'s groups `{g_name}`, setting it to `{g_name}`.')
            color_key = g_name
        if len(basis):
            warnings.warn(f'Cannot plot PAGA in the basis `{basis}`, removing them from layouts.')
            basis, components = [], []

    for bs, comp in zip(basis, components):
        shape = adata.obsm[f'X_{bs}'].shape
        assert f'X_{bs}' in adata.obsm.keys(), f'`X_{bs}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{bs}` with shape `{shape}`.'

    if paga_pos is not None:
        basis = ['paga']
        components = [0, 1]
    get_component = dict(zip(basis, components))

    is_categorical = False
    if color_key is not None:
        node_cmap = adata_ss.uns[f'{color_key}_colors'] if f'{color_key}_colors' in adata_ss.uns else node_cmap
        if color_key in adata.obs:
            color_vals = adata_ss.obs[color_key]
            if is_categorical_dtype(color_vals) or is_string_dtype(adata.obs[color_key]):
                color_vals = color_vals.astype('category').cat.categories
                is_categorical = True
                node_cmap = odict(zip(color_vals, to_hex_palette(node_cmap)))
            else:
                color_vals = adata_ss.obs[color_key].values
        else:
            print(data.shape)
            color_vals = np.array(color_key_reduction(data, axis=int(color_key == 'outgoing'))).flatten()

        if not is_categorical:
            color_key_map = linear_cmap(field_name=color_key, palette=node_cmap,
                                       low=np.min(color_vals), high=np.max(color_vals))

    if not is_categorical:
        legend_loc = None

    layouts = np.append(basis, layouts)
    if len(layouts) == 0:
        warnings.warn('Nothing to plot, no layouts found.')
        return

    # because of the categories
    graph = create_graph(adata_ss, data=data)

    kdims = [hv.Dimension('Layout', values=layouts)]
    g = hv.DynamicMap(partial(embed_graph, graph=graph), kdims=kdims).opts(axiswise=True, framewise=True)  # necessary as well

    if subsample != 'datashade':
        for layout_key in layouts:
            bs_key = f'X_{layout_key}'
            if bs_key in adata.obsm.keys():
                emb = adata_ss.obsm[bs_key][:, get_component[layout_key]]
                emb = normalize(emb)
                xlim = minmax(emb[:, 0])
                ylim = minmax(emb[:, 1])
            elif layout_key == 'paga':
                xlim = minmax(paga_pos[:, 0])
                ylim = minmax(paga_pos[:, 1])
            else:
                xlim, ylim = g[layout_key].range('x'), g[layout_key].range('y')
            xlim, ylim = pad(*xlim), pad(*ylim)
            g[layout_key].opts(xlim=xlim, ylim=ylim)  # other layouts are not normalized

    bundled = bundle_graph(g, **bundle_kwargs, weight=None) if bundle else g.clone()
    nodes = hv.DynamicMap(get_nodes, kdims=kdims).opts(axiswise=True, framewise=True)  # needed for datashade

    if subsample == 'datashade':
        g = (datashade(bundled, normalization='linear', color_key=color_edges_by, min_alpha=128,
                       cmap='black' if color_edges_by is None else edge_cmap,
                       streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize]))
        res = (g * nodes).opts(height=plot_height, width=plot_width).opts(
            hv.opts.Nodes(size=node_size, tools=['hover'], cmap=node_cmap,
                          fill_color='orange' if color_key is None else color_key)
        )
    else:
        res = bundled.opts(height=plot_height, width=plot_width).opts(
            hv.opts.Graph(
                node_size=node_size,
                node_fill_color='orange' if color_key is None else color_key,
                node_nonselection_alpha=0.05,
                edge_nonselection_alpha=0.05,
                edge_cmap=edge_cmap,
                node_cmap=node_cmap
            )
        )
        if legend_loc is not None and color_key is not None:
            res *= hv.NdOverlay({k: hv.Points([0,0], label=str(k)).opts(size=0, color=v)
                                 for k, v in node_cmap.items()})

    if legend_loc is not None and color_key is not None:
        res = res.opts(legend_position=legend_loc)

    labels = hv.Labels(nodes, ['x', 'y'], color_key)

    return res.opts(hv.opts.Graph(xaxis=None, yaxis=None))
