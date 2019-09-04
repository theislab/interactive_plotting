#!/usr/bin/env python3

from ._utils import *

from collections import Iterable
from collections import OrderedDict as odict

from functools import wraps, partial
from bokeh.palettes import Viridis256
from datashader.colors import Sets1to3
from pandas.core.indexes.base import Index
from holoviews.operation.datashader import datashade, shade, dynspread, rasterize, spread
from holoviews.operation import decimate

from holoviews import opts

import scanpy as sc
import numpy as np
import pandas as pd
import holoviews as hv
import datashader as ds

try:
    assert callable(sc.tl.dpt)
    dpt_fn = sc.tl.dpt
except AssertionError:
    from scanpy.api.tl import dpt as dpt_fn

#TODO: DRY

@wrap_as_panel
def scatter(adata, genes=None, bases=['umap', 'pca'], components=[1, 2], obs_keys=None,
            obsm_keys=None, use_raw=False, subsample='datashade', steps=40, keep_frac=None, lazy_loading=True,
            default_obsm_ixs=[0], sort=True, skip=True, seed=None, cols=2, size=4,
            perc=None, show_perc=True, cmap=None, plot_height=400, plot_width=400):
    '''
    Scatter plot for continuous observations.

    Params
    --------
    adata: anndata.Anndata
        anndata object
    genes: List[Str], optional (default: `None`)
        list of genes to add for visualization
        if `None`, use `adata.var_names`
    bases: Union[Str, List[Str]], optional (default: `['umap', 'pca']`)
        bases in `adata.obsm`
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `bases`
        if it's of type `List[Int]`, all the bases have use the same components
    obs_keys: List[Str], optional (default: `[]`)
        keys of categorical observations in `adata.obs`
    obsm_keys: List[Str], optional (default: `[]`)
        keys of categorical observations in `adata.obsm`
    use_raw: Bool, optional (default: `False`)
        use `adata.raw` for gene expression levels
    subsample: Str, optional (default: `'datashade'`)
        subsampling strategy for large data
        possible values are `None, 'none', 'datashade', 'decimate', 'density', 'uniform'`
        using `subsample='datashade'` is preferred over other options since it does not subset
        when using `subsample='datashade'`, colorbar is not visible
        `'density'` and `'uniform'` use first element of `bases` for their computation
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
        number of columns when plotting bases
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

    Returns
    --------
    plot: panel.panel
        holoviews plot wrapped in `panel.panel`
    '''

    def create_scatterplot(gene, perc_low, perc_high, *args, basis=None):
        ixs = np.where(bases == basis)[0][0]
        is_diffmap = basis == 'diffmap'

        if len(args) > 0:
            ixs = np.where(bases == basis)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (not is_diffmap)) % adata.obsm[f'X_{basis}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        if perc_low is not None and perc_high is not None:
            if perc_low > perc_high:
                perc_low, perc_high = perc_high, perc_low
            perc = [perc_low, perc_high]
        else:
            perc = None

        # because diffmap has small range, it iterferes with
        # the legend created
        emb = adata.obsm[f'X_{basis}'][:, comp] * (1000 if is_diffmap else 1)
        comp += not is_diffmap  # naming consistence

        basisu = basis.upper()
        x = hv.Dimension('x', label=f'{basisu}{comp[0]}')
        y = hv.Dimension('y', label=f'{basisu}{comp[1]}')

        #if ignore_after is not None and ignore_after in gene:
        if gene in adata.obsm.keys():
            data = adata.obsm[gene][:, 0]
        elif gene in adata.obs.keys():
            data = adata.obs[gene].values
        elif gene in adata_mraw.var_names:
            data = adata_mraw.obs_vector(gene)
        else:
            gene, ix = gene.split(ignore_after)
            ix = int(ix)
            data = adata.obsm[gene][:, ix]

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
                            xlabel=f'{basisu}{comp[0]}',
                            ylabel=f'{basisu}{comp[1]}')

    def _create_scatterplot_nl(basis, gene, perc_low, perc_high, *args):
        # arg switching
        return create_scatterplot(gene, perc_low, perc_high, *args, basis=basis)

    if perc is None:
        perc = [None, None]
    assert len(perc) == 2, f'Percentile must be of length 2, found `{len(perc)}`.'
    if all(map(lambda p: p is not None, perc)):
        perc = sorted(perc)

    if keep_frac is None:
        keep_frac = 0.2

    if isinstance(bases, str):
        bases = [bases]

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        adata = sample_unif(adata, steps, bases[0])
    elif subsample == 'density':
        adata = sample_density(adata, int(keep_frac * adata.n_obs), bases[0], seed=seed)

    # maybe raw
    # from this we'll be getting the expression
    adata_mraw = adata.raw if use_raw else adata

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
        warnings.warn(f'Nothing to plot.')
        return

    if not isinstance(components, np.ndarray):
        components = np.array(components)
    if components.ndim == 1:
        components = np.repeat(components[np.newaxis, :], len(bases), axis=0)

    assert components.ndim == 2, f'Only `2` dimensional components are supported, got `{components.ndim}`.'
    assert components.shape[-1] == 2, f'Components\' second dimension must be of size `2`, got `{components.shape[-1]}`.'
    if not isinstance(bases, np.ndarray):
        bases = np.array(bases)

    assert components.shape[0] == len(bases), f'Expected #components == `{len(bases)}`, got `{components.shape[0]}`.'
    assert np.all(components >= 0), f'Currently, only positive indices are supported, found `{list(map(list, components))}`.'

    diffmap_ix = np.where(bases != 'diffmap')[0]
    components[diffmap_ix, :] -= 1

    for basis, comp in zip(bases, components):
        shape = adata.obsm[f'X_{basis}'].shape
        assert f'X_{basis}' in adata.obsm.keys(), f'`X_{basis}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{basis}` with shape `{shape}`.'

    if adata.n_obs > SUBSAMPLE_THRESH and subsample in NO_SUBSAMPLE:
        warnings.warn(f'Number of cells `{adata.n_obs}` > `{SUBSAMPLE_THRESH}`. Consider specifying `subsample={SUBSAMPLING_STRATEGIES}`.')

    if len(conditions) > HOLOMAP_THRESH and not lazy_loading:
        warnings.warn(f'Number of conditions `{len(conditions)}` > `{HOLOMAP_THRESH}`. Consider specifying `lazy_loading=True`.')

    if cmap is None:
        cmap = Viridis256

    lims = dict(x=dict(), y=dict())
    for basis in bases:
        emb = adata.obsm[f'X_{basis}']
        lims['x'][basis] = minmax(emb[:, 0])
        lims['y'][basis] = minmax(emb[:, 1])

    kdims = [hv.Dimension('Basis', values=bases),
             hv.Dimension('Condition', values=conditions),
             hv.Dimension('Percentile (lower)', range=(0, 100), step=0.1, type=float, default=0 if perc[0] is None else perc[0]),
             hv.Dimension('Percentile (upper)', range=(0, 100), step=0.1, type=float, default=100 if perc[1] is None else perc[1])]

    cs = create_scatterplot
    _cs = _create_scatterplot_nl
    if not show_perc or subsample == 'datashade' or not lazy_loading:
        kdims = kdims[:2]
        cs = lambda gene, *args, **kwargs: create_scatterplot(gene, perc[0], perc[1], *args, **kwargs)
        _cs = lambda basis, gene, *args, **kwargs: _create_scatterplot_nl(basis, gene, perc[0], perc[1], *args, **kwargs)

    # TODO: test lazy loading
    if not lazy_loading:
        dynmaps = [hv.HoloMap({(g, b):cs(g, basis=b) for g in conditions for b in bases}, kdims=kdims[::-1])]
    else:
        for basis, comp in zip(bases, components):
            kdims.append(hv.Dimension(f'{basis.upper()}[X]',
                                      type=int, default=1, step=1,
                                      range=(1, adata.obsm[f'X_{basis}'].shape[-1])))
            kdims.append(hv.Dimension(f'{basis.upper()}[Y]',
                                      type=int, default=2, step=1,
                                      range=(1, adata.obsm[f'X_{basis}'].shape[-1])))
        if cols is None:
            dynmaps = [hv.DynamicMap(_cs, kdims=kdims)]
        else:
            dynmaps = [hv.DynamicMap(partial(cs, basis=basis), kdims=kdims[1:]) for basis in bases]

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
        return dynmaps[0].opts(title='', frame_height=plot_height, frame_width=plot_width)

    return hv.Layout(dynmaps).opts(title='', height=plot_height, width=plot_width).cols(cols)


@wrap_as_panel
def scatterc(adata, bases=['umap', 'pca'], components=[1, 2], obs_keys=None,
             obsm_keys=None, subsample='datashade', steps=40, keep_frac=None, lazy_loading=True,
             default_obsm_ixs=[0], sort=True, skip=True, seed=None, legend_loc='top_right', cols=2, size=4,
             cmap=None, show_legend=True, plot_height=400, plot_width=400):
    '''
    Scatter plot for categorical observations.

    Params
    --------
    adata: anndata.Anndata
        anndata object
    bases: List[Str], optional (default: `['umap', 'pca']`)
        bases in `adata.obsm`
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `bases`
        if it's of type `List[Int]`, all the bases have use the same components
    obs_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obs`
        if `None`, try searching
    obsm_keys: List[Str], optional (default: `None`)
        keys of categorical observations in `adata.obsm`
        if `None`, try searching
    subsample: Str, optional (default: `'datashade'`)
        subsampling strategy for large data
        possible values are `None, 'none', 'datashade', 'decimate', 'density', 'uniform'`
        using `subsample='datashade'` is preferred over other options since it does not subset
        when using `subsample='datashade'`, colorbar is not visible
        `'density'` and `'uniform'` use first element of `bases` for their computation
    steps: Union[Int, Tuple[Int, Int]], optional (default: `40`)
        step size when the embedding directions
        larger step size corresponds to higher density of points
    keep_frac: Float, optional (default: `adata.n_obs / 5`)
        number of observations to keep when `subsample='decimate'`
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
    cols: Int, optional (default: `2`)
        number of columns when plotting bases
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

    Returns
    --------
    plot: `panel.panel`
        holoviews plot wrapped in `panel.panel`
    '''

    def create_legend(condition, basis):
        # slightly hacky solution to get the correct initial limits
        xlim = lims['x'][basis]
        ylim = lims['y'][basis]

        return hv.NdOverlay({k: hv.Points([0, 0], label=str(k)).opts(size=0, color=v, xlim=xlim, ylim=ylim)  # alpha affects legend
                             for k, v in cmaps[condition].items()})

    def create_scatterplot(cond, *args, basis=None):
        ixs = np.where(bases == basis)[0][0]
        is_diffmap = basis == 'diffmap'

        if len(args) > 0:
            ixs = np.where(bases == basis)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (not is_diffmap)) % adata.obsm[f'X_{basis}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        # because diffmap has small range, it iterferes with
        # the legend created
        emb = adata.obsm[f'X_{basis}'][:, comp] * (1000 if is_diffmap else 1)
        comp += not is_diffmap  # naming consistence

        basisu = basis.upper()
        x = hv.Dimension('x', label=f'{basisu}{comp[0]}')
        y = hv.Dimension('y', label=f'{basisu}{comp[1]}')

        #if ignore_after is not None and ignore_after in gene:
        if cond in adata.obsm.keys():
            data = adata.obsm[cond][:, 0]
        elif cond in adata.obs.keys():
            data = adata.obs[cond]
        else:
            cond, ix = cond.split(ignore_after)
            ix = int(ix)
            data = adata.obsm[cond][:, ix]

        data = pd.Categorical(data).as_ordered()
        scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'cond': data},
                             kdims=[x, y], vdims='cond').sort('cond')

        return scatter.opts(color_index='cond', cmap=cmaps[cond],
                            show_legend=show_legend,
                            legend_position=legend_loc,
                            size=size,
                            xlim=minmax(emb[:, 0]),
                            ylim=minmax(emb[:, 1]),
                            xlabel=f'{basisu}{comp[0]}',
                            ylabel=f'{basisu}{comp[1]}')

    def _cs(basis, cond, *args):
        return create_scatterplot(cond, *args, basis=basis)

    if keep_frac is None:
        keep_frac = 0.2

    if isinstance(bases, str):
        bases = [bases]

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        adata = sample_unif(adata, steps, bases[0])
    elif subsample == 'density':
        adata = sample_density(adata, int(keep_frac * adata.n_obs), bases[0], seed=seed)

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
        warnings.warn('Nothing to plot.')
        return

    if not isinstance(components, np.ndarray):
        components = np.array(components)
    if components.ndim == 1:
        components = np.repeat(components[np.newaxis, :], len(bases), axis=0)

    assert components.ndim == 2, f'Only `2` dimensional components are supported, got `{components.ndim}`.'
    assert components.shape[-1] == 2, f'Components\' second dimension must be of size `2`, got `{components.shape[-1]}`.'

    if not isinstance(bases, np.ndarray):
        bases = np.array(bases)

    assert components.shape[0] == len(bases), f'Expected #components == `{len(bases)}`, got `{components.shape[0]}`.'
    assert np.all(components >= 0), f'Currently, only positive indices are supported, found `{list(map(list, components))}`.'

    diffmap_ix = np.where(bases != 'diffmap')[0]
    components[diffmap_ix, :] -= 1

    for basis, comp in zip(bases, components):
        shape = adata.obsm[f'X_{basis}'].shape
        assert f'X_{basis}' in adata.obsm.keys(), f'`X_{basis}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{basis}` with shape `{shape}`.'

    if adata.n_obs > SUBSAMPLE_THRESH and subsample in NO_SUBSAMPLE:
        warnings.warn(f'Number of cells `{adata.n_obs}` > `{SUBSAMPLE_THRESH}`. Consider specifying `subsample={SUBSAMPLING_STRATEGIES}`.')

    if len(conditions) > HOLOMAP_THRESH and not lazy_loading:
        warnings.warn(f'Number of  conditions `{len(conditions)}` > `{HOLOMAP_THRESH}`. Consider specifying `lazy_loading=True`.')

    if cmap is None:
        cmap = Sets1to3

    lims = dict(x=dict(), y=dict())
    for basis in bases:
        emb = adata.obsm[f'X_{basis}']
        lims['x'][basis] = minmax(emb[:, 0])
        lims['y'][basis] = minmax(emb[:, 1])

    kdims = [hv.Dimension('Basis', values=bases),
             hv.Dimension('Condition', values=conditions)]

    cmaps = dict()
    for cond in conditions:
        color_key = f'{cond}_colors'
        # use the datashader default cmap since setting it doesn't work
        cmaps[cond] = odict(zip(adata.obs[cond].cat.categories,
                                cmap if subsample == 'datashade' else adata.uns.get(color_key, cmap)))

    if not lazy_loading:
        dynmap = hv.HoloMap({(c, b):create_scatterplot(c, b) for c in conditions for b in bases}, kdims=kdims)
    else:
        for basis, comp in zip(bases, components):
            kdims.append(hv.Dimension(f'{basis.upper()}[X]',
                                      type=int, default=1, step=1,
                                      range=(1, adata.obsm[f'X_{basis}'].shape[-1])))
            kdims.append(hv.Dimension(f'{basis.upper()}[Y]',
                                      type=int, default=2, step=1,
                                      range=(1, adata.obsm[f'X_{basis}'].shape[-1])))

        if cols is None:
            dynmaps = [hv.DynamicMap(_cs, kdims=kdims)]
        else:
            dynmaps = [hv.DynamicMap(partial(create_scatterplot, basis=basis), kdims=kdims[1:]) for basis in bases]

    legend = None
    if subsample == 'datashade':
        dynmaps = [dynspread(datashade(d, aggregator=ds.count_cat('cond'), color_key=cmap,
                                       streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                       min_alpha=255).opts(axiswise=True, framewise=True), threshold=0.8, max_px=5)
                   for d in dynmaps]
        if show_legend:
            warnings.warn('Automatic adjustment of axes is currently not working when '
                          '`show_legend=True` and `subsample=\'datashade\'`.')
            legend = hv.DynamicMap(create_legend, kdims=kdims[:2][::-1])
    elif subsample == 'decimate':
        dynmaps = [decimate(d, max_samples=int(adata.n_obs * keep_frac),
                            streams=[hv.streams.RangeXY(transient=True)], random_seed=seed) for d in dynmaps]

    if cols is None:
        dynmap = dynmaps[0].opts(title='', frame_height=plot_height, frame_width=plot_width, axiswise=True, framewise=True)
        if legend is not None:
            dynmap = (dynmap * legend).opts(legend_position=legend_loc)
    else:
        if legend is not None:
            dynmaps = [(d * l).opts(legend_position=legend_loc)
                       for d, l in zip(dynmaps, legend.layout('Basis'))]

        dynmap = hv.Layout([d.opts(axiswise=True, framewise=True,
                                   frame_height=plot_height, frame_width=plot_width) for d in dynmaps])

    return dynmap.cols(cols).opts(title='', height=plot_height, width=plot_width) if cols is not None else dynmap


@wrap_as_col
def dpt(adata, key, genes=None, bases=['diffmap'], components=[1, 2],
        subsample='datashade', steps=40, use_raw=False, keep_frac=None,
        sort=True, skip=True, seed=None, show_legend=True,
        legend_loc='top_right', size=4, perc=None, cat_cmap=None, cont_cmap=None,
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
    bases: Union[Str, List[Str]], optional (default: `['umap', 'pca']`)
        bases in `adata.obsm`
    components: Union[List[Int], List[List[Int]]], optional (default: `[1, 2]`)
        components of specified `bases`
        if it's of type `List[Int]`, all the bases have use the same components
    use_raw: Bool, optional (default: `False`)
        use `adata.raw` for gene expression levels
    subsample: Str, optional (default: `'datashade'`)
        subsampling strategy for large data
        possible values are `None, 'none', 'datashade', 'decimate', 'density', 'uniform'`
        using `subsample='datashade'` is preferred over other options since it does not subset
        when using `subsample='datashade'`, colorbar is not visible
        `'density'` and `'uniform'` use first element of `bases` for their computation
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
    cols: Int, optional (default: `2`)
        number of columns when plotting bases
        if `None`, use togglebar
    size: Int, optional (default: `4`)
        size of the glyphs
        works only when `subsample!='datashade'`
    perc: List[Float], optional (default: `None`)
        percentile for colors
        works only when `subsample != 'datashade'`
    cat_cmap: List[Str], optional (default: `datashader.colors.Sets1to3`)
        categorical colormap in hex format
        used when `key` is categorical variable
    cont_cmap: List[Str], optional (default: `bokeh.palettes.Viridis256`)
        continuous colormap in hex format
        used when `key` is continuous variable
    plot_height: Int, optional (default: `400`)
        height of the plot in pixels
    plot_width: Int, optional (default: `400`)
        width of the plot in pixels
    *args, **kwargs:
        additional arguments for `sc.tl.dpt`

    Returns
    --------
    plot: `panel.Column`
        holoviews plot wrapped in `panel.Column`
    '''

    def create_scatterplot(root_cell, gene, basis, *args, typp='expr'):
        ixs = np.where(bases == basis)[0][0]
        is_diffmap = basis == 'diffmap'

        if len(args) > 0:
            ixs = np.where(bases == basis)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (not is_diffmap)) % adata.obsm[f'X_{basis}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        # because diffmap has small range, it iterferes with
        # the legend created
        emb = adata.obsm[f'X_{basis}'][:, comp] * (1000 if is_diffmap else 1)
        comp += not is_diffmap  # naming consistence

        basisu = basis.upper()
        x = hv.Dimension('x', label=f'{basisu}{comp[0]}')
        y = hv.Dimension('y', label=f'{basisu}{comp[1]}')

        if typp == 'emb_discrete':
            scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'condition': data},
                                 kdims=[x, y], vdims='condition').sort('condition')

            scatter = scatter.opts(title=key,
                                   color='condition',
                                   xlim=minmax(emb[:, 0]),
                                   ylim=minmax(emb[:, 1]),
                                   size=size,
                                   xlabel=f'{basisu}{comp[0]}',
                                   ylabel=f'{basisu}{comp[1]}')

            if is_cat:
                # we're manually creating legend (for datashade)
                return scatter.opts(cmap=cat_cmap, show_legend=False)

            return scatter.opts(colorbar=True, colorbar_opts={'width': CBW},
                                cmap=cont_cmap, clim=minmax(data, perc=perc))

        adata.uns['iroot'] = np.where(adata.obs_names == root_cell)[0][0]
        dpt_fn(adata, *args, **kwargs)

        pseudotime = adata.obs['dpt_pseudotime'].values
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
                                xlim=minmax(emb[:, 0]),
                                ylim=minmax(emb[:, 1]),
                                xlabel=f'{basisu}{comp[0]}',
                                ylabel=f'{basisu}{comp[1]}')

        if typp == 'expr':
            expr = adata_mraw.obs_vector(gene)

            x = hv.Dimension('x', label='pseudotime')
            y = hv.Dimension('y', label='expression')
            scatter_expr = hv.Scatter({'x': pseudotime, 'y': expr, 'condition': data},
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

    if keep_frac is None:
        keep_frac = 0.2

    if isinstance(bases, str):
        bases = [bases]

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        adata = sample_unif(adata, steps, bases[0])
    elif subsample == 'density':
        adata = sample_density(adata, int(keep_frac * adata.n_obs), bases[0], seed=seed)

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
        components = np.repeat(components[np.newaxis, :], len(bases), axis=0)

    assert components.ndim == 2, f'Only `2` dimensional components are supported, got `{components.ndim}`.'
    assert components.shape[-1] == 2, f'Components\' second dimension must be of size `2`, got `{components.shape[-1]}`.'

    if not isinstance(bases, np.ndarray):
        bases = np.array(bases)

    assert components.shape[0] == len(bases), f'Expected #components == `{len(bases)}`, got `{components.shape[0]}`.'
    assert np.all(components >= 0), f'Currently, only positive indices are supported, found `{list(map(list, components))}`.'

    diffmap_ix = np.where(bases != 'diffmap')[0]
    components[diffmap_ix, :] -= 1

    for basis, comp in zip(bases, components):
        shape = adata.obsm[f'X_{basis}'].shape
        assert f'X_{basis}' in adata.obsm.keys(), f'`X_{basis}` not found in `adata.obsm`'
        assert shape[-1] > np.max(comp), f'Requested invalid components `{list(comp)}` for basis `X_{basis}` with shape `{shape}`.'

    if adata.n_obs > SUBSAMPLE_THRESH and subsample in NO_SUBSAMPLE:
        warnings.warn(f'Number of cells `{adata.n_obs}` > `{SUBSAMPLE_THRESH}`. Consider specifying `subsample={SUBSAMPLING_STRATEGIES}`.')

    if cat_cmap is None:
        cat_cmap = Sets1to3

    if cont_cmap is None:
        cont_cmap = Viridis256

    kdims = [hv.Dimension('Cell', values=adata.obs_names),
             hv.Dimension('Gene', values=genes),
             hv.Dimension('Basis', values=bases)]

    data, is_cat = get_data(adata, key)
    if is_cat:
        data = pd.Categorical(data)
        aggregator = ds.count_cat
        cmap = cat_cmap
        legend = hv.NdOverlay({c: hv.Points([0, 0], label=str(c)).opts(size=0, color=color)
                               for c, color in zip(data.categories, cat_cmap)})
    else:
        aggregator = ds.mean
        cmap = cont_cmap
        legend = None

    emb = hv.DynamicMap(partial(create_scatterplot, typp='emb'), kdims=kdims)
    emb_d = hv.DynamicMap(partial(create_scatterplot, typp='emb_discrete'), kdims=kdims)
    expr = hv.DynamicMap(partial(create_scatterplot, typp='expr'), kdims=kdims)
    hist = hv.DynamicMap(partial(create_scatterplot, typp='hist'), kdims=kdims)

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

    emb = emb.opts(axiswise=False, framewise=True, frame_height=plot_height, frame_width=plot_width)
    expr = expr.opts(axiswise=True, framewise=True, frame_height=plot_height, frame_width=plot_width)
    emb_d = emb_d.opts(axiswise=True, framewise=True, frame_height=plot_height, frame_width=plot_width)
    hist = hist.opts(axiswise=True, framewise=True, frame_height=plot_height, frame_width=plot_width)

    if show_legend and legend is not None:
        emb_d = (emb_d * legend).opts(legend_position=legend_loc, show_legend=True)

    return ((emb + emb_d)  + (hist + expr).opts(axiswise=True, framewise=True)).cols(2)
