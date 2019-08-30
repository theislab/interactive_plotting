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

import scanpy as sc
import numpy as np
import pandas as pd
import holoviews as hv
import datashader as ds


@wrap_as_panel
def scatter(adata, genes=None, bases=['umap', 'pca'], components=[1, 2], obsm_keys=[],
            obs_keys=[], skip=True, subsample='datashade', keep_frac=0.5, use_holomap=False,
            sort=True, seed=42, cmap=Viridis256, cols=2, plot_height=400, plot_width=400):
    '''
    Params
    --------

    Returns
    --------
    '''

    def create_scatterplot(gene, *args, basis=None):
        ixs = np.where(bases == basis)[0][0]
        if len(args) > 0:
            ixs = np.where(bases == basis)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (basis != 'diffmap')) % adata.obsm[f'X_{basis}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy
        emb = adata.obsm[f'X_{basis}'][:, comp]
        comp += basis != 'diffmap'  # naming consistence

        basisu = basis.upper()
        x = hv.Dimension('x', label=f'{basisu}{comp[0]}')
        y = hv.Dimension('y', label=f'{basisu}{comp[1]}')

        #if ignore_after is not None and ignore_after in gene:
        if gene in adata.obsm.keys():
            data = adata.obsm[gene][:, 0]
        elif gene in adata.obs.keys():
            data = adata.obs[gene].values
        elif gene in adata.var_names:
            data = adata.obs_vector(gene)
        else:
            gene, ix = gene.split(ignore_after)
            ix = int(ix)
            data = adata.obsm[gene][:, ix]

        scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'gene': data},
                             kdims=[x, y], vdims='gene')

        return scatter.opts(cmap=cmap, color='gene',
                            colorbar=True,
                            clim=minmax(data),
                            xlim=minmax(emb[:, 0]),
                            ylim=minmax(emb[:, 1]),
                            xlabel=f'{basisu}{comp[0]}',
                            ylabel=f'{basisu}{comp[1]}')

    def _cs(basis, gene, *args):
        return create_scatterplot(gene, *args, basis=basis)

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if not iterable(obs_keys):
        obs_keys = [obs_keys]
    obs_keys = skip_or_filter(adata, obs_keys, adata.obs.keys(), dtype=is_numeric,
                              where='obs', skip=skip)

    if not iterable(obsm_keys):
        obsm_keys = [obsm_keys]

    ignore_after = OBSM_SEP if any((OBSM_SEP in obs_key for obs_key in obsm_keys)) else None
    obsm_keys = skip_or_filter(adata, obsm_keys, adata.obsm.keys(), where='obsm',
                               dtype=is_numeric, skip=skip, ignore_after=ignore_after)

    if genes is None:
        genes = adata.var_names
    elif not iterable(genes):
        genes = [genes]
        genes = skip_or_filter(adata, genes, adata.var_names, where='adata.var_names', skip=skip)

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
        warnings.warn(f'No conditions found. Consider speciying `skip=False`.')
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

    if len(conditions) > HOLOMAP_THRESH and use_holomap:
        warnings.warn(f'Number of conditions `{len(conditions)}` > `{HOLOMAP_THRESH}`. Consider specifying `use_holomap=False`.')

    lims = dict(x=dict(), y=dict())
    for basis in bases:
        emb = adata.obsm[f'X_{basis}']
        lims['x'][basis] = minmax(emb[:, 0])
        lims['y'][basis] = minmax(emb[:, 1])

    kdims = [hv.Dimension('Basis', values=bases),
             hv.Dimension('Condition', values=conditions)]

    if use_holomap:
        dynmaps = [hv.HoloMap({(g, b):create_scatterplot(g, basis=b) for g in conditions for b in bases}, kdims=kdims[::-1])]
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

    if subsample == 'datashade':
        dynmaps = [dynspread(datashade(d, aggregator=ds.mean('gene'), color_key='gene',
                                       cmap=cmap, streams=[hv.streams.RangeXY(transient=True)])) for d in dynmaps]
    elif subsample == 'decimate':
        dynmaps = [decimate(d, max_samples=int(adata.n_obs * keep_frac),
                            streams=[hv.streams.RangeXY(transient=True)], random_seed=seed) for d in dynmaps]

    dynmaps = [d.opts(framewise=True, axiswise=True, height=plot_height, width=plot_width) for d in dynmaps]

    if cols is None:
        return dynmaps[0].opts(title='', height=plot_height, width=plot_width)

    return hv.Layout(dynmaps).opts(title='', height=plot_height, width=plot_width).cols(cols)


@wrap_as_panel
def scatterc(adata, bases=['umap', 'pca'], components=[1, 2], obsm_keys=[],
             obs_keys=[], skip=True, subsample='decimate', keep_frac=0.5, use_holomap=False,
             sort=True, seed=42, cmap=Sets1to3, cols=None, legend_loc='top_right', show_legend=True,
             plot_height=400, plot_width=400):
    '''
    Params
    --------

    Returns
    --------
    '''

    def create_legend(condition, basis):
        # slightly hacky solution to get the correct initial limits
        xlim = lims['x'][basis]
        ylim = lims['y'][basis]

        return hv.NdOverlay({k: hv.Points([0, 0], label=str(k)).opts(size=0, color=v, xlim=xlim, ylim=ylim)  # alpha affects legend
                             for k, v in cmaps[condition].items()})

    def create_scatterplot(cond, *args, basis=None):
        ixs = np.where(bases == basis)[0][0]
        if len(args) > 0:
            ixs = np.where(bases == basis)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (basis != 'diffmap')) % adata.obsm[f'X_{basis}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy
        emb = adata.obsm[f'X_{basis}'][:, comp]
        comp += basis != 'diffmap'  # naming consistence

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
                            xlim=minmax(emb[:, 0]),
                            ylim=minmax(emb[:, 1]),
                            xlabel=f'{basisu}{comp[0]}',
                            ylabel=f'{basisu}{comp[1]}')

    def _cs(basis, cond, *args):
        return create_scatterplot(cond, *args, basis=basis)

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if not iterable(obs_keys):
        obs_keys = [obs_keys]
    obs_keys = skip_or_filter(adata, obs_keys, adata.obs.keys(),
                              dtype='category', where='obs', skip=skip)

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
        warnings.warn(f'No conditions found. Consider speciying `skip=False`.')
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

    if len(conditions) > HOLOMAP_THRESH and use_holomap:
        warnings.warn(f'Number of  conditions `{len(conditions)}` > `{HOLOMAP_THRESH}`. Consider specifying `use_holomap=False`.')

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

    if use_holomap:
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
                                       streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize]).opts(axiswise=True, framewise=True), threshold=0.5, max_px=100) for d in dynmaps]
        if show_legend:
            warnings.warn('Automatic adjustment of axes is currently not working when '
                          '`show_legend=True` and `subsample=\'datashade\'`.')
            legend = hv.DynamicMap(create_legend, kdims=kdims[:2][::-1])
    elif subsample == 'decimate':
        dynmaps = [decimate(d, max_samples=int(adata.n_obs * keep_frac),
                            streams=[hv.streams.RangeXY(transient=True)], random_seed=seed) for d in dynmaps]

    if cols is None:
        dynmap = dynmaps[0].opts(title='', height=plot_height, width=plot_width, axiswise=True, framewise=True)
        if legend is not None:
            dynmap = (dynmap * legend).opts(legend_position=legend_loc)
    else:
        if legend is not None:
            dynmaps = [d * l for d, l in zip(dynmaps, legend.layout('Basis'))]

        dynmap = hv.Layout([d.opts(axiswise=True, framewise=True, legend_position=legend_loc,
                                   height=plot_height, width=plot_width) for d in dynmaps])

    return dynmap.cols(cols).opts(title='', height=plot_height, width=plot_width) if cols is not None else dynmap


@wrap_as_col
def dpt(adata, cluster_key, genes=None, bases=['diffmap'], use_holomap=False,
        cat_cmap=Sets1to3, show_legend=True, cont_cmap=Viridis256, legend_loc='top_right',
        components=[1, 2], keep_frac=0.5, subsample='datashade', skip=True,
        sort=True, plot_height=400, plot_width=400, *args, **kwargs):
    '''
    Params
    --------

    Returns
    --------
    '''

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    def create_scatterplot(root_cell, gene, basis, *args, typp='expr'):
        ixs = np.where(bases == basis)[0][0]
        if len(args) > 0:
            ixs = np.where(bases == basis)[0][0] * 2
            comp = (np.array([args[ixs], args[ixs + 1]]) - (basis != 'diffmap')) % adata.obsm[f'X_{basis}'].shape[-1]
        else:
            comp = np.array(components[ixs])  # need to make a copy

        emb = adata.obsm[f'X_{basis}'][:, comp]
        comp += basis != 'diffmap'  # naming consistence

        adata.uns['iroot'] = np.where(adata.obs_names == root_cell)[0][0]
        sc.tl.dpt(adata, *args, **kwargs)

        pseudotime = adata.obs['dpt_pseudotime'].values
        pseudotime[pseudotime == np.inf] = 1
        pseudotime[pseudotime == -np.inf] = 0

        basisu = basis.upper()
        x = hv.Dimension('x', label=f'{basisu}{comp[0]}')
        y = hv.Dimension('y', label=f'{basisu}{comp[1]}')

        if typp == 'emb_discrete':
            scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'condition': data},
                                 kdims=[x, y], vdims='condition')

            scatter_ex = scatter.opts(title=cluster_key,
                                      color='condition',
                                      xlim=minmax(emb[:, 0]),
                                      ylim=minmax(emb[:, 1]),
                                      xlabel=f'{basisu}{comp[0]}',
                                      ylabel=f'{basisu}{comp[1]}')

            if is_cat:
                # we're manually creating legend (for datashade)
                return scatter.opts(cmap=cat_cmap, show_legend=False)

            return scatter.opts(colorbar=True, colorbar_opts={'width': 10},
                                cmap=cont_cmap, clim=minmax(data))

        if typp == 'emb':
            scatter = hv.Scatter({'x': emb[:, 0], 'y': emb[:, 1], 'pseudotime': pseudotime},
                                 kdims=[x, y], vdims='pseudotime')

            return scatter.opts(title='Pseudotime',
                                cmap=cont_cmap, color='pseudotime',
                                colorbar=True,
                                colorbar_opts={'width': 10},
                                clim=minmax(pseudotime),
                                xlim=minmax(emb[:, 0]),
                                ylim=minmax(emb[:, 1]),
                                xlabel=f'{basisu}{comp[0]}',
                                ylabel=f'{basisu}{comp[1]}')

        if typp == 'expr':
            expr = adata.obs_vector(gene)

            x = hv.Dimension('x', label='pseudotime')
            y = hv.Dimension('y', label='expression')
            scatter_expr = hv.Scatter({'x': pseudotime, 'y': expr, 'condition': data},
                                      kdims=[x, y], vdims='condition')

            scatter_expr = scatter_expr.opts(title=cluster_key,
                                             color='condition',
                                             xlim=minmax(pseudotime),
                                             ylim=minmax(expr))
            if is_cat:
                # we're manually creating legend (for datashade)
                return scatter_expr.opts(cmap=cat_cmap, show_legend=False)

            return scatter_expr.opts(colorbar=True, colorbar_opts={'width': 10},
                                     cmap=cont_cmap, clim=minmax(data))

        if typp == 'hist':
            return hv.Histogram(np.histogram(pseudotime, bins=20)).opts(xlabel='pseudotime', ylabel='frequence', color='#f2f2f2')

        raise RuntimeError(f'Unknown type `{typp}` for create_plot.')

    if genes is None:
        genes = adata.var_names
    elif not iterable(genes):
        genes = [genes]
        genes = skip_or_filter(adata, genes, adata.var_names, where='adata.var_names', skip=skip)

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

    if len(genes) > HOLOMAP_THRESH and use_holomap:
        warnings.warn(f'Number of genes `{len(genes)}` > `{HOLOMAP_THRESH}`. Consider specifying `use_holomap=False`.')

    kdims = [hv.Dimension('Cell', values=adata.obs_names),
             hv.Dimension('Gene', values=genes),
             hv.Dimension('Basis', values=bases)]

    data, is_cat = get_data(adata, cluster_key)
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
    hist = hv.DynamicMap(partial(create_scatterplot, typp='hist'), kdims=kdims).opts(height=plot_height, width=plot_width, axiswise=True)

    if subsample == 'datashade':
        emb = dynspread(datashade(emb, aggregator=ds.mean('pseudotime'), cmap=cont_cmap,
                                  streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize]))
        emb_d = dynspread(datashade(emb_d, aggregator=aggregator('condition'), cmap=cmap,
                                    streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize]))
        expr = dynspread(datashade(expr, aggregator=aggregator('condition'), cmap=cmap,
                                   streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize]))
    elif subsample == 'decimate':
        emb, emb_d, expr = (decimate(d, max_samples=int(adata.n_obs * keep_frac)) for d in (emb, emb_d, expr))

    emb = emb.opts(axiswise=False, framewise=True, height=plot_height, width=plot_width)
    expr = expr.opts(axiswise=True, framewise=True, height=plot_height, width=plot_width)
    emb_d = emb_d.opts(axiswise=True, framewise=True, height=plot_height, width=plot_width)

    if show_legend and legend is not None:
        emb_d = (emb_d * legend).opts(legend_position=legend_loc, show_legend=True)

    return ((emb + emb_d) + (hist + expr).opts(axiswise=True, framewise=True, title='')).cols(2)
