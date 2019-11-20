#!/usr/bin/env python3

from ..utils import *

from pandas.api.types import is_categorical_dtype, is_categorical, \
                             is_numeric_dtype, is_bool_dtype, \
                             is_datetime64_any_dtype, is_string_dtype, \
                             infer_dtype
from collections import OrderedDict as odict
from datashader.colors import *
from bokeh.palettes import Viridis256
from holoviews.streams import Selection1D
from holoviews.operation import decimate
from holoviews.operation.datashader import datashade, dynspread, rasterize, spread

import numpy as np
import pandas as pd
import datashader as ds
import holoviews as hv


def groupby(adata, key, genes=None, aggregators={}, callback=lambda _: None, agg_sep='_'):

    def mode(df):
        return df.mode()[0]

    if genes is None:
        genes = adata.var_names[:10]
    expr = pd.DataFrame(adata[:, genes].X, index=adata.obs.index, columns=genes)
    expr_raw = None
    if adata.raw is not None:
        expr_raw = pd.DataFrame(adata.raw[:, genes].X, index=adata.obs.index,
                                columns=[f'{gene}_gene_raw' for gene in genes])

    df = adata.obs
    for d in [expr] + ([] if expr_raw is None else [expr_raw]):
        df = adata.obs.merge(d, left_index=True, right_index=True)
    groups = df.groupby(key)

    agg_fns = {}
    for column in filter(lambda c: c != key, df.columns):
        col = df[column]
        agg = aggregators.get(column, None) or callback(col)
        if agg is not None:
            agg_fns[column] = agg
            continue
        if is_categorical_dtype(col):
            agg = (mode, )
        elif is_bool_dtype(col):
            agg = (mode, )
        elif is_datetime64_any_dtype(col):
            agg = (mode, 'min', 'max')
        elif is_string_dtype(col):
            agg = (mode, )
        elif is_numeric_dtype(col):
            agg = ('mean', 'max', 'min')
        else:
            raise RuntimeError(f'Inferred type `{infer_dtype(col)}` of column `{column}` is not supported.')

        agg_fns[column] = agg

    res = groups.agg(agg_fns)
    res.columns = list(map(lambda c: agg_sep.join(c), res.columns))

    return res


def heatmap(adata, genes, group_key='louvain', show_scatterplot=False):
    '''
    Params
    -------
    adata: anndata.AnnData
        adata object
    genes: List[Str]
        genes in `adata.var_names`
    group_key: Str
        key in adata.obs, must be categorical
    show_scatterplot: Bool, optional (default: `False`)
        whether to show gene expression
        in pseudotime for selected gene - TODO: make more general

    Returns
    -------
    '''

    def heatmap_hl(original, index):
        nonlocal gene_order
        nonlocal group_order

        if not index:
            return original

        x, y, z = [original.iloc[index][key] for key in ['x', 'y', 'z']]

        res = {(k1, k2):v for k1, k2, v in zip(x, y, z)}
        x = sorted(set(x), key=lambda g: gene_order[g])
        y = sorted(set(y), key=lambda g: group_order[g])

        return hv.HeatMap({'x': x, 'y': y, 'z': [[res[k1, k2] for k1 in x] for k2 in y]},
                          kdims=[('x', 'Gene'), ('y', 'Group')], vdims=[('z', 'Expression')])

    genes = sorted(genes)[::-1]
    groups = sorted(list(adata.obs[group_key].cat.categories))
    # groups = np.random.permutation(groups)
    group_order = dict(zip(groups, range(len(groups))))
    gene_order = dict(zip(genes, range(len(genes))))

    ad = adata[np.in1d(adata.obs['louvain'], groups)][:, genes]
    df = pd.DataFrame(ad.X, columns=genes)
    df['group'] = list(map(str, ad.obs['louvain']))
    val = df.groupby('group').max().values

    x = hv.Dimension('x', label='Gene')
    y = hv.Dimension('y', label='Group')
    z = hv.Dimension('z', label='Expression')

    heatmap = hv.HeatMap({'x': np.array(genes), 'y': np.array(groups), 'z': val},
                         kdims=[('x', 'Gene'), ('y', 'Group')], vdims=[('z', 'Expression')]).opts(tools=['box_select', 'hover'], xrotation=90)
    sel = Selection1D(source=heatmap)
    mean_sel = hv.DynamicMap(lambda index: heatmap_hl(heatmap, index),
                             kdims=[], streams=[sel])
    tap = hv.streams.Tap(source=mean_sel, x=genes[0], y=adata.obs[group_key].values[0])

    if show_scatterplot:
        df['pseudotime'] = list(adata.obs['dpt_pseudotime'])

    if '{}_colors'.format(group_key) in adata.uns.keys():
        color_dict = dict(zip(adata.obs[group_key].cat.categories, adata.uns[f'{group_key}_colors']))
    else:
        color_dict = dict(zip(adata.obs[group_key].cat.categories, ['black'] * len(df)))

    df['color'] = [color_dict[c] for c in df['group'].values]

    df = df.melt(value_vars=genes, id_vars=(['pseudotime'] if 'pseudotime' in df else []) + ['group', 'color'],
            var_name='gene', value_name='expression')
    dataset = hv.Dataset(df, vdims=['expression', 'color'])
    dmap = hv.DynamicMap(lambda x, y: scatter(adata, x, 'louvain', subsample=None),
                         #hv.Scatter(dataset.select(gene=x),
                         #                       kdims='pseudotime',
                         #                       label='{} @ {}[{}]'.format(x, by, y)).opts(color='color', axiswise=True, framewise=True),
                         streams=[tap])

    # TODO: pass as argument
    subsample = 'none'
    vdims = 'z'
    categorical = True
    show_legend = False
    cmap = Sets1to3
    cmap = odict(zip(adata.obs[group_key].cat.categories, adata.uns.get(f'{group_key}_colors', cmap)))
    keep_frac = adata.n_obs * 0.5
    seed = 42

    if subsample == 'datashade':
        aggregator = None
        if vdims is not None:
            aggregator = ds.count_cat(vdims) if categorical else ds.mean(vdims)
        dmap = dynspread(datashade(dmap, aggregator=aggregator,
                                      color_key=cmap, cmap=cmap,
                                      streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                      min_alpha=255).opts(axiswise=True, framewise=True), threshold=0.8, max_px=5)
        if show_legend and categorical:
            legend = hv.NdOverlay({k: hv.Points([0, 0], label=str(k)).opts(size=0, color=v)
                                   for k, v in cmap.items()})

    elif subsample == 'decimate':
        dmap = decimate(dmap, max_samples=int(adata.n_obs * keep_frac),
                           streams=[hv.streams.RangeXY(transient=True)], random_seed=seed)

    dmap = dmap.opts(axiswise=True, framewise=True)

    if show_scatterplot:
        return (heatmap.opts(frame_width=600, colorbar=True) + mean_sel.opts(colorbar=True) + dmap).cols(1)

    return (heatmap.opts(frame_width=600, colorbar=True) + mean_sel.opts(colorbar=True)).cols(1)


def pad(minn, maxx, padding=0.1):
    if minn > maxx:
        maxx, minn = minn, maxx
    delta = maxx - minn

    return minn - (delta * padding), maxx + (delta * padding)

def minmax(component, perc=None, is_sorted=False):
    if perc is not None:
        assert len(perc) == 2, 'Percentile must be of length 2.'
        component = np.clip(component, *np.percentile(component, sorted(perc)))

    return (np.nanmin(component), np.nanmax(component)) if not is_sorted else (component[0], component[-1])


def scatter(adata, gene, group, pseudotime_key='dpt_pseudotime', subsample='datashade', steps=40, keep_frac=None,
            seed=None, legend_loc='top_right', cols=None, size=4, use_raw=False,
            cmap=None, show_legend=True, plot_height=400, plot_width=400):

    adata_mraw = adata.raw if use_raw else adata
    gene_ix = np.where(adata_mraw.var_names == gene)[0][0]
    expression = np.squeeze(np.array(adata_mraw.X[:, gene_ix]))

    ixs = np.argsort(adata.obs[pseudotime_key])

    return _scatter(adata, x=adata.obs[pseudotime_key].values[ixs],
                    y=expression[ixs],
                    condition=adata.obs[group][ixs],
                    by=group,
                    xlabel=pseudotime_key,
                    ylabel='expression',
                    title=gene,
                    subsample=subsample, steps=steps, keep_frac=keep_frac, seed=seed, legend_loc=legend_loc,
                    size=size, cmap=cmap, show_legend=show_legend, plot_height=plot_height, plot_width=plot_width)


def _scatter(adata, x, y, condition=None, by=None, subsample='datashade', steps=40, keep_frac=None,
            seed=None, legend_loc='top_right', size=4, xlabel=None, ylabel=None, title=None,
            cmap=None, show_legend=True, plot_height=400, plot_width=400):
    '''
    Scatter plot for categorical observations. TODO: update docs, maybe not pass adata

    Params
    --------
    adata: anndata.Anndata
        anndata object
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
    cols: Int, optional (default: `None`)
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
    plot: panel.panel
        holoviews plot wrapped in `panel.panel`
    '''

    if keep_frac is None:
        keep_frac = 0.2

    assert keep_frac >= 0 and keep_frac <= 1, f'`keep_perc` must be in interval `[0, 1]`, got `{keep_frac}`.'
    # assert subsample in ALL_SUBSAMPLING_STRATEGIES, f'Invalid subsampling strategy `{subsample}`. Possible values are `{ALL_SUBSAMPLING_STRATEGIES}`.'

    if subsample == 'uniform':
        cb_kwargs = {'steps': steps}
    elif subsample == 'density':
        cb_kwargs = {'size': int(keep_frac * adata.n_obs), 'seed': seed}
    else:
        cb_kwargs = {}

    categorical = False
    if condition is None:
        cmap = 'black'
    elif is_categorical(condition):
        categorical = True
        cmap = Sets1to3 if cmap is None else cmap
        cmap = odict(zip(condition.cat.categories, adata.uns.get(f'{by}_colors', cmap)))
    else:
        perc = None
        cmap = Viridis256 if cmap is None else cmap
        if perc is not None:
            condition = percentile(condition, perc)

    data = {'x': x, 'y': y}
    vdims = None
    if condition is not None:
        data['z'] = condition
        vdims = 'z'

    xlim, ylim = pad(*minmax(x)), pad(*minmax(y))
    scatter = (hv.Scatter(data, kdims=[('x', 'x' if xlabel is None else xlabel),
                                       ('y', 'y' if ylabel is None else ylabel)], vdims=vdims)
               .sort(vdims)
               .opts(cmap=cmap, color_index=vdims, show_legend=show_legend))

    legend = None
    if subsample == 'datashade':
        aggregator = None
        if vdims is not None:
            aggregator = ds.count_cat(vdims) if categorical else ds.mean(vdims)
        scatter = dynspread(datashade(scatter, aggregator=aggregator,
                                      color_key=cmap, cmap=cmap,
                                      streams=[hv.streams.RangeXY(transient=True), hv.streams.PlotSize],
                                      min_alpha=255).opts(axiswise=True, framewise=True), threshold=0.8, max_px=5)
        if show_legend and categorical:
            legend = hv.NdOverlay({k: hv.Points([0, 0], label=str(k)).opts(size=0, color=v)
                                   for k, v in cmap.items()})

    elif subsample == 'decimate':
        scatter = decimate(scatter, max_samples=int(adata.n_obs * keep_frac),
                           streams=[hv.streams.RangeXY(transient=True)], random_seed=seed)

    if legend is not None:
        scatter *= legend

    return scatter.opts(title=title if title is not None else '',
                        height=plot_height, width=plot_width, xlim=xlim, ylim=ylim)
