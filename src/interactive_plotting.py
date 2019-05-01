#!/usr/bin/env python3
    
from sklearn.gaussian_process.kernels import *
from sklearn import neighbors
from scipy.sparse import issparse
from scipy.spatial import distance_matrix, ConvexHull

from functools import reduce
from collections import defaultdict
from itertools import product

import numpy as np
import pandas as pd
import scanpy as sc

import matplotlib.cm as cm
import matplotlib

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource, Slider, HoverTool, ColorBar, Patches, Legend, CustomJS, TextInput
#from bokeh.models.inputs import TextInput
from bokeh.models.mappers import CategoricalColorMapper
from bokeh.layouts import layout, column, row
from bokeh.transform import linear_cmap, factor_mark, factor_cmap
from bokeh.core.enums import MarkerType
from bokeh.palettes import Set1, Set2, Set3, viridis
from bokeh.models.widgets import CheckboxGroup
from bokeh.models.widgets.buttons import Button


_inter_hist_js_code="""
    // here is where original data is stored
    var x = orig.data['values'];

    x = x.sort((a, b) => a - b);
    var n_bins = parseInt(bins.value); // can be either string or int
    var bin_size = (x[x.length - 1] - x[0]) / n_bins;

    var hist = new Array(n_bins).fill().map((_, i) => { return 0; });
    var l_edges = new Array(n_bins).fill().map((_, i) => { return x[0] + bin_size * i; });
    var r_edges = new Array(n_bins).fill().map((_, i) => { return x[0] + bin_size * (i + 1); });
    var indices = new Array(n_bins).fill().map((_) => { return []; });

    // create the histogram
    for (var i = 0; i < x.length; i++) {
        for (var j = 0; j < r_edges.length; j++) {
            if (x[i] <= r_edges[j]) {
                hist[j] += 1;
                indices[j].push(i);
                break;
            }
        }
    }

    // make it a density
    var sum = hist.reduce((a, b) => a + b, 0);
    var deltas = r_edges.map((c, i) => { return c - l_edges[i]; });
    // just like in numpy
    hist = hist.map((c, i) => { return c / deltas[i] / sum; });

    source.data['hist'] = hist;
    source.data['l_edges'] = l_edges;
    source.data['r_edges'] = r_edges;
    source.data['indices'] = indices;

    source.change.emit();
"""


def interactive_histograms(adata, keys=['n_counts', 'n_genes'],
                           bins='auto', min_bins=1, max_bins=1000,
                           tools='pan, reset, wheel_zoom, save',
                           groups=None, fill_alpha=0.4,
                           palette=Set1[9] + Set2[8] + Set3[12],
                           legend_loc='top_right', display_all=True,
                           *args, **kwargs):
    """Utility function to plot count distributions\

    Uses the bokey library to create interactive histograms, which can be used
    e.g. to set filtering thresholds.

    Params
    --------
    adata: AnnData Object
        annotated data object
    keys: list, optional (default: `["n_counts", "n_genes"]`)
        keys in adata.obs or adata.var where the distibutions are stored
    bins: int, str, optional (default: `auto`)
        number of bins used for plotting or str from numpy.histogram
    min_bins: int, optional (default: `1`)
        minimum number of bins possible
    max_bins: int, optional (default: `1000`)
        maximum number of bins possible
    groups: list[str], (default: `None`)
        in adata.obs; groups by all possible combinations of values, e.g. for
        3 plates and 2 time points, we would create total of 6 groups
    fill_alpha: float[0.0, 1.0], (default: `0.4`)
        alpha channel of fill color
    legend_loc: str, (default: `top_right`)
        position of the legend
    tools: str, optional (default: `"pan,reset, wheel_zoom, save"`)
        palette of interactive tools for the user
    palette: list, optional (default: `Set1[9] + Set2[8] + Set3[12]`)
         colors from bokeh.palettes, e.g. Set1[9]
    display_all: bool, optional (default: `True`)
        display the statistics for all data
    **kwargs: keyword arguments for figure
        specify e.g. `"plot_width"` to set the width of the figure.

    Returns
    --------
    None
    """


    if min_bins < 1:
        raise ValueError(f'Expected min_bins >= 1, got min_bins={min_bins}.')
    if max_bins < min_bins:
        raise ValueError(f'Expected min_bins <= max_bins, got min_bins={min_bins}, max_bins={max_bins}.')

    # check the input
    for key in keys:
        if key not in adata.obs.keys() and \
           key not in adata.var.keys() and \
           key not in adata.var_names:
            raise ValueError(f'The key `{key}` does not exist in adata.obs, adata.var or adata.var_names.')

    def _create_adata_groups():
        if groups is None:
            return [('all',)], [adata]

        combs = list(product(*[set(adata.obs[g]) for g in groups]))
        adatas= [adata[reduce(lambda l, r: l & r,
                              (adata.obs[k] == v for k, v in zip(groups, vals)), True)]
                 for vals in combs] + [adata]

        if display_all:
            combs += [('all',)]
            adatas += [adata]

        return combs, adatas

    # group_v_combs contains the value combinations
    # used for grupping
    group_v_combs, adatas = _create_adata_groups()
    n_plots = len(group_v_combs)
    checkbox_group = CheckboxGroup(active=list(range(n_plots)), width=200)
    
    for key in keys:
        # create histogram
        cols, legends, callbacks = [], [], []
        plot_map = dict()
        fig = figure(*args, tools=tools, **kwargs)
        slider = Slider(start=min_bins, end=max_bins, value=0, step=1,
                        title='Bins')

        plot_ids = []
        for j, (ad, group_vs) in enumerate(zip(adatas, group_v_combs)):

            if ad.n_obs == 0:
                continue
            
            plot_ids.append(j)
            color = palette[len(plot_ids) - 1]

            if key in ad.obs.keys():
                orig = ad.obs[key]
                hist, edges = np.histogram(orig, density=True, bins=bins)
            elif key in ad.var.keys():
                orig = ad.var[key]
                hist, edges = np.histogram(orig, density=True, bins=bins)
            else:
                orig = ad[:, key].X
                hist, edges = np.histogram(orig, density=True, bins=bins)

            slider.value = len(hist)

            # original data, used for recalculation of histogram in JS code
            orig = ColumnDataSource(data=dict(values=orig))
            # data that we update in JS code
            source = ColumnDataSource(data=dict(hist=hist, l_edges=edges[:-1], r_edges=edges[1:]))

            legend = ', '.join(': '.join(map(str, gv)) for gv in zip(groups, group_vs)) \
                    if groups is not None else 'all'
            legends.append(legend)
            # create figure
            p = fig.quad(source=source, top='hist', bottom=0,
                         left='l_edges', right='r_edges',
                         fill_color=color, legend=legend,
                         line_color="#555555", fill_alpha=fill_alpha)

            # create callback and slider
            callback = CustomJS(args=dict(source=source, orig=orig), code=_inter_hist_js_code)
            callback.args['bins'] = slider
            callbacks.append(callback)

            # add the current plot so that we can set it
            # visible/invisible in JS code
            plot_map[f'p_{j}'] = p

        # slider now updates all values
        slider.js_on_change('value', *callbacks)
        plot_map['cb'] = checkbox_group

        button = Button(label='Toggle All', button_type='primary')
        code_t='\n'.join(f'p_{p_id}.visible = false;' for i, p_id in enumerate(plot_ids))
        code_f ='\n'.join(f'p_{p_id}.visible = true;' for i, p_id in enumerate(plot_ids))
        button.callback = CustomJS(
            args=plot_map,
            code=f'''if (cb.active.length == {len(plot_map) - 1}) {{
                cb.active = Array();
                {code_t};
            }} else {{
                cb.active = Array.from(Array({len(plot_map) - 1}).keys());
                {code_f};
            }}'''
        )

        checkbox_group.callback = CustomJS(
            args=plot_map,
            code='\n'.join(f'p_{p_id}.visible = cb.active.includes({i});' for i, p_id in enumerate(plot_ids))
        )
        checkbox_group.labels = legends

        fig.legend.location = legend_loc
        fig.xaxis.axis_label = key
        fig.yaxis.axis_label = 'normalized frequency'
        fig.plot_width = kwargs.get('plot_width', 400)
        fig.plot_height = kwargs.get('plot_height', 400)

        cols.append(column(slider, button, row(fig, checkbox_group)))


    # transform list of pairs of figures and sliders into list of lists, where
    # each sublist has length <= 2
    # note that bokeh does not like np.arrays
        grid = list(map(list, np.array_split(cols, np.ceil(len(cols) / 2))))

        show(layout(children=grid, sizing_mode='fixed', ncols=2))


def hist_thres(adata, key='n_counts', groups=[], bins='auto', basis='umap', components=[1, 2]):
    if not isinstance(components, np.ndarray):
        components = np.asarray(components)

    fig = figure()
    hist, edges = np.histogram(adata.obs[key], density=True, bins=bins)
    
    inputs = []
    source = ColumnDataSource(data=dict(hist=hist, l_edges=edges[:-1], r_edges=edges[1:], color=['#000000'] * len(hist), indices=[[]] * len(hist)))
    df = pd.DataFrame(adata.obsm[f'X_{basis}'][:, components - 1], columns=['x', 'y'])
    df['values'] = list(adata.obs[key])
    df['color'] = '#000000'

    orig = ColumnDataSource(df)
    p = fig.quad(source=source, top='hist', bottom=0,
                 left='l_edges', right='r_edges', color='color',
                 line_color="#555555")
    fig2 = figure()

    um = fig2.scatter('x', 'y', source=orig, color='color', size=20)

    source1 = ColumnDataSource(dict(xs=[40000, 40000], ys=[0, 0.00001]))
    fig.line('xs', 'ys', source=source1, line_width=5, color='red')

    source2 = ColumnDataSource(dict(xs=[80000, 80000], ys=[0, 0.00001]))
    fig.line('xs', 'ys', source=source2, line_width=5, color='red')

    # create callback and slider
    for group in groups:
        input = TextInput(name='test', value='40000', title=f'{group} min')
        input2 = TextInput(name='test', value='80000', title=f'{group} max')
        #callback = CustomJS(args=dict(source=source, orig=orig), code=_inter_hist_js_code)
        i1back = CustomJS(args={'source': source1,'bins': input}, code=f'''
            var x = parseInt(bins.value);
            source.data['xs'] = [x, x];
            source.change.emit();
        ''')
        i2back = CustomJS(args={'source': source, 'input_min': input, 'input_max': input2, 'orig': orig}, code='''
            var min = parseInt(input_min.value);
            var max = parseInt(input_max.value);
            console.log(source.data['indices']);
            for (var i = 0; i < source.data['hist'].length; i++) {
                var mid = (source.data['r_edges'][i] - source.data['l_edges'][i]) / 2;
                if (source.data['l_edges'][i] + mid >= min && source.data['r_edges'][i] - mid <= max) {
                    source.data['color'][i] = '#FF0000';
                    for (var j = 0; j < source.data['indices'][i].length; j++) {
                        orig.data['color'][source.data['indices'][i][j]] = '#FF0000';
                    }
                } else {
                    source.data['color'][i] = '#000000';
                    for (var j = 0; j < source.data['indices'][i].length; j++) {
                        orig.data['color'][source.data['indices'][i][j]] = '#000000';
                    }
                }
            }
            orig.change.emit();
            source.change.emit();
        ''')
        input.js_on_change('value', *[i1back, i2back])

        i3back = CustomJS(args={'source': source2,'bins': input2}, code=f'''
            var x = parseInt(bins.value);
            source.data['xs'] = [x, x];
            source.change.emit();
        ''')
        input2.js_on_change('value', *[i3back, i2back])

        slider = Slider(start=0, end=100, value=len(hist), title='Bins')
        c2back = CustomJS(args={'source': source, 'orig': orig, 'bins': slider}, code=_inter_hist_js_code)
        slider.js_on_change('value', *[c2back, i2back])
        inputs += [input, input2]


    show(column(row(fig, column(inputs + [slider])), fig2))


def smooth_expression(x, y, n_points=100, mode='gp', kernel_params=dict(), kernel_default_params=dict(),
                     kernel_expr=None, suppress=False, **opt_params):

    from sklearn.kernel_ridge import KernelRidge
    from sklearn.gaussian_process import GaussianProcessRegressor
    import operator as op
    import ast

    def _eval(node):
        if isinstance(node, ast.Num):
            return node.n

        if isinstance(node, ast.Name):
            if not suppress and node.id not in kernel_params:
                raise ValueError(f'Error while parsing `{kernel_expr}`: `{node.id}` is not a valid key in kernel_params. To use RBF kernel with default parameters, specify suppress=True.')
            params = kernel_params.get(node.id, kernel_default_params)
            kernel_type = params.pop('type', 'rbf')
            return kernels[kernel_type](**params)

        if isinstance(node, ast.BinOp):
            return operators[type(node.op)](_eval(node.left), _eval(node.right))

        if isinstance(node, ast.UnaryOp):
            return operators[type(node.op)](_eval(node.operand))

        raise TypeError(node)

    operators = {ast.Add : op.add,
                 ast.Mult: op.mul,
                 ast.Pow :op.pow}
    kernels = dict(const=ConstantKernel,
                   white=WhiteKernel,
                   rbf=RBF,
                   mat=Matern,
                   rq=RationalQuadratic,
                   esn=ExpSineSquared,
                   dp=DotProduct,
                   pw=PairwiseKernel)

    x_test = np.linspace(0, 1, n_points)[:, None]

    if mode == 'krr':
        gamma = opt_params.pop('gamma', None)

        if gamma is None:
            length_scale = kernel_default_params.get('length_scale', 0.2)
            gamma = 1 / (2 * length_scale ** 2)
            print(f'Smoothing using KRR with length_scale: {length_scale}.')

        model = KernelRidge(gamma=gamma, **opt_params)
        model.fit(x, y)

        return x_test, model.predict(x_test), None

    if mode == 'gp':

        if kernel_expr is None:
            assert len(kernel_params) == 1
            kernel_expr, = kernel_params.keys()

        kernel = _eval(ast.parse(kernel_expr, mode='eval').body)
        alpha = opt_params.pop('alpha', None)
        if alpha is None:
            alpha = np.std(y) 
        optimizer = opt_params.pop('optimizer', None)
        model = GaussianProcessRegressor(kernel=kernel, alpha=alpha, optimizer=None, **opt_params)
        model.fit(x, y)

        mean, cov = model.predict(x_test, return_cov=True)
        return x_test, mean, cov

    raise ValueError(f'Uknown type: `{type}`.')


def compute_dist(x_obs, x_theo, weights=None):
    """
    Utility funciton to compute distance between point cloud and curve
    :param x_obs: observed data
    :param x_theo: theoretical data/curve
    :return: distance measure
    """

    """
    score = 0
    for point in x_obs:
        point_extended = np.ones((x_theo.shape[0], 1)) @ point[None, :]
        dist = np.linalg.norm(point_extended-x_theo, axis=1)
        ix = np.argmin(dist)
        min_dist = dist[ix]
        if weights is not None:
            min_dist *= weights[ix]*min_dist

        score += min_dist
    """
    from fastdtw import fastdtw
    score, path = fastdtw(x_obs, x_theo, dist=2)

    return score


def shift_scale(x_obs, x_theo, fit_intercept=False):
    """Utility funciton to shift and scale the integrated velocities

    :param exp_pred: np.array
        Integrated velocities, predictor for gene_expression
    :param gene_exp: np.array
        Original gene expression values
    :param exp_mean: np.array
        Smoothed gene expression
    :return: exp_pred
        Shifted and scaled integrated velocities
    """

    # find the best possible scaling factor using simple lin reg
    # this accounts for not knowing beta
    from sklearn.linear_model import LinearRegression

    reg = LinearRegression(fit_intercept=fit_intercept)
    reg.fit(x_obs[:, None], x_theo)

    return reg.coef_, reg.intercept_


def pred_exp(X_test, y_test):
    """Predict gene expression based on velocities

    Parameters
    --------
    X_test: np.array
        grid of points in feature space for prediction
    y_test: np.array
        smoothed velocity values

    Returns
    --------
    y_pred: np.array
        predicted values from velocities

    """

    # integrate the velocity to get gene expression
    from scipy.integrate import simps
    n_points = X_test.shape[0]

    # define a function for the derivatife
    def integrate(t, y, x):
        return simps(y[:t], x[:t])

    # compute on a grid
    y_pred = np.array([integrate(t, y_test, X_test.flatten())
                       for t in range(1, n_points + 1)])
    return y_pred


def plot_velocity(adata, genes=None, paths=[], n_velocity_genes=5,
              exp_key='X', mode='gp', smooth=True, length_scale=0.2,
              differentiate=True,
              return_values=False,
              velo_key_ss='velocity',
              velo_key_dyn='velocity_dynamical',
              scatter_kwgs=None,
              color_key='louvain',
              path_key='louvain',
              plotting=True):
    """Plotting function which shows expression level as well as velocity per gene as a function of DPT.
    Aim here is to check for the consistency of the computed velocities

    Params
    --------
    adata: AnnData object
        Annotated data matrix
    genes: list or None, optional (default: `None`)
        List of genes to show
    n_points: int, optional (default: `100`)
        Number of points for the prediction
    smooth: bool, optional (default: `True`)
        Whether to compute smoothed curves
    length_scale : float, optional (default `0.2`)
        length scale for RBF kernel
    mode: str, optional (default: `krr`)
        Whether to use Kernel Ridge Regressin (krr) or a Gaussian Process (gp) for
        smoothing the expression values
    differentiate: bool, optional (default: `True`)
        Whether to take the derivative of gene expression
    return_values: bool, optional (default: `False`)
        Whether to return computed values
    exp_key: str, optional (default:`'X'"`)
        key from adata.layers or just 'X' to get gene expression values
    velo_key_ss, velo_key_dyn: str, optional (default: `"velocity"`)
        key from adata.layers to get velocity  values for the steady state (ss)
        and the dynamical (dyn) model
    scatter_kwgs: dict or None, optional (default: `None`)
        Keyword arguments for scv.pl.scatter
    plotting: bool, optional (default `True`)
        Whether to plot

    Returns
    --------
    Depends on the value of return_values. If True, returns the following:
    dpt: np.array
        Diffusion pseudotime
    gene_exp: np.array
        Gene expresion values for the last gene
    velo_exp_ss: np.array
        ss velocities
    velo_exp_dyn: np.array
        dyn velocities
    """

    for path in paths:
        for p in path:
            assert p in adata.obs[path_key].cat.categories, f'`{p}` is not in `adata.obs[path_key]`. Possible values are: `{list(adata.obs[path_key].cat.categories)}`.'

    # check the input
    if 'dpt_pseudotime' not in adata.obs.keys():
        raise ValueError('Compute `dpt` first.')
    if velo_key_ss not in adata.layers.keys():
        pass
        #raise ValueError(f'Compute `{velo_key_ss}` first.')
    if velo_key_dyn not in adata.layers.keys():
        pass
        #raise ValueError(f'Compute `{velo_key_dyn}` first.')

    # check the genes list
    if genes is None:
        genes = adata[:, adata.var['velocity_genes']].var_names[:n_velocity_genes]
    else:
        # check whether all of those genes exist in the adata object
        genes_indicator = [gene in adata.var_names for gene in genes]
        if not all(genes_indicator):
            genes_missing = np.array(genes)[np.invert(genes_indicator)]
            print(f'Could not find the following genes: `{genes_missing}`.')
            genes = list(np.array(genes)[genes_indicator])


    mapper = create_mapper(adata, color_key)

    figs = []
    for gene in genes:
        data = defaultdict(list)
        for path in paths:
            path_ix = np.in1d(adata.obs[path_key], path)
            ad = adata[path_ix].copy()
            if exp_key != 'X':
                gene_exp = ad[:, gene].layers[exp_key]
            else:
                gene_exp = ad.raw[:, gene].X
            # exclude dropouts
            ix = (gene_exp > 0)
            dpt = ad.obs['dpt_pseudotime']

            #velo_exp_ss = ad[:, gene].layers[velo_key_ss]
            #velo_exp_dyn = ad[:, gene].layers[velo_key_dyn]

            if issparse(gene_exp):
                gene_exp = gene_exp.A
            #if issparse(velo_exp_ss): velo_exp_ss = velo_exp_ss.A
            #if issparse(velo_exp_dyn): velo_exp_dyn = velo_exp_dyn.A
            gene_exp = gene_exp.flatten()
            data['expr'].append(gene_exp)
            #data['velo_expr_ss'].append(velo_exp_ss)
            #data['velo_expr_dyn'].append(velo_exp_dyn)

            # scale the steady state velocities
            #scaling_factor, _ = shift_scale(velo_exp_ss, velo_exp_dyn, fit_intercept=False)
            #velo_exp_ss = scaling_factor * velo_exp_ss

            # compute smoothed values from expression
                # gene expression
            data['dpt'].append(list(dpt))
            data[color_key].append(list(ad.obs[color_key]))
            if smooth:
                assert all(gene_exp[ix] > 0)
                x_test, exp_mean, exp_cov = smooth_expression(dpt[ix, None], gene_exp[ix], mode=mode,
                    n_points=len(dpt), kernel_params=dict(k=dict(length_scale=length_scale)))
                                                          
                data['x_test'].append(x_test)
                data['x_mean'].append(exp_mean)
                data['x_cov'].append(exp_cov)
            continue

            if differentiate:
                if not smooth:
                    raise ValueError('You must smooth the data to do compute derivatives.')

                print('Taking the derivative of gene expression...')
                spacing = x_test[1, 0] - x_test[0, 0]
                gene_grad = np.gradient(exp_mean, spacing)
                data['gene_grad'].append(gene_grad)

                # compute goodness-of-velocities measure
                x_obs_ss = np.concatenate((dpt[:, None], velo_exp_ss[:, None]), axis=1)
                x_obs_dyn = np.concatenate((dpt[:, None], velo_exp_dyn[:, None]), axis=1)
                x_theo = np.concatenate((x_test, gene_grad[:, None]), axis=1)

                weights = 1/np.sqrt(np.diag(exp_cov))
                weights = weights/sum(weights)

                score_ss = compute_dist(x_obs_ss, x_theo, weights)
                score_dyn = compute_dist(x_obs_dyn, x_theo, weights)

                data['score_ss'].append(score_ss)
                data['score_dyn'].append(score_dyn)

                print('ss_score = {:2.2e}\ndyn_score = {:2.2e}'.format(score_ss, score_dyn))

        dataframe = pd.DataFrame(data, index=list(map(lambda path: ', '.join(path), paths)))

        if plotting:
            figs.append(create_velocity_figure(dataframe, color_key, title=gene, color_mapper=mapper))

    show(column(*figs))

    if return_values:
        return data


def create_mapper(adata, key):
    # TODO:
    # plate colors return float
    palette = adata.uns.get(f'{key}_colors', viridis(len(adata.obs[key].unique())))
    key_col = adata.obs[key].astype('category') if adata.obs[key].dtype.name != 'category' else adata.obs[key]
    return CategoricalColorMapper(palette=palette, factors=list(map(str, key_col.cat.categories)))


def create_velocity_figure(dataframe, color_key, title, color_mapper):
    
    markers = [marker for marker in MarkerType if marker not in ['circle_cross', 'circle_x']] * 10
    fig = figure(title=title)

    for i, (marker, (path, df)) in enumerate(zip(markers, dataframe.iterrows())):
        ds = dict(df)
        source = ColumnDataSource(ds)
        fig.scatter('dpt', 'expr', source=source, color={'field': color_key, 'transform': color_mapper},
                    marker=marker, size=10, legend=f'{path}', muted_alpha=0)
        fig.xaxis.axis_label = 'dpt'
        fig.yaxis.axis_label = 'expression'
        if ds.get('x_test') is not None:
            if ds.get('x_mean') is not None:
                fig.line('x_test', 'x_mean', source=source, muted_alpha=0, legend=path)
                if ds.get('x_cov') is not None:
                    x_mean = ds['x_mean']
                    x_cov = ds['x_cov']
                    band_x = np.append(ds['x_test'][::-1], ds['x_test'])
                    band_y = np.append((x_mean - np.sqrt(np.diag(x_cov)))[::-1], (x_mean + np.sqrt(np.diag(x_cov))))
                    fig.patch(band_x, band_y, alpha=0.1, line_color='black', fill_color='black',
                              legend=path, line_dash='dotdash', muted_alpha=0)
            if ds.get('x_grad') is not None:
                fig.line('x_test', 'x_grad', source=source, muted_alpha=0)


    fig.legend.click_policy = 'mute'

    return fig


def highlight_de(adata, basis='umap', components=[1, 2], n_top_de_genes=10,
                 de_values='names, scores, pvals_adj, logfoldchanges',
                 cell_values=[], n_neighbors=5,
                 fill_alpha=0.1, show_hull=True):

    if 'rank_genes_groups' not in adata.uns_keys():
        raise ValueError('Run differential expression first.')


    if isinstance(de_values, str):
        de_values = list(dict.fromkeys(map(str.strip, de_values.split(','))))
        if de_values != ['']:
            assert all(map(lambda k: k in adata.uns['rank_genes_groups'].keys(), de_values)), 'Not all keys are in `adata.uns[\'rank_genes_groups\']`.'
        else:
            de_values = []

    if isinstance(cell_values, str):
        cell_values = list(dict.fromkeys(map(str.strip, cell_values.split(','))))
        if cell_values != ['']:
            assert all(map(lambda k: k in adata.obs.keys(), cell_values)), 'Not all keys are in `adata.obs.keys()`.'
        else:
            cell_values = []

    if f'X_{basis}' not in adata.obsm.keys():
        raise ValueError(f'Key `X_{basis}` not found in adata.obsm.')

    if not isinstance(components, np.ndarray):
        components = np.asarray(components)

    key = adata.uns['rank_genes_groups']['params']['groupby']
    if key not in cell_values:
        cell_values.insert(0, key)

    df = pd.DataFrame(adata.obsm[f'X_{basis}'][:, components - 1], columns=['x', 'y'])
    for k in cell_values:
        df[k] = list(map(str, adata.obs[k]))

    knn = neighbors.KNeighborsClassifier(n_neighbors)
    knn.fit(df[['x', 'y']], adata.obs[key])
    df['prediction'] = knn.predict(df[['x', 'y']])

    conv_hulls = df[df[key] == df['prediction']].groupby(key).apply(lambda df: df.iloc[ConvexHull(np.vstack([df['x'], df['y']]).T).vertices])

    mapper = create_mapper(adata, key)
    categories = adata.obs[key].cat.categories
    fig = figure(tools='pan, reset, wheel_zoom, lasso_select, save')
    legend_dict = defaultdict(list)

    for k in categories:
        d = df[df[key] == k]
        data_source =  ColumnDataSource(d)
        legend_dict[k].append(fig.scatter('x', 'y', source=data_source, color={'field': key, 'transform': mapper}, size=5, muted_alpha=0))

    hover_cell = HoverTool(renderers=[r[0] for r in legend_dict.values()], tooltips=[(f'{key}', f'@{key}')] + [(f'{k}', f'@{k}') for k in cell_values[1:]])

    c_hulls = conv_hulls.copy()
    de_possible = conv_hulls[key].isin(adata.uns['rank_genes_groups']['names'].dtype.names)
    ok_patches = []
    prev_cat = []
    for i, isin in enumerate((~de_possible, de_possible)):
        conv_hulls = c_hulls[isin]

        if len(conv_hulls) == 0:
            continue

        xs, ys, ks = zip(*conv_hulls.groupby(key).apply(lambda df: list(map(list, (df['x'], df['y'], df[key])))))
        tmp_data = defaultdict(list)
        tmp_data['xs'] = xs
        tmp_data['ys'] = ys
        tmp_data[key] = list(map(lambda k: k[0], ks))
        
        if i == 1:
            ix = list(map(lambda k: adata.uns['rank_genes_groups']['names'].dtype.names.index(k), tmp_data[key]))
            for k in de_values:
                tmp = np.array(list(zip(*adata.uns['rank_genes_groups'][k])))[ix, :n_top_de_genes]
                for j in range(n_top_de_genes):
                    tmp_data[f'{k}_{j}'] = tmp[:, j]

        tmp_data = pd.DataFrame(tmp_data)
        for k in categories:
            d = tmp_data[tmp_data[key] == k]
            source = ColumnDataSource(d)

            patches = fig.patches('xs', 'ys', source=source, fill_alpha=fill_alpha, muted_alpha=0, hover_alpha=0.5,
                                  color={'field': key, 'transform': mapper} if (show_hull and i == 1) else None,
                                  hover_color={'field': key, 'transform': mapper} if (show_hull and i == 1) else None)
            legend_dict[k].append(patches)
            if i == 1:
                ok_patches.append(patches)

    hover_group = HoverTool(renderers=ok_patches, tooltips=[(f'{key}', f'@{key}'),
        ('groupby', adata.uns['rank_genes_groups']['params']['groupby']),
        ('reference', adata.uns['rank_genes_groups']['params']['reference']),
        ('rank', ' | '.join(de_values))] + [(f'#{i + 1}', ' | '.join((f'@{k}_{i}' for k in de_values))) for i in range(n_top_de_genes)]
    )
    

    fig.toolbar.active_inspect = [hover_group]
    if len(cell_values) > 1:
        fig.add_tools(hover_group, hover_cell)
    else:
        fig.add_tools(hover_group)

    legend = Legend(items=list(legend_dict.items()), location='top_right')
    fig.add_layout(legend)
    fig.legend.click_policy = 'hide'

    show(fig)


def cmap_to_colors(cmap_vals):
    return list(('#' + ''.join(map(lambda val: '{:02x}'.format(val).upper(), item[:-1])) for item in cmap_vals))


def multi_link(adata, bases=['umap', 'pca'], components=[1, 2], markers=None,
               fade=True, highlight_only=None, palette=None):

    assert isinstance(palette, matplotlib.colors.Colormap), '`palette` must be instance of `matplotlib.colors.Colormap`'
    assert len(components) == 2, f'`components` argument must be of length 2.'

    if not isinstance(components, np.ndarray):
        components = np.asarray(components)

    if highlight_only is not None:
        assert highlight_only in adata.obs_keys(), f'`{highlight_only}` is not in adata.obs_keys().'

    if markers is None:
        markers = markers = adata.var_names 

    gene_subset = np.in1d(adata.var_names, markers)
    dmat = pd.DataFrame(distance_matrix(adata.X[:, gene_subset],
                                        adata.X[:, gene_subset]),
                        columns=list(map(str, range(adata.n_obs))))
    df = pd.concat([pd.DataFrame(adata.obsm[f'X_{base}'][:, components - (base != 'diffmap')], columns=[f'x{i}', f'y{i}'])
                    for i, base in enumerate(bases)] + [dmat], axis=1)
    df['color'] = np.nan
    df['index'] = range(len(df))
    df['hl_key'] = list(adata.obs[highlight_only]) if highlight_only is not None else 0

    ds = ColumnDataSource(df)
    start_ix = str(adata.uns.get('iroot', 0))
    palette = cm.RdYlBu if palette is None else palette
    mapper = linear_cmap(field_name='color', palette=cmap_to_colors(palette(range(palette.N), 1., bytes=True)),
                         low=df[start_ix].min(), high=df[start_ix].max())

    figs, renderers = [], []
    for i, basis in enumerate(bases):
        fig = figure(tools='pan, reset, save, ' + ('zoom_in, zoom_out' if i == 0 else 'wheel_zoom'),
                     title=basis, plot_width=400, plot_height=400)
        scatter = fig.scatter(f'x{i}', f'y{i}', source=ds, line_color=mapper,color=mapper,
                          hover_color='black', size=6, line_width=8, line_alpha=0)
        figs.append(fig)
        renderers.append(scatter)
    fig = figs[0]

    end = dmat.max().max()
    slider = Slider(start=0, end=end, value=end / 2, step=end / 1000,
            title='Distance')
    col_ds = ColumnDataSource(dict(value=[start_ix]))
    update_color_code = f'''
        source.data['color'] = source.data[first].map(
            (x, i) => {{ return isNaN(x) ||
                        {'' if fade else 'x > slider.value || '}
                        source.data['hl_key'][first] != source.data['hl_key'][i]  ? NaN : x; }}
        );
    '''
    slider.callback = CustomJS(args={'slider': slider, 'mapper': mapper['transform'], 'source': ds, 'col': col_ds}, code=f'''
        mapper.high = slider.value;
        var first = col.data['value']
        {update_color_code}
    ''')

    h_tool = HoverTool(renderers=renderers, tooltips=[], show_arrow=False)
    h_tool.callback = CustomJS(args=dict(source=ds, slider=slider, col=col_ds), code=f'''
        var indices = cb_data.index['1d'].indices;
        if (indices.length == 0) {{
            source.data['color'] = source.data['color'];
        }} else {{
            first = indices[0];
            source.data['color'] = source.data[first];
            {update_color_code}
            col.data['value'] = first;
            col.change.emit();
        }}
        source.change.emit();
    ''')
    fig.add_tools(h_tool)

    color_bar = ColorBar(color_mapper=mapper['transform'], width=12, location=(0,0))
    fig.add_layout(color_bar, 'left')

    fig.add_tools(h_tool)
    show(column(slider, row(*figs)))


if __name__ == '__main__':
    pass
