from ..utils import *

from bokeh.core.properties import Any, Dict, Instance, String
from bokeh.models import (
    ColumnDataSource,
    LayoutDOM,
    Legend,
    LegendItem,
    ColorBar,
    LinearColorMapper,
    FixedTicker
)
from bokeh.io import save
from bokeh.resources import CDN
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.colors import RGB
from pandas.api.types import is_categorical_dtype
from anndata import AnnData
from typing import Union, Optional, Sequence, Tuple
from time import sleep

import matplotlib
import matplotlib.cm as cm
import numpy as np
import webbrowser
import tempfile


_DEFAULT = {
    'width': '600px',
    'height': '600px',
    'style': 'dot-color',
    'showPerspective': False,
    'showGrid': True,
    'keepAspectRatio': True,
    'verticalRatio': 1.0,
    'cameraPosition': {
        'horizontal': 1,
        'vertical': 0.25,
        'distance': 2,
    }
}


class Surface3d(LayoutDOM):
    __implementation__ = 'surface3d.ts'
    __javascript__ = 'https://unpkg.com/vis-graph3d@latest/dist/vis-graph3d.min.js'

    data_source = Instance(ColumnDataSource)

    x = String
    y = String
    z = String
    color = String

    options = Dict(String, Any, default=_DEFAULT)


def _to_hex_colors(values, cmap, perc=None):
    minn, maxx = minmax(values, perc)
    norm = matplotlib.colors.Normalize(vmin=minn, vmax=maxx, clip=True)

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    return [matplotlib.colors.to_hex(mapper.to_rgba(v)) for v in values], minn, maxx


def _mpl_to_hex_palette(cmap):
    rgb_cmap = (255 * cmap(range(256))).astype('int')

    return [RGB(*tuple(rgb)).to_hex() for rgb in rgb_cmap]


def scatter3d(adata: AnnData,
              key: str,
              basis: str = 'umap',
              components: Sequence[int] = [0, 1, 2],
              steps: Union[Tuple[int, int], int] = 100,
              perc: Optional[Tuple[int, int]] = None,
              n_ticks: int = 10,
              vertical_ratio: float = 1,
              show_axes: bool = False,
              keep_aspect_ratio: bool = True,
              perspective:bool = True,
              tooltips: Optional[Sequence[str]] = [],
              cmap: Optional[matplotlib.colors.ListedColormap] = None,
              dot_size_ratio: float = 0.01,
              plot_height: Optional[int] = 1400,
              plot_width: Optional[int] = 1400):
    """
    Parameters
    ----------
    adata : :class:`anndata.AnnData`
        Annotated data object.
    key
        Key in `adata.obs` or `adata.var_names` to color in.
    basis
        Basis to use.
    components
        Components of the basis to plot.
    steps
        Step size when the subsampling the data.
        Larger step size corresponds to higher density of points.
    perc
        Percentile by which to clip colors.
    n_ticks
        Number of ticks for colorbar if `key` is not categorical.
    vertical_ratio
        Ratio by which to squish the z-axis.
    show_axes
        Whether to show axes.
    keep_aspect_ratio
        Whether to keep aspect ratio.
    perspective
        Whether to keep the perspective.
    tooltips
        Keys in `adata.obs` to visualize when hovering over cells.
    cmap
        Colormap to use.
    dot_size_ratio
        Ratio of the dots with respect to the plot size.
    plot_height
        Height of the plot in pixels. If `None`, try getting the screen height.
    plot_width
        Width of the plot in pixels. If `None`, try getting the screen width.

    Returns
    -------
    None
        Nothing, just plots in a new tab.
    """

    def _wrap_as_div(row, sep=':'):
        res = []
        for kind, val in zip(tooltips, row):
            if isinstance(val, float):
                res.append(f'<div><strong>{kind}</strong>{sep} {val:.04f}</div>')
            else:
                res.append(f'<div><strong>{kind}</strong>{sep} {val}</div>')

        return ''.join(res)

    basis_key = f'X_{basis}'
    assert basis_key in adata.obsm, f'Basis `{basis_key}` not found in `adata.obsm`.'
    if perc is not None:
        assert len(perc) == 2, f'Percentile must be of length `2`, found `{len(perc)}`.'
    assert len(components) == 3, f'Number of components must be `3`, found `{len(components)}`.'
    assert all(c >= 0 for c in components), f'All components must be non-negative, found `{min(components)}`.'
    assert max(components) < adata.obsm[basis_key].shape[-1], \
        f'Component `{max(components)}` is >= than number of components `{adata.obsm[basis_key].shape[-1]}`.'
    assert key in adata.obs or key in adata.var_names, f'Key `{key}` not found in `adata.obs` or `adata.var_names`.'

    if steps is not None:
        adata, _ = sample_unif(adata, steps, bs=basis, components=components[:2])

    data = dict(x=adata.obsm[basis_key][:, components[0]],
                y=adata.obsm[basis_key][:, components[1]],
                z=adata.obsm[basis_key][:, components[2]])

    fig = figure(tools=[], outline_line_width=0, toolbar_location='left', disabled=True)

    if key in adata.obs and is_categorical_dtype(adata.obs[key]):
        hex_palette = _mpl_to_hex_palette(cm.tab20b if cmap is None else cmap)
        mapper = dict(zip(adata.obs[key].cat.categories, adata.uns.get(f'{key}_colors', hex_palette)))
        colors = [str(mapper[c]) for c in adata.obs[key]]

        n_cls = len(adata.obs[key].cat.categories)
        _ = fig.circle([0] * n_cls, [0] * n_cls,
                       color=list(mapper.values()),
                       visible=False, radius=0)
        to_add = Legend(items=[
            LegendItem(label=c, index=i, renderers=[_])
            for i, c in enumerate(mapper.keys())
        ])
    else:
        vals = adata.obs_vector(key) if key in adata.var_names else adata.obs[key]
        colors, minn, maxx = _to_hex_colors(vals, cmap, perc=perc)

        hex_palette = _mpl_to_hex_palette(cm.viridis if cmap is None else cmap)

        _ = fig.circle(0, 0, visible=False, radius=0)

        color_mapper = LinearColorMapper(palette=hex_palette, low=minn, high=maxx)
        to_add = ColorBar(color_mapper=color_mapper, ticker=FixedTicker(ticks=np.linspace(minn, maxx, n_ticks)),
                          label_standoff=12, border_line_color=None, location=(0, 0))

    data['color'] = colors
    if tooltips is None:
        tooltips = adata.obs_keys()
    if len(tooltips):
        data['tooltip'] = adata.obs[tooltips].apply(_wrap_as_div, axis=1)

    source = ColumnDataSource(data=data)
    if plot_width is None or plot_height is None:
        try:
            import screeninfo
            for monitor in screeninfo.get_monitors():
                break
            plot_width = max(monitor.width - 300, 0) if plot_width is None else plot_width
            plot_height = max(monitor.height, 300) if plot_height is None else plot_height
        except ImportError:
            print('Unable to get the screen size, please install package `screeninfo` as `pip install screeninfo`.')
            plot_width = 1200 if plot_width is None else plot_width
            plot_height = 1200 if plot_height is None else plot_height
        except:
            plot_width = 1200 if plot_width is None else plot_width
            plot_height = 1200 if plot_height is None else plot_height

    surface = Surface3d(x="x", y="y", z="z", color="color",
                        data_source=source, options={**_DEFAULT,
                                                     **dict(dotSizeRatio=dot_size_ratio,
                                                            showXAxis=show_axes,
                                                            showYAxis=show_axes,
                                                            showZAxis=show_axes,
                                                            xLabel=f'{basis}_{components[0]}',
                                                            yLabel=f'{basis}_{components[1]}',
                                                            zLabel=f'{basis}_{components[2]}',
                                                            showPerspective=perspective,
                                                            height=f'{plot_height}px',
                                                            width=f'{plot_width}px',
                                                            verticalRatio=vertical_ratio,
                                                            keepAspectRatio=keep_aspect_ratio,
                                                            showLegend=False,
                                                            tooltip='tooltip' in data,
                                                            xCenter='50%',
                                                            yCenter='50%',
                                                            showGrid=show_axes)})

    fig.add_layout(to_add, 'left')

    # dirty little trick, makes plot disappear
    # ideally, one would modify the DOM in the .ts file but I'm just lazy
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.xaxis.visible = False
    fig.yaxis.visible = False

    with tempfile.NamedTemporaryFile(suffix='.html') as fout:
        path = save(row(surface, fig), fout.name, resources=CDN, title=f'Scatter3D - {key}')
        fout.flush()
        webbrowser.open_new_tab(path)
        sleep(2)  # better safe than sorry
