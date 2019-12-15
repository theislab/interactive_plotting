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
from bokeh.io import show
from bokeh.layouts import row
from bokeh.plotting import figure
from bokeh.colors import RGB
from pandas.api.types import is_categorical_dtype

import matplotlib
import matplotlib.cm as cm
import numpy as np


DEFAULTS = {
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

    options = Dict(String, Any, default=DEFAULTS)


def _to_hex_colors(values, cmap, perc=None):
    minn, maxx = minmax(values, perc)
    norm = matplotlib.colors.Normalize(vmin=minn, vmax=maxx, clip=True)

    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    return [matplotlib.colors.to_hex(mapper.to_rgba(v)) for v in values], minn, maxx


def _mpl_to_hex_palette(cmap):
    rgb_cmap = (255 * cmap(range(256))).astype('int')
    return [RGB(*tuple(rgb)).to_hex() for rgb in rgb_cmap]


def scatter3d(adata, key, basis, components=[0, 1, 2],
              steps=100,
              perc=None, n_ticks=10,
              vertical_ratio=1,
              show_axes=False,
              keep_aspect_ratio=True,
              perspective=True,
              tooltips=[], cmap=None,
              dot_size_ratio=0.01,
              height=1400, width=1400):
    def _wrap_as_div(row, sep=':'):
        res = []
        for kind, val in zip(tooltips, row):
            if isinstance(val, float):
                res.append(f'<div><strong>{kind}</strong>{sep} {val:.04f}</div>')
            else:
                res.append(f'<div><strong>{kind}</strong>{sep} {val}</div>')

        return ''.join(res)

    basis_key = f'X_{basis}'
    assert basis_key in adata.obsm
    if perc is not None:
        assert len(perc) == 2
    assert len(components) == 3
    assert all(c >= 0 for c in components)
    assert max(components) < adata.obsm[basis_key].shape[-1]
    assert key in adata.obs or key in adata.var_names

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
    if width is None or height is None:
        try:
            import screeninfo
            for monitor in screeninfo.get_monitors():
                break
            width = max(monitor.width - 300, 0) if width is None else width
            height = max(monitor.height, 300) if height is None else height
        except:
            width = 1200 if width is None else width
            height = 1200 if height is None else height

    surface = Surface3d(x="x", y="y", z="z", color="color",
                        data_source=source, options={**DEFAULTS,
                                                     **dict(dotSizeRatio=dot_size_ratio,
                                                            showXAxis=show_axes,
                                                            showYAxis=show_axes,
                                                            showZAxis=show_axes,
                                                            xLabel=f'{basis}_{components[0]}',
                                                            yLabel=f'{basis}_{components[1]}',
                                                            zLabel=f'{basis}_{components[2]}',
                                                            showPerspective=perspective,
                                                            width=f'{width}px',
                                                            height=f'{height}px',
                                                            verticalRatio=vertical_ratio,
                                                            keepAspectRatio=keep_aspect_ratio,
                                                            showLegend=False,
                                                            tooltip='tooltip' in data,
                                                            xCenter='50%',
                                                            yCenter='50%',
                                                            showGrid=show_axes)})

    fig.add_layout(to_add, 'left')

    # dirty little trick, makes plot disappear
    fig.xgrid.visible = False
    fig.ygrid.visible = False
    fig.xaxis.visible = False
    fig.yaxis.visible = False

    show(row(surface, fig))