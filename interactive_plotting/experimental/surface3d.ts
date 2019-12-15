import {HTMLBox, HTMLBoxView} from "models/layouts/html_box"
import {ColumnDataSource} from "models/sources/column_data_source"
import * as p from "core/properties"

declare namespace vis {
  class Graph3d {
    constructor(el: HTMLElement, data: object, OPTIONS: object)
    setData(data: vis.DataSet): void
  }

  class DataSet {
    add(data: unknown): void
  }
}

function _tooltip(obj: {x: number, y: number, z: number, data: {tooltip: string}}) {
    return obj.data["tooltip"];
}

const OPTIONS = {
  width: '1200px',
  height: '1200px',
  style: 'dot-color',
  showPerspective: true,
  tooltip: _tooltip,
  showGrid: false,
  showXAxis: false,
  showYAxis: false,
  showZAxis: false,
  keepAspectRatio: true,
  verticalRatio: 1.0,
  cameraPosition: {
    horizontal: 1,
    vertical: 0.25,
    distance: 2.0,
  },
}

export class Surface3dView extends HTMLBoxView {
  model: Surface3d

  private _graph: vis.Graph3d

  render(): void {
    super.render()
    if (this.model.options["tooltip"]) {  // we want to show tooltips
        this.model.options["tooltip"] = _tooltip
    }
    this._graph = new vis.Graph3d(this.el, this.get_data(), this.model.options)
  }

  connect_signals(): void {
    super.connect_signals()
    this.connect(this.model.data_source.change, () => this._graph.setData(this.get_data()))
  }

  get_data(): vis.DataSet {
    const data = new vis.DataSet()
    const source = this.model.data_source

    if ("tooltip" in source.data) {
        for (let i = 0; i < source.get_length()!; i++) {
          data.add({
            x: source.data[this.model.x][i],
            y: source.data[this.model.y][i],
            z: source.data[this.model.z][i],
            style: source.data[this.model.color][i],
            tooltip: source.data["tooltip"][i]
          })
        }
    } else {
        for (let i = 0; i < source.get_length()!; i++) {
          data.add({
            x: source.data[this.model.x][i],
            y: source.data[this.model.y][i],
            z: source.data[this.model.z][i],
            style: source.data[this.model.color][i],
          })
        }
    }

    return data
  }
}

export namespace Surface3d {
  export type Attrs = p.AttrsOf<Props>

  export type Props = HTMLBox.Props & {
    x: p.Property<string>
    y: p.Property<string>
    z: p.Property<string>
    color: p.Property<string>
    data_source: p.Property<ColumnDataSource>
    options: p.Property<{[key: string]: unknown}>
  }
}

export interface Surface3d extends Surface3d.Attrs {}

export class Surface3d extends HTMLBox {
  properties: Surface3d.Props

  constructor(attrs?: Partial<Surface3d.Attrs>) {
    super(attrs)
  }

  static __name__ = "Surface3d"

  static init_Surface3d(): void {
    this.prototype.default_view = Surface3dView

    this.define<Surface3d.Props>({
      x:           [ p.String           ],
      y:           [ p.String           ],
      z:           [ p.String           ],
      color:       [ p.String           ],
      data_source: [ p.Instance         ],
      options:     [ p.Any,     OPTIONS ]
    })
  }
}
