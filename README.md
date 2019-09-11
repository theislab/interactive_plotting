# Interactive Plotting in Scanpy


## About
This repository contains 10 different interactive plotting functions, which may be useful during exploratory analysis.

Almost every function provides some information when hovering over the plot and some parts of the plots can be hidden by clicking the legend.

## Installation
To install this package, do the following:
```bash
git clone https://github.com/michalk8/interactive_plotting  
cd interactive_plotting  
pip install -e .
```

## Getting Started
We recommend checking out the [tutorial notebook](./notebooks/interactive_plotting_tutorial.ipynb).
```ipl.scatter```, ```ipl.scatterc``` ```ipl.dpt``` can handle large number of cells (100K+).

In your Jupyter Notebook, execute the following lines:
```python
import holoviews as hv  # needed for scatter, scatterc and dpt
hv.extension('bokeh')

import interactive_plotting as ipl  

from bokeh.io import output_notebook
output_notebook()
```

## Gallery
Here are some exemplary figures for each of the plotting functions.
```python
ipt.scatter
```
![Scatterplot (cont)](resources/images/scatter_cont.png?raw=true "Scatterplot (continous)")

---

```python
ipt.scatterc
```
![Scatterplot (cat)](resources/images/scatter_cat.png?raw=true "Scatterplot (categorical)")

---

```python
ipt.dpt
```
![DPT plot](resources/images/dpt_plot.png?raw=true "DPT plot")

```python
ipt.graph
```
![Graph plot](resources/images/graph_plot.png?raw=true "Graph plot")

---

```python
ipl.link_plot
   ``` 
![link plot](resources/images/link_plot.png?raw=true "Link plot")

---

```python
ipl.highlight_de
```
![highlight differential expression plot](resources/images/highlight_de.png?raw=true "Highlight differential expression")

---

```python
ipl.gene_trend
```
![gene trend](resources/images/gene_trend.png?raw=true "Gene trend")

---

```python
ipl.interactive_hist
```
![interactive histogram](resources/images/inter_hist.png?raw=true "Interactive histogram")

---

```python
ipl.thresholding_hist
```
![thresholding histogram](resources/images/thresh_hist.png?raw=true "Thresholding histogram")

---

```python
ipl.highlight_indices
```
![highlight cell indices plot](resources/images/highlight_indices.png?raw=true "Highlight cell indices")

## Troubleshooting
* [Notebook size is **huge**](https://github.com/theislab/interactive_plotting/issues/2) - This has to do with ```ipl.link_plot``` and ```ipl.velocity_plot```. Until a fix is found, we suggest removing these figures after you're done using them.
* [Getting "OPub data rate exceeded" error](https://github.com/theislab/interactive_plotting/issues/7) - Try starting jupyter notebook as following:

    ```jupyter notebook --NotebookApp.iopub_data_rate_limit=1e10```

  For generating jupyter config file, see [here](https://stackoverflow.com/questions/43288550/iopub-data-rate-exceeded-in-jupyter-notebook-when-viewing-image).
