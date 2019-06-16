# Interactive Plotting in Scanpy


## About
This repository contains 6 different interactive plotting functions, which may be useful during exploratory analysis.

Almost every function provides some information when hovering over the plot and some parts of the plots can be hidden by clicking the legend.

## Installation
To install this package, do the following:
```bash
git clone https://github.com/michalk8/interactive_plotting  
cd interactive_plotting  
pip install .
```

## Getting Started
We recommend checking out the [tutorial notebook](./notebooks/interactive_plotting_tutorial.ipynb).

In your Jupyter Notebook, execute the following lines:
```python
import interactive_plotting as ipl  

from bokeh.io import output_notebook
output_notebook()
```

## Examples
Here are some exemplary figures for each of the plotting functions.
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
