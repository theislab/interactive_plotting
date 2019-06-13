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

## Usage
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
![link plot](https://github.com/theislab/interactive_plotting/tree/master/resources/images/link_plot.png "Link plot")

---

```python
ipl.highlight_de
```
![highlight differential expression plot](https://github.com/theislab/interactive_plotting/tree/master/resources/images/highlight_de.png "Highlight differential expression")

---

```python
ipl.velocity_plot
```
![velocity plot](https://github.com/theislab/interactive_plotting/tree/master/resources/images/velocity_plot.png "Velocity plot")

---

```python
ipl.interactive_hist
```
![interactive histogram](https://github.com/theislab/interactive_plotting/tree/master/resources/images/inter_hist.png "Interactive histogram")

---

```python
ipl.thresholding_hist
```
![thresholding histogram](https://github.com/theislab/interactive_plotting/tree/master/resources/images/thresh_hist.png "Thresholding histogram")

---

```python
ipl.highlight_indices
```
![highlight cell indices plot](https://github.com/theislab/interactive_plotting/tree/master/resources/images/highlight_indices.png "Highlight cell indices")
