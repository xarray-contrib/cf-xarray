---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# Grid Mappings

`cf_xarray` understands the concept of coordinate projections using the [grid_mapping](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#grid-mappings-and-projections) attribute convention. For example, the dataset might contain two sets of coordinates:

- native coordinates in which the data is defined, e.g., regular 1D coordinates
- projected coordinates which probably denote some "real" coordinates in [latitude and longitude](https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude)

Due to the projection, those real coordinates are probably 2D data variables. The `grid_mapping` attribute of a data variable makes a connection to another data variable defining the coordinate reference system (CRS) of those native coordinates. It should enable you to project the native coordinates into any other CRS, including the real 2D latitude and longitude coordinates. This is often useful for plotting, e.g., you can [tell cartopy how to correctly plot coastlines](https://scitools.org.uk/cartopy/docs/latest/tutorials/understanding_transform.html) for the CRS your data is defined in.

```{code-cell}
from cf_xarray.datasets import rotds
rotds
```

```{code-cell}
rotds.cf.grid_mappings
```
