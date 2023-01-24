---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{eval-rst}
.. currentmodule:: xarray
```

# Grid Mappings

See

1. {py:attr}`Dataset.cf.grid_mappings`,
1. {py:func}`Dataset.cf.get_grid_mapping`,
1. {py:func}`Dataset.cf.get_grid_mapping_name`

`cf_xarray` understands the concept of coordinate projections using the [grid_mapping](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#grid-mappings-and-projections) attribute convention. For example, the dataset might contain two sets of coordinates:

- native coordinates in which the data is defined, e.g., regular 1D coordinates
- projected coordinates which probably denote some "real" coordinates in [latitude and longitude](https://en.wikipedia.org/wiki/Geographic_coordinate_system#Latitude_and_longitude)

Due to the projection, those real coordinates are probably 2D data variables. The `grid_mapping` attribute of a data variable makes a connection to another data variable defining the coordinate reference system (CRS) of those native coordinates. It should enable you to project the native coordinates into any other CRS, including the real 2D latitude and longitude coordinates. This is often useful for plotting, e.g., you can [tell cartopy how to correctly plot coastlines](https://scitools.org.uk/cartopy/docs/latest/tutorials/understanding_transform.html) for the CRS your data is defined in.

To access `grid_mapping` attributes, consider this example:

```{code-cell}
from cf_xarray.datasets import rotds
rotds
```

The related grid mappings can be accessed using:

```{code-cell}
rotds.cf.grid_mappings
```

## Use `grid_mapping` in projections

The grid mapping information use very useful in projections, e.g., for plotting. [pyproj](https://pyproj4.github.io/pyproj/stable/api/crs/crs.html#pyproj.crs.CRS.from_cf) understands CF conventions right away, e.g.

```python
from pyproj import CRS

CRS.from_cf(rotds.cf.get_grid_mapping("temp").attrs)
```

gives you more details on the projection:

```
<Derived Geographic 2D CRS: {"$schema": "https://proj.org/schemas/v0.2/projjso ...>
Name: undefined
Axis Info [ellipsoidal]:
- lon[east]: Longitude (degree)
- lat[north]: Latitude (degree)
Area of Use:
- undefined
Coordinate Operation:
- name: Pole rotation (netCDF CF convention)
- method: Pole rotation (netCDF CF convention)
Datum: World Geodetic System 1984
- Ellipsoid: WGS 84
- Prime Meridian: Greenwich
```

For use in cartopy, there is some more overhead due to [this issue](https://github.com/SciTools/cartopy/issues/2099). So you should select the right cartopy CRS and just feed in the grid mapping info:

```python
from cartopy import crs as ccrs

grid_mapping = rotds.cf.get_grid_mapping("temp")
pole_latitude = grid_mapping.grid_north_pole_latitude
pole_longitude = grid_mapping.grid_north_pole_longitude

ccrs.RotatedPole(pole_longitude, pole_latitude)
```

![cartopy rotated pole projection](cartopy_rotated_pole.png)
