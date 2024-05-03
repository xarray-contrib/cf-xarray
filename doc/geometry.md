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

# Geometries

```seealso
1. [The CF conventions on Geometries](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#geometries)
1. {py:func}`cf_xarray.shapely_to_cf`
1. {py:func}`cf_xarray.cf_to_shapely`
```

In order to support vectors as well as arrays `cf_xarray` can convert between shapely objects
and CF-compliant representations of those geometries.

Let's start by creating an xarray object containing some shapely geometries. This example uses
a `xr.DataArray` but these functions also work with a `xr.Dataset` where one of the data variables
contains an array of shapes.

```{code-cell}
import cf_xarray as cfxr
import xarray as xr

from shapely.geometry import MultiPoint, Point

da = xr.DataArray(
    [
        MultiPoint([(1.0, 2.0), (2.0, 3.0)]),
        Point(3.0, 4.0),
        Point(4.0, 5.0),
        Point(3.0, 4.0),
    ],
    dims=("index",),
    name="geometry"
)
```

```{warning}
`cf_xarray` does not support handle multiple types of shapes (Point, Line, Polygon) in one
`xr.DataArray`, but multipart geometries are supported and can be mixed with single-part
geometries of the same type.
```

Now we can take that `xr.DataArray` containing shapely geometries and convert it to cf:

```{code-cell}
ds_cf = cfxr.shapely_to_cf(da)
ds_cf
```

This function returns a `xr.Dataset` containing the CF fields needed to reconstruct the
geometries. In particular there are:

- `'x'`, `'y'` : the node coordinates
- `'crd_x'`, `'crd_y'` : the feature coordinates (might have different names if `grid_mapping` is available).
- `'node_count'` : The number of nodes per feature. Always present for Lines and Polygons. For
  Points: only present if there are multipart geometries.
- `'part_node_count'` : The number of nodes per individual geometry. Only for Lines with multipart
  geometries and for Polygons with multipart geometries or holes.
- `'interior_ring'` : Integer boolean indicating whether ring is interior or exterior. Only for
  Polygons with holes.
- `'geometry_container`' : Empty variable with attributes describing the geometry type.

Here are the attributes on `geometry_container`. This pattern mimics the convention of
specifying spatial reference information in the attrs of the empty array `spatial_ref`.

```{code-cell}
ds_cf.geometry_container.attrs
```

```{note}
Z axis is not yet supported for any shapes.
```

This `xr.Dataset` can be converted back into a `xr.DataArray` of shapely geometries:

```{code-cell}
cfxr.cf_to_shapely(ds_cf)
```

This conversion adds coordinates that aren't in the `xr.DataArray` that we started with.
By default these are called `crd_x` and `crd_y` unless `grid_mapping` is specified.

## Gotchas

For MultiPolygons with holes the CF notation is slightly ambiguous on which hole is associated
with which polygon. This is problematic because shapely stores holes within the polygon
object that they are associated with. `cf_xarray` assumes that the the shapes are interleaved
such that the holes (interior rings) are associated with the exteriors (exterior rings) that
immediately precede them.
