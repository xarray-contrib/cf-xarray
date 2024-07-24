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

```{seealso}
1. [The CF conventions on Geometries](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.11/cf-conventions.html#geometries)
1. {py:attr}`Dataset.cf.geometries`
```

```{eval-rst}
.. currentmodule:: cf_xarray
```

First read an example dataset with CF-encoded geometries

```{code-cell}
import cf_xarray as cfxr
import cf_xarray.datasets
import xarray as xr

ds = cfxr.datasets.encoded_point_dataset()
ds
```

The {py:attr}`Dataset.cf.geometries` property will yield a mapping from geometry type to geometry container variable name.

```{code-cell}
ds.cf.geometries
```

The `"geometry"` name is special, and will return the geometry *container* present in the dataset

```{code-cell}
ds.cf["geometry"]
```

Request all variables needed to represent a geometry as a Dataset using the geometry type as key.

```{code-cell}
ds.cf[["point"]]
```

You *must* request a Dataset as return type, that is provide the list `["point]`, because the CF conventions encode geometries across multiple variables with dimensions that are not present on all variables. Xarray's data model does *not* allow representing such a collection of variables as a DataArray.

## Encoding & decoding

`cf_xarray` can convert between vector geometries represented as shapely objects
and CF-compliant array representations of those geometries.

Let's start by creating an xarray object containing some shapely geometries. This example uses
a `xr.DataArray` but these functions also work with a `xr.Dataset` where one of the data variables
contains an array of shapes.

```{warning}
`cf_xarray` does not support handle multiple types of shapes (Point, Line, Polygon) in one
`xr.DataArray`, but multipart geometries are supported and can be mixed with single-part
geometries of the same type.
```

`cf-xarray` provides {py:func}`geometry.encode_geometries` and {py:func}`geometry.decode_geometries` to
encode and decode xarray Datasets to/from a CF-compliant form that can be written to any array storage format.

For example, here is a Dataset with shapely geometries

```{code-cell}
ds = cfxr.datasets.point_dataset()
ds
```

Encode with the CF-conventions

```{code-cell}
encoded = cfxr.geometry.encode_geometries(ds)
encoded
```

This dataset can then be written to any format supported by Xarray.
To decode back to shapely geometries, reverse the process using {py:func}`geometry.decode_geometries`

```{code-cell}
decoded = cfxr.geometry.decode_geometries(encoded)
ds.identical(decoded)
```

### Limitations

The following limitations can be relaxed in the future. PRs welcome!

1. cf-xarray uses `"geometry_container"` as the name for the geometry variable always. If there are multiple geometry variables then `"geometry_N"`is used where N is an integer >= 0. cf-xarray behaves similarly for all associated geometry variables names: i.e. `"node"`, `"node_count"`, `"part_node_count"`, `"part"`, `"interior_ring"`. `"x"`, `"y"` (with suffixes if needed) are always the node coordinate variable names, and `"crd_x"`, `"crd_y"` are the nominal X, Y coordinate locations. None of this is configurable at the moment.
1. CF xarray will not set the `"geometry"` attribute that links a variable to a geometry by default unless the geometry variable is a dimension coordinate for that variable. This heuristic works OK for vector data cubes (e.g. [xvec](https://xvec.readthedocs.io/en/stable/)). You should set the `"geometry"` attribute manually otherwise. Suggestions for better behaviour here are very welcome.

## Lower-level conversions

Encoding a single DataArray is possible using {py:func}`geometry.shapely_to_cf`.

```{code-cell}
da = ds["geometry"]
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
By default these are called `'crd_x'` and `'crd_y'` unless `grid_mapping` is specified.

## Gotchas

For MultiPolygons with holes the CF notation is slightly ambiguous on which hole is associated
with which polygon. This is problematic because shapely stores holes within the polygon
object that they are associated with. `cf_xarray` assumes that the shapes are interleaved
such that the holes (interior rings) are associated with the exteriors (exterior rings) that
immediately precede them.
