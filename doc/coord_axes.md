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

```{code-cell}
---
tags: [remove-cell]
---
import xarray as xr
xr.set_options(display_expand_data=False)
```

# Axes and Coordinates

One powerful feature of `cf_xarray` is the ability to refer to named dimensions by standard `axis` or `coordinate` names in Dataset or DataArray methods.

For example, one can call `ds.cf.mean("latitude")` instead of `ds.mean("lat")`

```{code-cell}
from cf_xarray.datasets import airds

# identical to airds.mean("lat")
airds.cf.mean("latitude")
```

```{tip}
Most xarray methods are wrapped by cf-xarray. Simply access them as `DataArray.cf.method(dim="latitude")` for example  and try it! If something does not work, please raise an issue.
```

(coordinate-criteria)=

## Coordinate Criteria

How does this work? `cf_xarray` has an internal table of criteria (mostly copied from MetPy) that lets it identify specific coordinate names `"latitude", "longitude", "vertical", "time"`.

```{tip}
See {ref}`custom_criteria` to find out how to define your own custom criteria.
```

This table lists these internal criteria

```{eval-rst}
.. csv-table::
   :file: _build/csv/coords_criteria.csv
   :header-rows: 1
   :stub-columns: 1
```

Any DataArray that has `standard_name: "latitude"` or `_CoordinateAxisType: "Lat"` or `"units": "degrees_north"` in its `attrs` will be identified as the `"latitude"` variable by cf-xarray.  Similarly for other coordinate names.

## Axis Names

Similar criteria exist for the concept of "axes".

```{eval-rst}
.. csv-table::
   :file: _build/csv/axes_criteria.csv
   :header-rows: 1
   :stub-columns: 1
```

## `.axes` and  `.coordinates` properties

Alternatively use the special properties {py:attr}`DataArray.cf.axes` or {py:attr}`DataArray.cf.coordinates` to access the variable names. These properties return dictionaries that map "CF names" to a list of variable names. Note that a list is always returned even if only one variable name matches the name `"latitude"` (for example).

```{code-cell}
airds.cf.axes
```

```{code-cell}
airds.cf.coordinates
```

## Axes or Coordinate?

TODO describe latitude vs Y; longitude vs X; vertical vs Z

## Checking presence of axis or coordinate

Note that a given "CF name" is only present if there is at least one variable that can be identified with that name. The `airds` dataset has no `"vertical"` coordinate or `"Z"` axis, so those keys are not present. So to check whether a `"vertical"` coordinate or `"Z"` axis is present, one can

```{code-cell}
"Z" in airds.cf.axes
```

```{code-cell}
"vertical" in airds.cf.coordinates
```

Or one can check the dataset as a whole:

```{code-cell}
"Z" in airds.cf
```

## Using the repr

It is always useful to check the variables identified by cf-xarray using the `repr`

```{code-cell}
airds.cf
```
