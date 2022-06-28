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
import numpy as np
import xarray as xr
xr.set_options(display_expand_data=False)
```

# Selecting DataArrays

A second powerful feature of `cf_xarray` is the ability select DataArrays using special "CF names" like the "latitude", or "longitude" coordinate names, "X"  or "Y" axes names, oreven using the `standard_name` attribute if present.

To demonstrate this, let's load a few datasets

```{code-cell}
from cf_xarray.datasets import airds, anc, multiple, popds as pop
```

## By axis and coordinate name

Lets select the `"X"` axis on `airds`.

```{code-cell}
# identical to airds["lon"]
airds.cf["X"]
```

This works because `airds.lon.attrs` contains `axis: "X"`

```{code-cell}
airds.cf
```

## By standard name

The variable `airds.air` has `standard_name: "air_temperature"`, so we can use that to pull it out:

```{code-cell}
airds.cf["air_temperature"]
```

## By `cf_role`

`cf_xarray` supports identifying variables by the [`cf_role` attribute](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#discrete-sampling-geometries).

```{code-cell}
ds = xr.Dataset(
    {"temp": ("x", np.arange(10))},
    coords={"cast": ("x", np.arange(10), {"cf_role": "profile_id"})}
)
ds.cf["profile_id"]
```

## Associated variables

`.cf[key]` will return a DataArray or Dataset containing all variables associated with the `key` including ancillary variables and bounds variables (if possible).

In the following, note that the "ancillary variables" `q_error_limit` and `q_detection_limit` were also returned

```{code-cell}
anc.cf["specific_humidity"]
```

even though they are "data variables" and not "coordinate variables" in the original Dataset.

```{code-cell}
anc
```

## Selecting multiple variables

Sometimes a Dataset may contain multiple `X` or multiple `longitude` variables. In that case a simple `.cf["X"]` will raise an error. Instead follow Xarray convention and pass a  list `.cf[["X"]]` to receive a Dataset with all available `"X"` variables

```{code-cell}
multiple.cf[["X"]]
```

```{code-cell}
pop.cf[["longitude"]]
```

## Mixing names

cf_xarray aims to be as friendly as possible, so it is  possible to mix "CF names" and normal variable names. Here we select `UVEL` and `TEMP` by using the `standard_name` of `TEMP` (which is `sea_water_potential_temperature`)

```{code-cell}
pop.cf[["sea_water_potential_temperature", "UVEL"]]
```
