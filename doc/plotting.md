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
import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 120
mpl.rcParams["font.size"] = 9
xr.set_options(display_expand_data=False)
```

# Plotting

Plotting is where `cf_xarray` really shines in our biased opinion.

```{code-cell}
from cf_xarray.datasets import airds

air = airds.air
air.cf
```

```{tip}
Only ``DataArray.plot`` is currently supported.
```

## Using CF standard names

Note the use of `"latitude"` and `"longitude"` (or `"X"` and `"Y"`) in the following as a "standard" substitute for the dataset-specific `"lat"` and `"lon"` variables.

```{code-cell}
air.isel(time=0).cf.plot(x="X", y="Y")
```

```{code-cell}
air.cf.isel(T=1, Y=[0, 1, 2]).cf.plot(x="longitude", hue="latitude")
```

```{code-cell}
air.cf.plot(x="longitude", y="latitude", col="T")
```

## Automatic axis placement

Now let's create a fake dataset representing a `(x,z)` cross-section of the ocean. The vertical coordinate here is "pressure" which increases downwards.
We follow CF conventions and mark `pres` as `axis: Z, positive: "down"` to indicate these characeristics.

```{code-cell}
import matplotlib as mpl
import numpy as np
import xarray as xr

ds = xr.Dataset(
    coords={
        "pres": ("pres", np.arange(20), {"axis": "Z", "positive": "down"}),
        "x": ("x", np.arange(50), {"axis": "X"})
    }
)
ds["temp"] = 20 * xr.ones_like(ds.x) *  np.exp(- ds.pres / 30)
ds.temp.cf
```

The default xarray plot has some deficiencies

```{code-cell}
ds.temp.plot(cmap=mpl.cm.RdBu_r)
```

cf_xarray can interpret attributes to make two decisions:

1. That `pres` should be the Y-Axis
1. Since `pres` increases downwards (`positive: "down"`), the axis should be reversed so that low pressure is at the top of the plot.
   Now we have a more physically meaningful figure where warmer water is at the top of the water column!

```{code-cell}
ds.temp.cf.plot(cmap=mpl.cm.RdBu_r)
```
