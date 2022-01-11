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
mpl.rcParams["figure.dpi"] = 140
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
