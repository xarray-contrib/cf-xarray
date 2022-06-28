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
import cf_xarray
import numpy as np
import xarray as xr
xr.set_options(display_expand_data=False)
```

# Discrete Sampling Geometries

`cf_xarray` supports identifying variables by the [`cf_role` attribute](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html#discrete-sampling-geometries).

```{code-cell}
ds = xr.Dataset(
    {"temp": ("x", np.arange(10))},
    coords={"cast": ("x", np.arange(10), {"cf_role": "profile_id"})}
)
ds.cf
```

Access `"cast"` using it's `cf_role`

```{code-cell}
ds.cf["profile_id"]
```

Find all `cf_role` variables using {py:attr}`Dataset.cf.cf_roles` and {py:attr}`DataArray.cf.cf_roles`

```{code-cell}
ds.cf.cf_roles
```
