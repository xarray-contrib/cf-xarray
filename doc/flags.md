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

(flags)=

# Flag Variables

`cf_xarray` has some support for [flag variables](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags).

```{code-cell}
import cf_xarray
import xarray as xr

da = xr.DataArray(
    [1, 2, 1, 1, 2, 2, 3, 3, 3, 3],
    dims=("time",),
    attrs={
        "flag_values": [1, 2, 3],
        "flag_meanings": "atlantic_ocean pacific_ocean indian_ocean",
        "standard_name": "region",
    },
  name="region",
)
da.cf
```

Now you can perform meaningful boolean comparisons that take advantage of the `flag_meanings` attribute:

```{code-cell}
# compare to da == 1
da.cf == "atlantic_ocean"
```

Similarly with membership tests using {py:meth}`DataArray.cf.isin`

```{code-cell}
# compare to da.isin([2, 3])
da.cf.isin(["indian_ocean", "pacific_ocean"])
```

You can also check whether a DataArray has the appropriate attributes to be recognized as a flag variable using {py:meth}`DataArray.cf.is_flag_variable`

```{code-cell}
da.cf.is_flag_variable
```

```{tip}
`cf_xarray` does not support [flag masks](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags) yet but a Pull Request to add this functionality is very welcome!
```
