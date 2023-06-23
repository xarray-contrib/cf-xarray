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

`cf_xarray` has some support for [flag variables](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags), including flag masks.

## Flag Values

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

## Flag Masks

```{warning}
Interpreting flag masks is very lightly tested.
Please double-check the results and open an issue
or pull request to suggest improvements.
```

Load an example dataset:

```{code-cell}
from cf_xarray.datasets import flag_indep

flag_indep
```

```{code-cell}
flag_indep.cf.is_flag_variable
```

```{code-cell}
flag_indep.cf == "flag_1"
```

And `isin`

```{code-cell}
flag_indep.cf.isin(["flag_1", "flag_3"])
```

## Combined masks and values

```{warning}
Interpreting a mix of flag masks and flag values
is very lightly tested. Please double-check the results
and open an issue or pull request to suggest improvements.
```

Load an example dataset:

```{code-cell}
from cf_xarray.datasets import flag_mix

flag_mix
```

```{code-cell}
flag_mix.cf.is_flag_variable
```

```{code-cell}
flag_mix.cf == 'flag_4'
```
