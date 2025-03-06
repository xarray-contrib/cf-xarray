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

```{seealso}
1. [CF conventions on flag variables and masks](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#flags)
```

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

## GroupBy

Flag variables, such as that above, are naturally used for GroupBy operations.
cf-xarray provides a `FlagGrouper` that understands the `flag_meanings` and `flag_values` attributes.

Let's load an example dataset where the `flag_var` array has the needed attributes.

```{code-cell}
import cf_xarray as cfxr
import numpy as np

from cf_xarray.datasets import flag_excl

ds = flag_excl.to_dataset().set_coords('flag_var')
ds["foo"] = ("time", np.arange(8))
ds.flag_var
```

Now use the :py:class:`~cf_xarray.groupers.FlagGrouper` to group by this flag variable:

```{code-cell}
from cf_xarray.groupers import FlagGrouper

ds.groupby(flag_var=FlagGrouper()).mean()
```

Note how the output coordinate has the values from `flag_meanings`!

```{seealso}
See the Xarray docs on using [Grouper objects](https://docs.xarray.dev/en/stable/user-guide/groupby.html#grouper-objects).
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
flag_indep.cf
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
flag_mix.cf
```

```{code-cell}
flag_mix.cf == 'flag_4'
```
