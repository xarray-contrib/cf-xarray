---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{eval-rst}
.. currentmodule:: cf_xarray
```

```{code-cell}
---
tags: [remove-cell]
---
import xarray as xr
xr.set_options(display_expand_data=False)
```

(custom_criteria)=

# Custom Criteria

Fundamentally, cf_xarray uses rules or "criteria" to interpret user input using the
attributes of an Xarray object (`.attrs`). These criteria are simple dictionaries. For example, here are the criteria used for identifying a "latitude" variable:

```python
coordinate_criteria = {
    "latitude": {
        "standard_name": ("latitude",),
        "units": (
            "degree_north",
            "degree_N",
            "degreeN",
            "degrees_north",
            "degrees_N",
            "degreesN",
        ),
        "_CoordinateAxisType": ("Lat",),
    },
}
```

This dictionary maps the user input (`"latitude"`) to another dictionary which in turn maps an attribute name to a tuple of acceptable values for that attribute. So any variable with either `standard_name: latitude` or `_CoordinateAxisType: Lat_` or any of the `unit`s listed above will match the user-input `"latitude"`.

cf_xarray lets you provide your own custom criteria in addition to those built-in. Here's an example:

```{code-cell}
import cf_xarray as cfxr
import numpy as np
import xarray as xr

ds = xr.Dataset({
    "salt1": ("x", np.arange(10), {"standard_name": "sea_water_salinity"}),
    "salt2": ("x", np.arange(10), {"standard_name": "sea_water_practical_salinity"}),
})

# first define our criteria
salt_criteria = {
    "sea_water_salinity": {
        "standard_name": "sea_water_salinity|sea_water_practical_salinity"
        }
}
```

Now we apply our custom criteria temporarily using {py:func}`set_options` as a context manager. The following sets `"sea_water_salinity"` as an alias for variables that have either `"sea_water_salinity"` or `"sea_water_practical_salinity"` (note the use of regular expressions as a value). Here's how that works in practice

```{code-cell}
with cfxr.set_options(custom_criteria=salt_criteria):
    salty = ds.cf[["sea_water_salinity"]]
salty
```

Note that `salty` contains both `salt1` and `salt2`. Without setting these criteria, we  would only get `salt1` by default

```{code-cell}
ds.cf[["sea_water_salinity"]]
```

We can also use {py:func}`set_options` to set the criteria globally.

```{code-cell}
cfxr.set_options(custom_criteria=salt_criteria)
ds.cf[["sea_water_salinity"]]
```

Again we get back both `salt1` and `salt2`. To limit side effects of setting criteria globally, we recommend that you use `set_options` as a context manager.

```{tip}
To reset your custom criteria use `cfxr.set_options(custom_criteria=())`
```

You can also match on the variable name, though be careful!

```{code-cell}
salt_criteria = {
    "salinity": {"name": "salt*"}
}
cfxr.set_options(custom_criteria=salt_criteria)

ds.cf[["salinity"]]
```

## More complex matches with `regex`

Here is an example of a more complicated custom criteria, which requires the package [`regex`](https://github.com/mrabarnett/mrab-regex) to be installed since a behavior (allowing global flags like "(?i)" for matching case insensitive) was recently deprecated in the `re` package. The custom criteria, called "vocab", matches – case insensitive – to the variable alias "sea_ice_u" a variable whose name includes "sea" and "ice" and "u" but not "qc" or "status", or "sea" and "ice" and "x" and "vel" but not "qc" or "status".

```{code-cell}
import cf_xarray as cfxr
import xarray as xr

vocab = {"sea_ice_u": {"name": "(?i)^(?!.*(qc|status))(?=.*sea)(?=.*ice)(?=.*u)|(?i)^(?!.*(qc|status))(?=.*sea)(?=.*ice)(?=.*x)(?=.*vel)"}}
ds = xr.Dataset()
ds["sea_ice_velocity_x"] = [0,1,2]

with cfxr.set_options(custom_criteria=vocab):
    seaiceu = ds.cf["sea_ice_u"]
seaiceu
```
