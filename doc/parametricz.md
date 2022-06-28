---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
hide-toc: true
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

# Parametric Vertical Coordinates

`cf_xarray` supports decoding [parametric vertical coordinates](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-vertical-coordinate) encoded in the `formula_terms` attribute using {py:meth}`Dataset.cf.decode_vertical_coords`. Right now, only the two ocean s-coordinates and `ocean_sigma_coordinate` are supported, but support for the [rest](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord) should be easy to add (Pull Requests are very welcome!).

## Decoding parametric coordinates

```{code-cell}
from cf_xarray.datasets import romsds

romsds
```

Now we decode the vertical coordinates **in-place**. Note the new `z_rho` variable. `cf_xarray` sees that `s_rho` has a `formula_terms` attribute, looks up the right formula using `s_rho.attrs["standard_name"]` and computes a new vertical coordinate variable.

```{code-cell}
romsds.cf.decode_vertical_coords(outnames={'s_rho': 'z_rho'})  # adds new z_rho variable
romsds.z_rho
```

## Formula terms

To see whether decoding is possible, use the {py:attr}`Dataset.cf.formula_terms` attribute

```{code-cell}
romsds.cf.formula_terms
```
