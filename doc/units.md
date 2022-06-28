---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
hide-toc: true
---

# Units

The xarray ecosystem supports unit-aware arrays using  [pint](https://pint.readthedocs.io) and [pint-xarray](https://pint-xarray.readthedocs.io). Some changes are required to make these packages work well with [UDUNITS format recommended by the CF conventions](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units).

`cf_xarray` makes those recommended changes when you `import cf_xarray.units`. These changes allow pint to parse and format UDUNIT units strings, and add several custom units like `degrees_north`, `psu` etc.

## Formatting units

For now, only the short format using [symbols](https://www.unidata.ucar.edu/software/udunits/udunits-2.2.28/udunits2lib.html#Syntax) is supported:

```{code-cell}
from pint import application_registry as ureg
import cf_xarray.units

u = ureg.Unit("m ** 3 / s ** 2")
f"{u:~cf}"
```
