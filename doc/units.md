---
jupytext:
  text_representation:
    format_name: myst    
kernelspec:
  display_name: Python 3
  name: python3
---

# Units

The xarray ecosystem supports unit-aware arrays using  [pint](https://pint.readthedocs.io) and [pint-xarray](https://pint-xarray.readthedocs.io). Some changes are required to make these packages work well with [UDUNITS format recommended by the CF conventions](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#units).

`cf_xarray` makes those recommended changes when you `import cf_xarray.units`. These changes allow pint to parse UDUNIT units strings, and add several custom units like `degrees_north`, `psu` etc.
