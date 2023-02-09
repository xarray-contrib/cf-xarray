---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# SGRID / UGRID

## Topology variable

`cf_xarray` support identifying either the `mesh_topology` (UGRID) or `grid_topology` (SGRID) variables using the `cf_role` attribute.

## SGRID

`cf_xarray` can parse the attributes on the `grid_topology` variable to identify dimension names with axes names `X`, `Y`, `Z`.

Consider this representative dataset

```{code-cell}
from cf_xarray.datasets import sgrid_roms

sgrid_roms
```

Note that `xi_u`, `eta_u` are identified as `X`, `Y` axes below even though
there is no data associated with them in the repr above.

```{code-cell}
sgrid_roms.cf
```

So now the following will return `xi_u`

```{code-cell}
sgrid_roms.cf["X"]
```

## More?

Further support for interpreting the SGRID and UGRID conventions can be added. Contributions are welcome!
