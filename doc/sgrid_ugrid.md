---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# SGRID / UGRID

## SGRID

`cf_xarray` can parse the attributes on the `grid_topology` variable to identify dimension names with axes names `X`, `Y`, `Z`.

Consider this representative dataset

```{code-cell}
from cf_xarray.datasets import sgrid_roms

sgrid_roms
```

A new `SGRID` section is added to the repr:

```{code-cell}
sgrid_roms.cf
```

### Topology variable

`cf_xarray` supports identifying `grid_topology` using the `cf_role` attribute.

```{code-cell}
sgrid_roms.cf["grid_topology"]
```

### Dimensions

Let's look at the repr again:

```{code-cell}
sgrid_roms.cf
```

Note that `xi_u`, `eta_u` were identified as `X`, `Y` axes even though
there is no data associated with them. So now the following will return `xi_u`

```{code-cell}
sgrid_roms.cf["X"]
```

```{tip}
The repr only shows variable names that can be used as `object[variable_name]`. That is why
only `xi_u`, `eta_u` are listed in the repr even though the attributes on the `grid_topology`
variable `grid` list many more dimension names.
```

## UGRID

### Topology variable

`cf_xarray` supports identifying  the `mesh_topology` variable using the `cf_role` attribute.

## More?

Further support for interpreting the SGRID and UGRID conventions can be added. Contributions are welcome!
