---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

# SGRID / UGRID

```{seealso}
1. [SGRID conventions](https://sgrid.github.io/sgrid/)
1. [UGRID conventions](http://ugrid-conventions.github.io/ugrid-conventions/)
```

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

### Coordinate variables

`cf_xarray` also follows the `node_coordinates`, `face_coordinates`,
`edge1_coordinates`, `edge2_coordinates`, and `volume_coordinates` attributes
on the `grid_topology` variable. When you select a data variable that
references a `grid_topology` via its `grid` attribute, the referenced
coordinate variables are pulled in alongside it:

```python
ds.cf[["u"]]  # includes `grid`, lon_psi/lat_psi, lon_rho/lat_rho, ...
```

Only names actually present in the dataset are propagated. For the
`DataArray` form (`ds.cf["u"]`) xarray only attaches coordinates whose
dimensions are compatible with the variable, so e.g. only `lon_u`/`lat_u`
appear as coords on `u`.

## UGRID

### Topology variable

`cf_xarray` supports identifying the `mesh_topology` variable using the `cf_role` attribute.

## More?

Further support for interpreting the SGRID and UGRID conventions can be added. Contributions are welcome!
