# CF and xarray terminology

CF describes the meaning of data and its metadata. Xarray describes how arrays
are organized in memory. The terms overlap, but are not interchangeable.

## Coordinates

| CF-netCDF term                                       | Typical xarray representation                                                           |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Coordinate variable, such as `lat(lat)`              | Dimension coordinate: a one-dimensional coordinate with the same name as its dimension. |
| Auxiliary coordinate variable, such as `lat(y, x)`   | Non-dimension coordinate: a coordinate whose name is not a dimension name.              |
| Scalar coordinate variable, such as a single height  | Scalar coordinate: a coordinate with no dimensions.                                     |
| Boundary variable, such as `lat_bounds(lat, bounds)` | A variable referenced by the coordinate's `bounds` attribute; see [Bounds](bounds.md).  |

These are correspondences, not a compliance test. In particular, a CF coordinate
variable must be numeric, strictly increasing or decreasing, and contain no
missing values. Xarray can represent dimension coordinates that do not satisfy
those requirements. A dimension can also exist without a coordinate variable.

Whether an xarray coordinate has an **index** is a separate question from its
CF meaning. An index enables operations such as label-based selection and
alignment. Identifying a coordinate with cf-xarray does not create an index for
it.

## Names, axes, and physical coordinates

A variable's name is its key in a Dataset. Its `standard_name` attribute describes
the physical quantity. An `axis` attribute identifies a coordinate as `X`, `Y`,
`Z`, or `T`; it is not the same as an xarray dimension name.

For example, the following Dataset has dimensions `y` and `x`, dimension
coordinates with those names, and two-dimensional auxiliary latitude and
longitude coordinates:

```python
import xarray as xr
import cf_xarray  # registers the .cf accessor

ds = xr.Dataset(
    data_vars={
        "temp": (
            ("y", "x"),
            [[280.0, 281.0], [282.0, 283.0]],
            {
                "standard_name": "air_temperature",
                "units": "K",
                "coordinates": "lat lon",
            },
        ),
    },
    coords={
        "x": ("x", [0, 1], {"axis": "X"}),
        "y": ("y", [0, 1], {"axis": "Y"}),
        "lat": (
            ("y", "x"),
            [[50.0, 50.1], [51.0, 51.1]],
            {"standard_name": "latitude", "units": "degrees_north"},
        ),
        "lon": (
            ("y", "x"),
            [[0.0, 1.0], [0.1, 1.1]],
            {"standard_name": "longitude", "units": "degrees_east"},
        ),
    },
)

ds.cf.axes  # {"X": ["x"], "Y": ["y"]}
ds.cf.coordinates  # {"longitude": ["lon"], "latitude": ["lat"]}
ds.cf["air_temperature"]  # selects temp using its standard_name
ds.cf["latitude"]  # selects lat, not the y dimension coordinate
```

The `coordinates` attribute associates `lat` and `lon` with `temp` in CF metadata.
Here, `Y` and `latitude` identify different variables. cf-xarray interprets
metadata to find them; it does not rename the variables or transform the grid.
See [Axes and Coordinates](coord_axes.md) for the matching criteria.

## Further reading

- [Xarray terminology](https://docs.xarray.dev/en/stable/user-guide/terminology.html)
- [CF 1.12 terminology](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#terminology)
- [CF coordinate systems](https://cfconventions.org/Data/cf-conventions/cf-conventions-1.12/cf-conventions.html#coordinate-system)
