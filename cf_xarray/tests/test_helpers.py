from numpy.testing import assert_array_equal
from xarray.testing import assert_equal

import cf_xarray as cfxr  # noqa

from ..datasets import airds, mollwds

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = None


def test_bounds_to_vertices():
    # 1D case
    ds = airds.cf.add_bounds(["lon", "lat", "time"])
    lat_c = cfxr.bounds_to_vertices(ds.lat_bounds, bounds_dim="bounds")
    assert_array_equal(ds.lat.values + 1.25, lat_c.values[:-1])

    # 2D case, CF- order
    lat_ccw = cfxr.bounds_to_vertices(
        mollwds.lat_bounds, bounds_dim="bounds", order="counterclockwise"
    )
    lat_no = cfxr.bounds_to_vertices(
        mollwds.lat_bounds, bounds_dim="bounds", order=None
    )
    assert_equal(mollwds.lat_vertices, lat_ccw)
    assert_equal(lat_no, lat_ccw)

    # Transposing the array changes the bounds direction
    ds = mollwds.transpose("bounds", "y", "x", "y_vertices", "x_vertices")
    lon_c = cfxr.bounds_to_vertices(
        ds.lon_bounds, bounds_dim="bounds", order="clockwise"
    )
    lon_c2 = cfxr.bounds_to_vertices(ds.lon_bounds, bounds_dim="bounds", order=None)
    assert_equal(ds.lon_vertices, lon_c)
    assert_equal(ds.lon_vertices, lon_c2)

    # Preserves dask-backed arrays
    if DaskArray is not None:
        lon_bounds = ds.lon_bounds.chunk()
        lon_c = cfxr.bounds_to_vertices(
            lon_bounds, bounds_dim="bounds", order="clockwise"
        )
        assert isinstance(lon_c.data, DaskArray)


def test_vertices_to_bounds():
    # 1D case
    ds = airds.cf.add_bounds(["lon", "lat", "time"])
    lat_c = cfxr.bounds_to_vertices(ds.lat_bounds, bounds_dim="bounds")
    lat_b = cfxr.vertices_to_bounds(lat_c, out_dims=("bounds", "lat"))
    assert_array_equal(ds.lat_bounds, lat_b)

    # Datetime
    time_c = cfxr.bounds_to_vertices(ds.time_bounds, bounds_dim="bounds")
    time_b = cfxr.vertices_to_bounds(time_c, out_dims=("bounds", "time"))
    assert_array_equal(ds.time_bounds, time_b)

    # 2D case
    lon_b = cfxr.vertices_to_bounds(mollwds.lon_vertices, out_dims=("bounds", "x", "y"))
    assert_array_equal(mollwds.lon_bounds, lon_b)
