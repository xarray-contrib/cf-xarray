import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal
from xarray.testing import assert_equal

import cf_xarray as cfxr  # noqa

from ..datasets import airds, mollwds, rotds
from . import requires_cftime

try:
    from dask.array import Array as DaskArray
except ImportError:
    DaskArray = None  # type: ignore[assignment, misc]


def test_bounds_to_vertices() -> None:
    # 1D case (stricly monotonic, descending bounds)
    ds = airds.cf.add_bounds(["lon", "lat", "time"])
    lat_c = cfxr.bounds_to_vertices(ds.lat_bounds, bounds_dim="bounds")
    assert_array_equal(ds.lat.values + 1.25, lat_c.values[:-1])

    # 2D case
    lat_ccw = cfxr.bounds_to_vertices(
        mollwds.lat_bounds, bounds_dim="bounds", order="counterclockwise"
    )
    lat_no = cfxr.bounds_to_vertices(
        mollwds.lat_bounds, bounds_dim="bounds", order=None
    )
    assert_equal(mollwds.lat_vertices, lat_ccw)
    assert_equal(lat_no, lat_ccw)

    # 2D case with precision issues, check if CF- order is "detected" correctly
    lon_ccw = cfxr.bounds_to_vertices(
        rotds.lon_bounds, bounds_dim="bounds", order="counterclockwise"
    )
    lon_no = cfxr.bounds_to_vertices(rotds.lon_bounds, bounds_dim="bounds", order=None)
    assert_equal(lon_no, lon_ccw)

    # 2D case (monotonicly increasing coords, non-monotonic bounds)
    bounds_2d_desc = xr.DataArray(
        [[50.5, 50.0], [51.0, 50.5], [51.0, 50.5], [52.0, 51.5], [52.5, 52.0]],
        dims=("lat", "bounds"),
        coords={"lat": [50.75, 50.75, 51.25, 51.75, 52.25]},
    )
    expected_vertices_2d_desc = xr.DataArray(
        [50.0, 50.5, 50.5, 51.5, 52.0, 52.5],
        dims=["lat_vertices"],
    )
    vertices_2d_desc = cfxr.bounds_to_vertices(bounds_2d_desc, bounds_dim="bounds")
    assert_equal(expected_vertices_2d_desc, vertices_2d_desc)

    # 3D case (non-monotonic bounds, monotonicly increasing coords)
    bounds_3d = xr.DataArray(
        [
            [
                [50.0, 50.5],
                [50.5, 51.0],
                [51.0, 51.5],
                [51.5, 52.0],
                [52.0, 52.5],
            ],
            [
                [60.0, 60.5],
                [60.5, 61.0],
                [61.0, 61.5],
                [61.5, 62.0],
                [62.0, 62.5],
            ],
        ],
        dims=("extra", "lat", "bounds"),
        coords={
            "extra": [0, 1],
            "lat": [0, 1, 2, 3, 4],
            "bounds": [0, 1],
        },
    )
    expected_vertices_3d = xr.DataArray(
        [
            [50.0, 50.5, 51.0, 51.5, 52.0, 52.5],
            [60.0, 60.5, 61.0, 61.5, 62.0, 62.5],
        ],
        dims=("extra", "lat_vertices"),
        coords={
            "extra": [0, 1],
        },
    )
    vertices_3d = cfxr.bounds_to_vertices(
        bounds_3d, bounds_dim="bounds", core_dims=["lat"]
    )
    assert_equal(vertices_3d, expected_vertices_3d)

    # Transposing the array changes the bounds direction
    ds = mollwds.transpose("x", "y", "x_vertices", "y_vertices", "bounds")
    lon_cw = cfxr.bounds_to_vertices(
        ds.lon_bounds, bounds_dim="bounds", order="clockwise"
    )
    lon_no2 = cfxr.bounds_to_vertices(ds.lon_bounds, bounds_dim="bounds", order=None)
    assert_equal(ds.lon_vertices, lon_cw)
    assert_equal(ds.lon_vertices, lon_no2)

    # Preserves dask-backed arrays
    if DaskArray is not None:
        lon_bounds = ds.lon_bounds.chunk()
        lon_c = cfxr.bounds_to_vertices(
            lon_bounds, bounds_dim="bounds", order="clockwise"
        )
        assert isinstance(lon_c.data, DaskArray)


def test_vertices_to_bounds() -> None:
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


@pytest.mark.parametrize("ndim", [1, 2])
@pytest.mark.parametrize("chunked", [False, True])
def test_vertices_to_bounds_broadcast_dimensions(ndim, chunked):
    core_dims = ["a", "b"][:ndim]
    shape = (2,) + (4,) * ndim
    vertices = xr.DataArray(
        np.arange(np.prod(shape)).reshape(shape),
        dims=["time", *core_dims],
        coords={"time": [10, 20]},
    ).transpose(*core_dims, "time")
    if chunked:
        pytest.importorskip("dask.array")
        vertices = vertices.chunk({"time": 1})
    out_dims = ["bounds", "x", "y"][: ndim + 1]

    result = cfxr.vertices_to_bounds(vertices, out_dims=out_dims, core_dims=core_dims)

    expected = xr.concat(
        [
            cfxr.vertices_to_bounds(vertices.isel(time=i), out_dims=out_dims)
            for i in range(2)
        ],
        dim=vertices.time,
    )
    assert_equal(result, expected)
    assert result.dtype == vertices.dtype
    assert result.dims == ("time", *out_dims[1:], "bounds")
    if chunked:
        assert isinstance(result.data, DaskArray)
        assert result.chunks[0] == (1, 1)


def test_vertices_to_bounds_same_core_dimension_name():
    vertices = xr.DataArray(np.arange(12).reshape(3, 4), dims=("time", "depth"))

    result = cfxr.vertices_to_bounds(
        vertices, out_dims=("bounds", "depth"), core_dims=["depth"]
    )

    assert result.sizes == {"time": 3, "depth": 3, "bounds": 2}
    assert_array_equal(result[..., 0], vertices[:, :-1])
    assert_array_equal(result[..., 1], vertices[:, 1:])


def test_vertices_to_bounds_datetime_broadcast():
    dates = np.arange("2000-01-01", "2000-01-05", dtype="datetime64[D]")
    vertices = xr.DataArray(dates, dims="edge").expand_dims(time=[1, 2])

    result = cfxr.vertices_to_bounds(
        vertices, out_dims=("bounds", "depth"), core_dims=["edge"]
    )

    assert result.dtype == vertices.dtype
    assert_array_equal(result[0, :, 0], dates[:-1])
    assert_array_equal(result[0, :, 1], dates[1:])


def test_vertices_to_bounds_invalid_dimensions():
    vertices = xr.DataArray(np.ones((2, 3, 4)), dims=("a", "b", "c"))
    with pytest.raises(ValueError, match="vertices format not understood"):
        cfxr.vertices_to_bounds(vertices)
    with pytest.raises(ValueError, match="out_dims"):
        cfxr.vertices_to_bounds(vertices, out_dims=["bounds"], core_dims=["c"])


@requires_cftime
def test_bounds_to_vertices_cftime() -> None:
    import cftime

    # Create cftime objects for monthly bounds
    periods = 3
    # start = cftime.DatetimeGregorian(2000, 1, 1)
    edges = [cftime.DatetimeGregorian(2000, m, 1) for m in range(1, periods + 2)]

    # Bounds as [start, end) for each month
    bnds = np.array([[edges[i], edges[i + 1]] for i in range(periods)])
    mid = np.array([edges[i] + (edges[i + 1] - edges[i]) / 2 for i in range(periods)])

    # Sample data
    values = xr.DataArray(
        np.arange(periods, dtype=float), dims=("time",), coords={"time": mid}
    )

    # Build dataset with CF-style bounds
    ds = xr.Dataset(
        {"foo": values},
        coords={
            "time": ("time", mid, {"bounds": "time_bounds"}),
            "time_bounds": (("time", "bounds"), bnds),
            "bounds": ("bounds", [0, 1]),
        },
    )

    time_c = cfxr.bounds_to_vertices(ds.time_bounds, "bounds")
    time_b = cfxr.vertices_to_bounds(time_c, out_dims=("bounds", "time"))
    assert_array_equal(ds.time_bounds, time_b)
