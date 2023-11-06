import numpy as np
import pytest
import xarray as xr

import cf_xarray as cfxr

from . import requires_shapely


@pytest.fixture
def geometry_ds():
    from shapely.geometry import MultiPoint, Point

    # empty/fill workaround to avoid numpy deprecation(warning) due to the array interface of shapely geometries.
    geoms = np.empty(4, dtype=object)
    geoms[:] = [
        MultiPoint([(1.0, 2.0), (2.0, 3.0)]),
        Point(3.0, 4.0),
        Point(4.0, 5.0),
        Point(3.0, 4.0),
    ]

    ds = xr.Dataset(
        {
            "data": xr.DataArray(range(len(geoms)), dims=("index",)),
            "time": xr.DataArray([0, 0, 0, 1], dims=("index",)),
        }
    )
    shp_ds = ds.assign(geometry=xr.DataArray(geoms, dims=("index",)))

    cf_ds = ds.assign(
        x=xr.DataArray([1.0, 2.0, 3.0, 4.0, 3.0], dims=("node",), attrs={"axis": "X"}),
        y=xr.DataArray([2.0, 3.0, 4.0, 5.0, 4.0], dims=("node",), attrs={"axis": "Y"}),
        node_count=xr.DataArray([2, 1, 1, 1], dims=("index",)),
        crd_x=xr.DataArray([1.0, 3.0, 4.0, 3.0], dims=("index",), attrs={"nodes": "x"}),
        crd_y=xr.DataArray([2.0, 4.0, 5.0, 4.0], dims=("index",), attrs={"nodes": "y"}),
        geometry_container=xr.DataArray(
            attrs={
                "geometry_type": "point",
                "node_count": "node_count",
                "node_coordinates": "x y",
                "coordinates": "crd_x crd_y",
            }
        ),
    )

    cf_ds = cf_ds.set_coords(["x", "y", "crd_x", "crd_y"])

    return cf_ds, shp_ds


@pytest.fixture
def geometry_line_ds():
    from shapely.geometry import LineString, MultiLineString

    # empty/fill workaround to avoid numpy deprecation(warning) due to the array interface of shapely geometries.
    geoms = np.empty(3, dtype=object)
    geoms[:] = [
        MultiLineString([[[0, 0], [1, 2]], [[4, 4], [5, 6]]]),
        LineString([[0, 0], [1, 0], [1, 1]]),
        LineString([[1.0, 1.0], [2.0, 2.0], [1.7, 9.5]]),
    ]

    ds = xr.Dataset()
    shp_da = xr.DataArray(geoms, dims=("index",), name="geometry")

    cf_ds = ds.assign(
        x=xr.DataArray(
            [0, 1, 4, 5, 0, 1, 1, 1.0, 2.0, 1.7], dims=("node",), attrs={"axis": "X"}
        ),
        y=xr.DataArray(
            [0, 2, 4, 6, 0, 0, 1, 1.0, 2.0, 9.5], dims=("node",), attrs={"axis": "Y"}
        ),
        part_node_count=xr.DataArray([4, 3, 3], dims=("index",)),
        node_count=xr.DataArray([2, 2, 3, 3], dims=("segment",)),
        crd_x=xr.DataArray([0.0, 0.0, 1.0], dims=("index",), attrs={"nodes": "x"}),
        crd_y=xr.DataArray([0.0, 0.0, 1.0], dims=("index",), attrs={"nodes": "y"}),
        geometry_container=xr.DataArray(
            attrs={
                "geometry_type": "line",
                "node_count": "node_count",
                "part_node_count": "part_node_count",
                "node_coordinates": "x y",
                "coordinates": "crd_x crd_y",
            }
        ),
    )

    cf_ds = cf_ds.set_coords(["x", "y", "crd_x", "crd_y"])

    return cf_ds, shp_da


@requires_shapely
def test_shapely_to_cf(geometry_ds):
    from shapely.geometry import Point

    expected, in_ds = geometry_ds

    out = xr.merge([in_ds.drop_vars("geometry"), cfxr.shapely_to_cf(in_ds.geometry)])
    xr.testing.assert_identical(out, expected)

    out = xr.merge(
        [
            in_ds.drop_vars("geometry").isel(index=slice(1, None)),
            cfxr.shapely_to_cf(in_ds.geometry.isel(index=slice(1, None))),
        ]
    )
    expected = expected.isel(index=slice(1, None), node=slice(2, None)).drop_vars(
        "node_count"
    )
    del expected.geometry_container.attrs["node_count"]
    xr.testing.assert_identical(out, expected)

    out = xr.merge(
        [
            in_ds.drop_vars("geometry").isel(index=slice(1, None)),
            cfxr.shapely_to_cf(
                in_ds.geometry.isel(index=slice(1, None)),
                grid_mapping="longitude_latitude",
            ),
        ]
    )
    np.testing.assert_array_equal(out.lon, expected.crd_x)
    assert "longitude" in out.cf
    assert "latitude" in out.cf

    out = cfxr.shapely_to_cf([Point(2, 3)])
    assert set(out.dims) == {"features", "node"}


@requires_shapely
def test_shapely_to_cf_errors():
    from shapely.geometry import Polygon, Point

    geoms = [Polygon([[1, 1], [1, 3], [3, 3], [1, 1]]), Polygon([[1, 1, 4], [1, 3, 4], [3, 3, 3], [1, 1, 4]])]
    with pytest.raises(NotImplementedError, match="Polygon geometry conversion is not implemented"):
        cfxr.shapely_to_cf(geoms)

    geoms.append(Point(1, 2))
    with pytest.raises(ValueError, match="Mixed geometry types are not supported"):
        cfxr.shapely_to_cf(geoms)

    with pytest.raises(
        NotImplementedError, match="Only grid mapping longitude_latitude"
    ):
        cfxr.shapely_to_cf([Point(4, 5)], grid_mapping="albers_conical_equal_area")


@requires_shapely
def test_cf_to_shapely(geometry_ds):
    in_ds, exp = geometry_ds

    xr.testing.assert_identical(
        cfxr.cf_to_shapely(in_ds).drop_vars(["crd_x", "crd_y"]), exp.geometry
    )

    in_ds = in_ds.isel(index=slice(1, None), node=slice(2, None)).drop_vars(
        "node_count"
    )
    del in_ds.geometry_container.attrs["node_count"]
    out = cfxr.cf_to_shapely(in_ds)
    assert out.dims == ("index",)


@requires_shapely
def test_cf_to_shapely_for_line(geometry_line_ds):
    in_ds, expected = geometry_line_ds

    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)

    xr.testing.assert_identical(actual.drop_vars(["crd_x", "crd_y"]), expected)


@requires_shapely
def test_shapely_to_cf_for_line(geometry_line_ds):
    expected, in_da = geometry_line_ds

    actual = cfxr.shapely_to_cf(in_da)

    xr.testing.assert_identical(actual, expected)


@requires_shapely
def test_cf_to_shapely_errors(geometry_ds):
    in_ds, _ = geometry_ds
    in_ds.geometry_container.attrs["geometry_type"] = "polygon"
    with pytest.raises(NotImplementedError, match="Polygon geometry"):
        cfxr.cf_to_shapely(in_ds)

    in_ds.geometry_container.attrs["geometry_type"] = "punkt"
    with pytest.raises(ValueError, match="Valid CF geometry types are "):
        cfxr.cf_to_shapely(in_ds)


@requires_shapely
def test_reshape_unique_geometries(geometry_ds):
    _, in_ds = geometry_ds

    out = cfxr.geometry.reshape_unique_geometries(in_ds)
    assert out.geometry.dims == ("features",)
    assert out.data.dims == ("features", "index")
    np.testing.assert_array_equal(
        out.geometry, in_ds.geometry.values[np.array([1, 2, 0])]
    )

    in_ds["index"] = in_ds.time
    in_ds = in_ds.drop_vars("time").rename(index="time")

    out = cfxr.geometry.reshape_unique_geometries(in_ds)
    assert out.geometry.dims == ("features",)
    assert out.data.dims == ("features", "time")
    np.testing.assert_array_equal(out.time, [0, 1])

    geoms = in_ds.geometry.expand_dims(n=[1, 2])
    in_ds = in_ds.assign(geometry=geoms)
    with pytest.raises(ValueError, match="The geometry variable must be 1D"):
        cfxr.geometry.reshape_unique_geometries(in_ds)
