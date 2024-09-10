import numpy as np
import pytest
import xarray as xr

import cf_xarray as cfxr

from ..geometry import decode_geometries, encode_geometries
from . import requires_shapely


@pytest.fixture
def polygon_geometry() -> xr.DataArray:
    from shapely.geometry import Polygon

    # empty/fill workaround to avoid numpy deprecation(warning) due to the array interface of shapely geometries.
    geoms = np.empty(2, dtype=object)
    geoms[:] = [
        Polygon(([50, 0], [40, 15], [30, 0])),
        Polygon(([70, 50], [60, 65], [50, 50])),
    ]
    return xr.DataArray(geoms, dims=("index",), name="geometry")


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
        node_count=xr.DataArray([4, 3, 3], dims=("index",)),
        part_node_count=xr.DataArray([2, 2, 3, 3], dims=("part",)),
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


@pytest.fixture
def geometry_line_without_multilines_ds():
    from shapely.geometry import LineString

    # empty/fill workaround to avoid numpy deprecation(warning) due to the array interface of shapely geometries.
    geoms = np.empty(2, dtype=object)
    geoms[:] = [
        LineString([[0, 0], [1, 0], [1, 1]]),
        LineString([[1.0, 1.0], [2.0, 2.0], [1.7, 9.5]]),
    ]

    ds = xr.Dataset()
    shp_da = xr.DataArray(geoms, dims=("index",), name="geometry")

    cf_ds = ds.assign(
        x=xr.DataArray([0, 1, 1, 1.0, 2.0, 1.7], dims=("node",), attrs={"axis": "X"}),
        y=xr.DataArray([0, 0, 1, 1.0, 2.0, 9.5], dims=("node",), attrs={"axis": "Y"}),
        node_count=xr.DataArray([3, 3], dims=("index",)),
        crd_x=xr.DataArray([0.0, 1.0], dims=("index",), attrs={"nodes": "x"}),
        crd_y=xr.DataArray([0.0, 1.0], dims=("index",), attrs={"nodes": "y"}),
        geometry_container=xr.DataArray(
            attrs={
                "geometry_type": "line",
                "node_count": "node_count",
                "node_coordinates": "x y",
                "coordinates": "crd_x crd_y",
            }
        ),
    )

    cf_ds = cf_ds.set_coords(["x", "y", "crd_x", "crd_y"])

    return cf_ds, shp_da


@pytest.fixture
def geometry_polygon_without_holes_ds(polygon_geometry):
    shp_da = polygon_geometry
    ds = xr.Dataset()

    cf_ds = ds.assign(
        x=xr.DataArray(
            [50, 40, 30, 50, 70, 60, 50, 70], dims=("node",), attrs={"axis": "X"}
        ),
        y=xr.DataArray(
            [0, 15, 0, 0, 50, 65, 50, 50], dims=("node",), attrs={"axis": "Y"}
        ),
        node_count=xr.DataArray([4, 4], dims=("index",)),
        crd_x=xr.DataArray([50, 70], dims=("index",), attrs={"nodes": "x"}),
        crd_y=xr.DataArray([0, 50], dims=("index",), attrs={"nodes": "y"}),
        geometry_container=xr.DataArray(
            attrs={
                "geometry_type": "polygon",
                "node_count": "node_count",
                "node_coordinates": "x y",
                "coordinates": "crd_x crd_y",
            }
        ),
    )

    cf_ds = cf_ds.set_coords(["x", "y", "crd_x", "crd_y"])

    return cf_ds, shp_da


@pytest.fixture
def geometry_polygon_without_multipolygons_ds():
    from shapely.geometry import Polygon

    # empty/fill workaround to avoid numpy deprecation(warning) due to the array interface of shapely geometries.
    geoms = np.empty(2, dtype=object)
    geoms[:] = [
        Polygon(([50, 0], [40, 15], [30, 0])),
        Polygon(
            ([70, 50], [60, 65], [50, 50]),
            [
                ([55, 55], [60, 60], [65, 55]),
            ],
        ),
    ]

    ds = xr.Dataset()
    shp_da = xr.DataArray(geoms, dims=("index",), name="geometry")

    cf_ds = ds.assign(
        x=xr.DataArray(
            [50, 40, 30, 50, 70, 60, 50, 70, 55, 60, 65, 55],
            dims=("node",),
            attrs={"axis": "X"},
        ),
        y=xr.DataArray(
            [0, 15, 0, 0, 50, 65, 50, 50, 55, 60, 55, 55],
            dims=("node",),
            attrs={"axis": "Y"},
        ),
        node_count=xr.DataArray([4, 8], dims=("index",)),
        part_node_count=xr.DataArray([4, 4, 4], dims=("part",)),
        interior_ring=xr.DataArray([0, 0, 1], dims=("part",)),
        crd_x=xr.DataArray([50, 70], dims=("index",), attrs={"nodes": "x"}),
        crd_y=xr.DataArray([0, 50], dims=("index",), attrs={"nodes": "y"}),
        geometry_container=xr.DataArray(
            attrs={
                "geometry_type": "polygon",
                "node_count": "node_count",
                "part_node_count": "part_node_count",
                "interior_ring": "interior_ring",
                "node_coordinates": "x y",
                "coordinates": "crd_x crd_y",
            }
        ),
    )

    cf_ds = cf_ds.set_coords(["x", "y", "crd_x", "crd_y"])

    return cf_ds, shp_da


@pytest.fixture
def geometry_polygon_ds():
    from shapely.geometry import MultiPolygon, Polygon

    # empty/fill workaround to avoid numpy deprecation(warning) due to the array interface of shapely geometries.
    geoms = np.empty(2, dtype=object)
    geoms[:] = [
        MultiPolygon(
            [
                (
                    ([20, 0], [10, 15], [0, 0]),
                    [
                        ([5, 5], [10, 10], [15, 5]),
                    ],
                ),
                (([20, 20], [10, 35], [0, 20]),),
            ]
        ),
        Polygon(([50, 0], [40, 15], [30, 0])),
    ]

    ds = xr.Dataset()
    shp_da = xr.DataArray(geoms, dims=("index",), name="geometry")

    cf_ds = ds.assign(
        x=xr.DataArray(
            [20, 10, 0, 20, 5, 10, 15, 5, 20, 10, 0, 20, 50, 40, 30, 50],
            dims=("node",),
            attrs={"axis": "X"},
        ),
        y=xr.DataArray(
            [0, 15, 0, 0, 5, 10, 5, 5, 20, 35, 20, 20, 0, 15, 0, 0],
            dims=("node",),
            attrs={"axis": "Y"},
        ),
        node_count=xr.DataArray([12, 4], dims=("index",)),
        part_node_count=xr.DataArray([4, 4, 4, 4], dims=("part",)),
        interior_ring=xr.DataArray([0, 1, 0, 0], dims=("part",)),
        crd_x=xr.DataArray([20, 50], dims=("index",), attrs={"nodes": "x"}),
        crd_y=xr.DataArray([0, 0], dims=("index",), attrs={"nodes": "y"}),
        geometry_container=xr.DataArray(
            attrs={
                "geometry_type": "polygon",
                "node_count": "node_count",
                "part_node_count": "part_node_count",
                "interior_ring": "interior_ring",
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
    expected = expected.copy(deep=True)

    # This isn't really a roundtrip test
    out = xr.merge([in_ds.drop_vars("geometry"), cfxr.shapely_to_cf(in_ds.geometry)])
    del expected.data.attrs["geometry"]
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
                in_ds.geometry.isel(index=slice(1, None)).data,
                grid_mapping="latitude_longitude",
            ),
        ]
    )
    np.testing.assert_array_equal(out.lon, expected.crd_x)
    assert "longitude" in out.cf
    assert "latitude" in out.cf

    out = cfxr.shapely_to_cf([Point(2, 3)])
    assert set(out.dims) == {"features", "node"}


@requires_shapely
def test_shapely_to_cf_for_lines_as_da(geometry_line_ds):
    expected, in_da = geometry_line_ds

    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected)

    in_da = in_da.assign_coords(index=["a", "b", "c"])
    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected.assign_coords(index=["a", "b", "c"]))


@requires_shapely
def test_shapely_to_cf_for_lines_as_sequence(geometry_line_ds):
    expected, in_da = geometry_line_ds
    actual = cfxr.shapely_to_cf(in_da.values)
    xr.testing.assert_identical(actual, expected)


@requires_shapely
def test_shapely_to_cf_for_lines_without_multilines(
    geometry_line_without_multilines_ds,
):
    expected, in_da = geometry_line_without_multilines_ds
    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected)


@requires_shapely
def test_shapely_to_cf_for_polygons_as_da(geometry_polygon_ds):
    expected, in_da = geometry_polygon_ds

    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected)

    in_da = in_da.assign_coords(index=["a", "b"])
    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected.assign_coords(index=["a", "b"]))


@requires_shapely
def test_shapely_to_cf_for_polygons_as_sequence(geometry_polygon_ds):
    expected, in_da = geometry_polygon_ds
    actual = cfxr.shapely_to_cf(in_da.values)
    xr.testing.assert_identical(actual, expected)


@requires_shapely
def test_shapely_to_cf_for_polygons_without_multipolygons(
    geometry_polygon_without_multipolygons_ds,
):
    expected, in_da = geometry_polygon_without_multipolygons_ds
    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected)


@requires_shapely
def test_shapely_to_cf_for_polygons_without_holes(
    geometry_polygon_without_holes_ds,
):
    expected, in_da = geometry_polygon_without_holes_ds
    actual = cfxr.shapely_to_cf(in_da)
    xr.testing.assert_identical(actual, expected)


@requires_shapely
def test_shapely_to_cf_errors():
    from shapely.geometry import Point, Polygon

    geoms = [
        Polygon([[1, 1], [1, 3], [3, 3], [1, 1]]),
        Polygon([[1, 1, 4], [1, 3, 4], [3, 3, 3], [1, 1, 4]]),
        Point(1, 2),
    ]
    with pytest.raises(ValueError, match="Geometry type combination"):
        cfxr.shapely_to_cf(geoms)

    encoded = cfxr.shapely_to_cf(
        [Point(4, 5)], grid_mapping="albers_conical_equal_area"
    )
    assert encoded["x"].attrs["standard_name"] == "projection_x_coordinate"
    assert encoded["y"].attrs["standard_name"] == "projection_y_coordinate"
    for name in ["x", "y", "crd_x", "crd_y"]:
        assert "grid_mapping" not in encoded[name].attrs


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
def test_cf_to_shapely_for_lines(geometry_line_ds):
    in_ds, expected = geometry_line_ds

    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(actual.drop_vars(["crd_x", "crd_y"]), expected)


@requires_shapely
def test_cf_to_shapely_for_lines_without_multilines(
    geometry_line_without_multilines_ds,
):
    in_ds, expected = geometry_line_without_multilines_ds
    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(actual.drop_vars(["crd_x", "crd_y"]), expected)

    in_ds = in_ds.assign_coords(index=["b", "c"])
    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(
        actual.drop_vars(["crd_x", "crd_y"]), expected.assign_coords(index=["b", "c"])
    )


@requires_shapely
def test_cf_to_shapely_for_polygons(geometry_polygon_ds):
    in_ds, expected = geometry_polygon_ds

    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(actual.drop_vars(["crd_x", "crd_y"]), expected)


@requires_shapely
def test_cf_to_shapely_for_polygons_without_multipolygons(
    geometry_polygon_without_multipolygons_ds,
):
    in_ds, expected = geometry_polygon_without_multipolygons_ds
    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(actual.drop_vars(["crd_x", "crd_y"]), expected)

    in_ds = in_ds.assign_coords(index=["b", "c"])
    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(
        actual.drop_vars(["crd_x", "crd_y"]), expected.assign_coords(index=["b", "c"])
    )


@requires_shapely
def test_cf_to_shapely_for_polygons_without_holes(
    geometry_polygon_without_holes_ds,
):
    in_ds, expected = geometry_polygon_without_holes_ds
    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(actual.drop_vars(["crd_x", "crd_y"]), expected)

    in_ds = in_ds.assign_coords(index=["b", "c"])
    actual = cfxr.cf_to_shapely(in_ds)
    assert actual.dims == ("index",)
    xr.testing.assert_identical(
        actual.drop_vars(["crd_x", "crd_y"]), expected.assign_coords(index=["b", "c"])
    )


@requires_shapely
def test_cf_to_shapely_errors(geometry_ds, geometry_line_ds, geometry_polygon_ds):
    in_ds, _ = geometry_ds
    in_ds.geometry_container.attrs["geometry_type"] = "punkt"
    with pytest.raises(ValueError, match="Valid CF geometry types are "):
        cfxr.cf_to_shapely(in_ds)

    in_ds, _ = geometry_line_ds
    del in_ds.geometry_container.attrs["node_count"]
    with pytest.raises(ValueError, match="'node_count' must be provided"):
        cfxr.cf_to_shapely(in_ds)

    in_ds, _ = geometry_polygon_ds
    del in_ds.geometry_container.attrs["node_count"]
    with pytest.raises(ValueError, match="'node_count' must be provided"):
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


@requires_shapely
def test_encode_decode(geometry_ds, polygon_geometry):
    geom_dim_ds = xr.Dataset()
    geom_dim_ds = geom_dim_ds.assign_coords(
        xr.Coordinates(
            coords={"geoms": xr.Variable("geoms", polygon_geometry.variable)},
            indexes={},
        )
    ).assign({"foo": ("geoms", [1, 2])})

    polyds = (
        polygon_geometry.rename("polygons").rename({"index": "index2"}).to_dataset()
    )
    multi_ds = xr.merge([polyds, geometry_ds[1]])
    for ds in (geometry_ds[1], polygon_geometry.to_dataset(), geom_dim_ds, multi_ds):
        roundtripped = decode_geometries(encode_geometries(ds))
        xr.testing.assert_identical(ds, roundtripped)
