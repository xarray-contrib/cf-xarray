import numpy as np
import pytest
import xarray as xr


@pytest.fixture(scope="session")
def geometry_ds():
    pytest.importorskip("shapely")

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
            "data": xr.DataArray(
                range(len(geoms)),
                dims=("index",),
                attrs={
                    "coordinates": "crd_x crd_y",
                },
            ),
            "time": xr.DataArray([0, 0, 0, 1], dims=("index",)),
        }
    )
    shp_ds = ds.assign(geometry=xr.DataArray(geoms, dims=("index",)))
    # Here, since it should not be present in shp_ds
    ds.data.attrs["geometry"] = "geometry_container"

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
