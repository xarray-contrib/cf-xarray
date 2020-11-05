import numpy as np
from numpy.testing import assert_array_equal

import cf_xarray as cf  # noqa

from .datasets import airds, mollwds


def test_bounds_to_vertices():
    # 1D case
    ds = airds.cf.add_bounds(["lon", "lat"])
    lat_c = cf.bounds_to_vertices(ds.lat_bounds, bounds_dim="bounds")
    assert np.all(ds.lat.values + 1.25 == lat_c.values[:-1])

    # 2D case, CF- order
    lat_c = cf.bounds_to_vertices(mollwds.lat_bounds, bounds_dim="bounds")
    assert mollwds.lat_vertices.equals(lat_c)

    # Transposing the array changes the bounds direction
    ds = mollwds.transpose("bounds", "y", "x", "y_vertices", "x_vertices")
    lon_c = cf.bounds_to_vertices(ds.lon_bounds, bounds_dim="bounds", order="clockwise")
    lon_c2 = cf.bounds_to_vertices(ds.lon_bounds, bounds_dim="bounds", order=None)
    assert ds.lon_vertices.equals(lon_c)
    assert ds.lon_vertices.equals(lon_c2)


def test_vertices_to_bounds():
    # 1D case
    ds = airds.cf.add_bounds(["lon", "lat"])
    lat_c = cf.bounds_to_vertices(ds.lat_bounds, bounds_dim="bounds")
    lat_b = cf.vertices_to_bounds(lat_c, out_dims=("bounds", "lat"))
    assert_array_equal(ds.lat_bounds, lat_b)

    # 2D case
    lon_b = cf.vertices_to_bounds(mollwds.lon_vertices, out_dims=("bounds", "x", "y"))
    assert (mollwds.lon_bounds == lon_b).all()
