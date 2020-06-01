import pytest
import xarray as xr
from xarray.testing import assert_identical

import cf_xarray  # noqa

from . import raise_if_dask_computes

ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4))
objects = [
    pytest.param(ds, marks=pytest.mark.xfail),
    ds.air,
    pytest.param(ds.chunk({"lat": 5}), marks=pytest.mark.xfail),
    ds.air.chunk({"lat": 5}),
]


@pytest.mark.parametrize("obj", objects)
def test_wrapped_classes(obj):
    with raise_if_dask_computes():
        expected = obj.resample(time="M").mean("lat")
        actual = obj.cf.resample(T="M").mean("Y")
    assert_identical(expected, actual)

    # groupby
    # rolling
    # coarsen
    # weighted


@pytest.mark.parametrize("obj", objects)
def test_other_methods(obj):
    with raise_if_dask_computes():
        expected = obj.isel(time=slice(2))
        actual = obj.cf.isel(T=slice(2))
    assert_identical(expected, actual)

    with raise_if_dask_computes():
        expected = obj.sum("time")
        actual = obj.cf.sum("T")
    assert_identical(expected, actual)


@pytest.mark.parametrize("obj", objects)
def test_plot(obj):
    obj.isel(time=1).cf.plot(x="X", y="Y")
    obj.isel(time=1).cf.plot.contourf(x="X", y="Y")

    obj.cf.plot(x="X", y="Y", col="T")
    obj.cf.plot.contourf(x="X", y="Y", col="T")

    obj.isel(lat=[0, 1], lon=1).cf.plot.line(x="T", hue="Y")
