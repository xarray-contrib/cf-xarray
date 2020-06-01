import matplotlib as mpl
import matplotlib.pyplot as plt
import pytest
import xarray as xr
from xarray.testing import assert_identical

import cf_xarray  # noqa

from . import raise_if_dask_computes

mpl.use("Agg")
ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4), lon=slice(50))
datasets = [ds, ds.chunk({"lat": 5})]
dataarrays = [ds.air, ds.air.chunk({"lat": 5})]
objects = datasets + dataarrays


@pytest.mark.parametrize("obj", objects)
@pytest.mark.parametrize(
    "attr, xrkwargs, cfkwargs",
    (
        ("resample", {"time": "M"}, {"T": "M"}),
        ("rolling", {"lat": 5}, {"Y": 5}),
        ("groupby", {"group": "time"}, {"group": "T"}),
        pytest.param(
            "coarsen",
            {"lon": 2, "lat": 5},
            {"X": 2, "Y": 5},
            marks=pytest.mark.skip(
                reason="xarray GH4120. any test after this will fail since attrs are lost"
            ),
        ),
        # order of above tests is important: See xarray GH4120
        # groupby("time.day")?
        # groupby_bins
        # weighted
    ),
)
def test_wrapped_classes(obj, attr, xrkwargs, cfkwargs):

    if attr in ("rolling", "coarsen"):
        # TODO: xarray bug, rolling and coarsen don't accept ellipsis
        args = ()
    else:
        args = (...,)

    with raise_if_dask_computes():
        expected = getattr(obj, attr)(**xrkwargs).mean(*args)
        actual = getattr(obj.cf, attr)(**cfkwargs).mean(*args)
    assert_identical(expected, actual)

    if attr in ("groupby", "groupby_bins"):
        # TODO: this should work for resample too?
        with raise_if_dask_computes():
            expected = getattr(obj, attr)(**xrkwargs).mean("lat")
            actual = getattr(obj.cf, attr)(**cfkwargs).mean("Y")
        assert_identical(expected, actual)


@pytest.mark.parametrize("obj", objects)
def test_kwargs_methods(obj):
    with raise_if_dask_computes():
        expected = obj.isel(time=slice(2))
        actual = obj.cf.isel(T=slice(2))
    assert_identical(expected, actual)


@pytest.mark.parametrize("obj", objects)
def test_args_methods(obj):
    with raise_if_dask_computes():
        expected = obj.sum("time")
        actual = obj.cf.sum("T")
    assert_identical(expected, actual)


@pytest.mark.parametrize("obj", dataarrays)
def test_dataarray_plot(obj):

    rv = obj.isel(time=1).cf.plot(x="X", y="Y")
    assert isinstance(rv, mpl.collections.QuadMesh)
    plt.close()

    rv = obj.isel(time=1).cf.plot.contourf(x="X", y="Y")
    assert isinstance(rv, mpl.contour.QuadContourSet)
    plt.close()

    rv = obj.cf.plot(x="X", y="Y", col="T")
    assert isinstance(rv, xr.plot.FacetGrid)
    plt.close()

    rv = obj.cf.plot.contourf(x="X", y="Y", col="T")
    assert isinstance(rv, xr.plot.FacetGrid)
    plt.close()

    rv = obj.isel(lat=[0, 1], lon=1).cf.plot.line(x="T", hue="Y")
    assert all([isinstance(line, mpl.lines.Line2D) for line in rv])
    plt.close()


@pytest.mark.parametrize("obj", datasets)
def test_dataset_plot(obj):
    pass


@pytest.mark.parametrize("obj", objects)
@pytest.mark.parametrize(
    "key, expected_key", (("X", "lon"), ("Y", "lat"), ("T", "time"))
)
def test_getitem(obj, key, expected_key):
    actual = obj.cf[key]
    expected = obj[expected_key]
    assert_identical(actual, expected)
