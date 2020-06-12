import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_identical

import cf_xarray  # noqa

from . import raise_if_dask_computes

mpl.use("Agg")
ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4), lon=slice(50))
ds.air.attrs["cell_measures"] = "area: cell_area"
ds.coords["cell_area"] = (
    xr.DataArray(np.cos(ds.lat * np.pi / 180)) * xr.ones_like(ds.lon) * 105e3 * 110e3
)
datasets = [ds, ds.chunk({"lat": 5})]
dataarrays = [ds.air, ds.air.chunk({"lat": 5})]
objects = datasets + dataarrays


def test_describe():
    actual = ds.cf._describe()
    expected = (
        "Axes:\n\tX: ['lon']\n\tY: ['lat']\n\tZ: [None]\n\tT: ['time']\n"
        "\nCoordinates:\n\tlongitude: ['lon']\n\tlatitude: ['lat']"
        "\n\tvertical: [None]\n\ttime: ['time']\n"
        "\nCell Measures:\n\tarea: unsupported\n\tvolume: unsupported\n"
    )
    assert actual == expected


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


@pytest.mark.parametrize("obj", dataarrays)
def test_weighted(obj):
    with raise_if_dask_computes(max_computes=2):
        # weights are checked for nans
        expected = obj.weighted(obj["cell_area"]).sum("lat")
        actual = obj.cf.weighted("area").sum("Y")
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
    "key, expected_key",
    (
        ("X", "lon"),
        ("Y", "lat"),
        ("T", "time"),
        ("longitude", "lon"),
        ("latitude", "lat"),
        ("time", "time"),
        pytest.param(
            "area",
            "cell_area",
            marks=pytest.mark.xfail(reason="measures not implemented for dataset"),
        ),
    ),
)
def test_getitem(obj, key, expected_key):
    actual = obj.cf[key]
    if isinstance(obj, xr.Dataset):
        expected_key = [expected_key]
    expected = obj[expected_key]
    assert_identical(actual, expected)


@pytest.mark.parametrize("obj", objects)
def test_getitem_errors(obj,):
    with pytest.raises(KeyError):
        obj.cf["XX"]
    obj.lon.attrs = {}
    with pytest.raises(KeyError):
        obj.cf["X"]


def test_getitem_uses_coordinates():
    # POP-like dataset
    ds = xr.Dataset()
    ds.coords["TLONG"] = (
        ("nlat", "nlon"),
        np.ones((20, 30)),
        {"axis": "X", "units": "degrees_east"},
    )
    ds.coords["TLAT"] = (
        ("nlat", "nlon"),
        2 * np.ones((20, 30)),
        {"axis": "Y", "units": "degrees_north"},
    )
    ds.coords["ULONG"] = (
        ("nlat", "nlon"),
        0.5 * np.ones((20, 30)),
        {"axis": "X", "units": "degrees_east"},
    )
    ds.coords["ULAT"] = (
        ("nlat", "nlon"),
        2.5 * np.ones((20, 30)),
        {"axis": "Y", "units": "degrees_north"},
    )
    ds["UVEL"] = (
        ("nlat", "nlon"),
        np.ones((20, 30)) * 15,
        {"coordinates": "ULONG ULAT"},
    )
    ds["TEMP"] = (
        ("nlat", "nlon"),
        np.ones((20, 30)) * 15,
        {"coordinates": "TLONG TLAT"},
    )

    assert_identical(
        ds.cf["X"], ds.reset_coords()[["ULONG", "TLONG"]].set_coords(["ULONG", "TLONG"])
    )
    assert_identical(
        ds.cf["Y"], ds.reset_coords()[["ULAT", "TLAT"]].set_coords(["ULAT", "TLAT"])
    )
    assert_identical(ds.UVEL.cf["X"], ds["ULONG"].reset_coords(drop=True))
    assert_identical(ds.TEMP.cf["X"], ds["TLONG"].reset_coords(drop=True))
