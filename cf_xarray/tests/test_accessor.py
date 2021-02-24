import itertools
from textwrap import dedent

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from xarray import Dataset
from xarray.testing import assert_allclose, assert_identical

import cf_xarray  # noqa

from ..datasets import (
    airds,
    anc,
    ds_no_attrs,
    forecast,
    mollwds,
    multiple,
    popds,
    romsds,
)
from . import raise_if_dask_computes

mpl.use("Agg")

ds = airds
datasets = [airds, airds.chunk({"lat": 5})]
dataarrays = [airds.air, airds.air.chunk({"lat": 5})]
objects = datasets + dataarrays


def assert_dicts_identical(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    for k in dict1:
        assert_identical(dict1[k], dict2[k])


def test_repr():
    # Dataset.
    # Stars: axes, coords, and std names
    actual = airds.cf.__repr__()
    expected = """\
    Coordinates:
    - CF Axes: * X: ['lon']
               * Y: ['lat']
               * T: ['time']
                 Z: n/a

    - CF Coordinates: * longitude: ['lon']
                      * latitude: ['lat']
                      * time: ['time']
                        vertical: n/a

    - Cell Measures:   area: ['cell_area']
                       volume: n/a

    - Standard Names: * latitude: ['lat']
                      * longitude: ['lon']
                      * time: ['time']

    Data Variables:
    - Cell Measures:   area, volume: n/a

    - Standard Names:   air_temperature: ['air']
    """
    assert actual == dedent(expected)

    # DataArray (Coordinates section same as Dataset)
    assert airds.cf.__repr__().startswith(airds["air"].cf.__repr__())
    actual = airds["air"].cf.__repr__()
    expected = """\
    Coordinates:
    - CF Axes: * X: ['lon']
               * Y: ['lat']
               * T: ['time']
                 Z: n/a

    - CF Coordinates: * longitude: ['lon']
                      * latitude: ['lat']
                      * time: ['time']
                        vertical: n/a

    - Cell Measures:   area: ['cell_area']
                       volume: n/a

    - Standard Names: * latitude: ['lat']
                      * longitude: ['lon']
                      * time: ['time']
    """
    assert actual == dedent(expected)

    # Empty Standard Names
    actual = popds.cf.__repr__()
    expected = """\
    Coordinates:
    - CF Axes: * X: ['nlon']
               * Y: ['nlat']
                 Z, T: n/a

    - CF Coordinates:   longitude: ['TLONG', 'ULONG']
                        latitude: ['TLAT', 'ULAT']
                        vertical, time: n/a

    - Cell Measures:   area, volume: n/a

    - Standard Names:   n/a

    Data Variables:
    - Cell Measures:   area, volume: n/a

    - Standard Names:   sea_water_potential_temperature: ['TEMP']
                        sea_water_x_velocity: ['UVEL']
    """
    assert actual == dedent(expected)


def test_axes():
    expected = dict(T=["time"], X=["lon"], Y=["lat"])
    actual = airds.cf.axes
    assert actual == expected

    expected = dict(X=["nlon"], Y=["nlat"])
    actual = popds.cf.axes
    assert actual == expected


def test_coordinates():
    expected = dict(latitude=["lat"], longitude=["lon"], time=["time"])
    actual = airds.cf.coordinates
    assert actual == expected

    expected = dict(latitude=["TLAT", "ULAT"], longitude=["TLONG", "ULONG"])
    actual = popds.cf.coordinates
    assert actual == expected


def test_cell_measures():
    ds = airds.copy(deep=True)
    ds["foo"] = xr.DataArray(ds["cell_area"], attrs=dict(standard_name="foo_std_name"))
    ds["air"].attrs["cell_measures"] += " foo_measure: foo"
    assert ("foo_std_name" in ds.cf["air_temperature"].cf) and ("foo_measure" in ds.cf)

    ds["air"].attrs["cell_measures"] += " volume: foo"
    ds["foo"].attrs["cell_measures"] = ds["air"].attrs["cell_measures"]
    expected = dict(area=["cell_area"], foo_measure=["foo"], volume=["foo"])
    actual_air = ds["air"].cf.cell_measures
    actual_foo = ds.cf["foo_measure"].cf.cell_measures
    assert actual_air == actual_foo == expected

    actual = ds.cf.cell_measures
    assert actual == expected

    # Additional cell measure in repr
    actual = ds.cf.__repr__()
    expected = """\
    Data Variables:
    - Cell Measures:   foo_measure: ['foo']
                       volume: ['foo']
                       area: n/a

    - Standard Names:   air_temperature: ['air']
                        foo_std_name: ['foo']
    """
    assert actual.endswith(dedent(expected))


def test_standard_names():
    expected = dict(
        air_temperature=["air"], latitude=["lat"], longitude=["lon"], time=["time"]
    )
    actual = airds.cf.standard_names
    assert actual == expected

    dsnew = xr.Dataset()
    dsnew["a"] = ("a", np.arange(10), {"standard_name": "a"})
    dsnew["b"] = ("a", np.arange(10), {"standard_name": "a"})
    assert dsnew.cf.standard_names == dict(a=["a", "b"])


def test_getitem_standard_name():
    actual = airds.cf["air_temperature"]
    expected = airds["air"]
    assert_identical(actual, expected)

    actual = airds.lat.cf["latitude"]
    expected = airds["lat"]
    assert_identical(actual, expected)

    ds = airds.copy(deep=True)
    ds["air2"] = ds.air
    with pytest.raises(KeyError):
        ds.cf["air_temperature"]
    actual = ds.cf[["air_temperature"]]
    expected = ds[["air", "air2"]]
    assert_identical(actual, expected)

    with pytest.raises(KeyError):
        ds.air.cf["air_temperature"]


def test_getitem_ancillary_variables():
    expected = anc.set_coords(["q_error_limit", "q_detection_limit"])["q"]
    assert_identical(anc.cf["q"], expected)
    assert_identical(anc.cf["specific_humidity"], expected)

    with pytest.warns(UserWarning):
        anc[["q"]].cf["q"]

    for k in ["ULONG", "ULAT"]:
        assert k not in popds.cf["TEMP"].coords

    for k in ["TLONG", "TLAT"]:
        assert k not in popds.cf["UVEL"].coords


def test_rename_like():
    original = popds.copy(deep=True)

    # it'll match for axis: X (lon, nlon) and coordinate="longitude" (lon, TLONG)
    # so delete the axis attributes
    newair = airds.copy(deep=True)
    del newair.lon.attrs["axis"]
    del newair.lat.attrs["axis"]

    renamed = popds.cf["TEMP"].cf.rename_like(newair)
    for k in ["TLONG", "TLAT"]:
        assert k not in renamed.coords
        assert k in original.coords
        assert original.TEMP.attrs["coordinates"] == "TLONG TLAT"

    assert "lon" in renamed.coords
    assert "lat" in renamed.coords
    assert renamed.attrs["coordinates"] == "lon lat"

    # standard name matching
    newroms = romsds.expand_dims(latitude=[1], longitude=[1]).cf.guess_coord_axis()
    renamed = popds.cf["UVEL"].cf.rename_like(newroms)
    assert renamed.attrs["coordinates"] == "longitude latitude"
    assert "longitude" in renamed.coords
    assert "latitude" in renamed.coords
    assert "ULON" not in renamed.coords
    assert "ULAT" not in renamed.coords

    # should change "temp" to "TEMP"
    renamed = romsds.cf.rename_like(popds)
    assert "temp" not in renamed
    assert "TEMP" in renamed


@pytest.mark.parametrize("obj", objects)
@pytest.mark.parametrize(
    "attr, xrkwargs, cfkwargs",
    (
        ("resample", {"time": "M"}, {"T": "M"}),
        ("rolling", {"lat": 5}, {"Y": 5}),
        ("groupby", {"group": "time"}, {"group": "T"}),
        ("groupby", {"group": "time.month"}, {"group": "T.month"}),
        ("groupby_bins", {"group": "lat", "bins": 5}, {"group": "latitude", "bins": 5}),
        pytest.param(
            "coarsen",
            {"lon": 2, "lat": 5},
            {"X": 2, "Y": 5},
            marks=pytest.mark.skip(
                reason="xarray GH4120. any test after this will fail since attrs are lost"
            ),
        ),
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
def test_groupby_reduce_multiple_dims(obj):
    actual = obj.cf.groupby("time.month").mean(["lat", "X"])
    expected = obj.groupby("time.month").mean(["lat", "lon"])
    assert_identical(actual, expected)


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

    with raise_if_dask_computes():
        expected = obj.isel({"time": slice(2)})
        actual = obj.cf.isel({"T": slice(2)})
    assert_identical(expected, actual)


def test_pos_args_methods():
    expected = airds.transpose("lon", "time", "lat")
    actual = airds.cf.transpose("longitude", "T", "latitude")
    assert_identical(actual, expected)

    actual = airds.cf.transpose("longitude", ...)
    assert_identical(actual, expected)

    expected = multiple.transpose("y2", "y1", "x1", "x2")
    actual = multiple.cf.transpose("Y", "X")
    assert_identical(actual, expected)


def test_preserve_unused_keys():

    ds = airds.copy(deep=True)
    ds.time.attrs.clear()
    actual = ds.cf.sel(X=260, Y=40, time=airds.time[:2], method="nearest")
    expected = ds.sel(lon=260, lat=40, time=airds.time[:2], method="nearest")
    assert_identical(actual, expected)


def test_kwargs_expand_key_to_multiple_keys():

    actual = multiple.cf.isel(X=5, Y=3)
    expected = multiple.isel(x1=5, y1=3, x2=5, y2=3)
    assert_identical(actual, expected)

    actual = multiple.cf.mean("X")
    expected = multiple.mean(["x1", "x2"])
    assert_identical(actual, expected)

    # Commenting out lines that use Coarsen
    # actual = multiple.cf.coarsen(X=10, Y=5)
    # expected = multiple.coarsen(x1=10, y1=5, x2=10, y2=5)
    # assert_identical(actual.mean(), expected.mean())


@pytest.mark.parametrize(
    "obj, expected",
    [
        (
            ds,
            {"latitude", "longitude", "time", "X", "Y", "T", "air_temperature", "area"},
        ),
        (ds.air, {"latitude", "longitude", "time", "X", "Y", "T", "area"}),
        (ds_no_attrs.air, set()),
    ],
)
def test_keys(obj, expected):
    actual = obj.cf.keys()
    assert actual == expected


@pytest.mark.parametrize("obj", objects)
def test_args_methods(obj):
    with raise_if_dask_computes():
        expected = obj.sum("time")
        actual = obj.cf.sum("T")
    assert_identical(expected, actual)


def test_dataarray_getitem():

    air = airds.air
    air.name = None

    assert_identical(air.cf["longitude"], air["lon"])
    with pytest.raises(KeyError):
        air.cf[["longitude"]]
    with pytest.raises(KeyError):
        air.cf[["longitude", "latitude"]]

    air["cell_area"].attrs["standard_name"] = "area_grid_cell"
    assert_identical(air.cf["area_grid_cell"], air.cell_area.reset_coords(drop=True))


def test_dataarray_plot():

    obj = airds.air

    rv = obj.isel(time=1).transpose("lon", "lat").cf.plot()
    assert isinstance(rv, mpl.collections.QuadMesh)
    assert all(v > 180 for v in rv.axes.get_xlim())
    assert all(v < 200 for v in rv.axes.get_ylim())
    plt.close()

    rv = obj.isel(time=1).transpose("lon", "lat").cf.plot.contourf()
    assert isinstance(rv, mpl.contour.QuadContourSet)
    assert all(v > 180 for v in rv.axes.get_xlim())
    assert all(v < 200 for v in rv.axes.get_ylim())
    plt.close()

    rv = obj.cf.plot(x="X", y="Y", col="T")
    assert isinstance(rv, xr.plot.FacetGrid)
    plt.close()

    rv = obj.cf.plot.contourf(x="X", y="Y", col="T")
    assert isinstance(rv, xr.plot.FacetGrid)
    plt.close()

    rv = obj.isel(lat=[0, 1], lon=1).cf.plot.line(x="T", hue="Y")
    assert all(isinstance(line, mpl.lines.Line2D) for line in rv)
    plt.close()

    # set y automatically
    rv = obj.isel(time=0, lon=1).cf.plot.line()
    np.testing.assert_equal(rv[0].get_ydata(), obj.lat.data)
    plt.close()

    # don't set y automatically
    rv = obj.isel(time=0, lon=1).cf.plot.line(x="lat")
    np.testing.assert_equal(rv[0].get_xdata(), obj.lat.data)
    plt.close()

    rv = obj.isel(time=0, lon=1).cf.plot(x="lat")
    np.testing.assert_equal(rv[0].get_xdata(), obj.lat.data)
    plt.close()

    # various line plots and automatic guessing
    rv = obj.cf.isel(T=1, Y=[0, 1, 2]).cf.plot.line()
    np.testing.assert_equal(rv[0].get_xdata(), obj.lon.data)
    plt.close()

    # rv = obj.cf.isel(T=1, Y=[0, 1, 2]).cf.plot(hue="Y")
    # np.testing.assert_equal(rv[0].get_xdata(), obj.lon.data)
    # plt.close()

    rv = obj.cf.isel(T=1, Y=[0, 1, 2]).cf.plot.line()
    np.testing.assert_equal(rv[0].get_xdata(), obj.lon.data)
    plt.close()

    obj = obj.copy(deep=True)
    obj.time.attrs.clear()
    rv = obj.cf.plot(x="X", y="Y", col="time")
    assert isinstance(rv, xr.plot.FacetGrid)
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
    assert key in obj.cf

    actual = obj.cf[key]
    expected = obj[expected_key]
    assert_identical(actual, expected)


@pytest.mark.parametrize("obj", objects)
def test_getitem_errors(obj):
    with pytest.raises(KeyError):
        obj.cf["XX"]
    obj2 = obj.copy(deep=True)
    obj2.lon.attrs = {}
    with pytest.raises(KeyError):
        obj2.cf["X"]


def test_getitem_regression():
    ds = xr.Dataset()
    ds.coords["area"] = xr.DataArray(np.ones(10), attrs={"standard_name": "cell_area"})
    assert_identical(ds.cf["cell_area"], ds["area"].reset_coords(drop=True))


def test_getitem_uses_coordinates():
    # POP-like dataset
    ds = popds
    assert_identical(
        ds.cf[["longitude"]],
        ds.reset_coords()[["ULONG", "TLONG"]].set_coords(["ULONG", "TLONG"]),
    )
    assert_identical(
        ds.cf[["latitude"]],
        ds.reset_coords()[["ULAT", "TLAT"]].set_coords(["ULAT", "TLAT"]),
    )
    assert_identical(ds.UVEL.cf["longitude"], ds["ULONG"].reset_coords(drop=True))
    assert_identical(ds.TEMP.cf["latitude"], ds["TLAT"].reset_coords(drop=True))


def test_getitem_uses_dimension_names_when_coordinates_attr():
    # POP-like dataset
    ds = popds
    assert_identical(ds.cf["X"], ds["nlon"])
    assert_identical(ds.cf["Y"], ds["nlat"])
    assert_identical(ds.UVEL.cf["X"], ds["nlon"])
    assert_identical(ds.TEMP.cf["Y"], ds["nlat"])


def test_plot_xincrease_yincrease():
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4), lon=slice(50))
    ds.lon.attrs["positive"] = "down"
    ds.lat.attrs["positive"] = "down"

    _, ax = plt.subplots(1, 1)
    ds.air.isel(time=1).cf.plot(ax=ax, x="X", y="Y")

    for lim in [ax.get_xlim(), ax.get_ylim()]:
        assert lim[0] > lim[1]


@pytest.mark.parametrize("dims", ["lat", "time", ["lat", "lon"]])
@pytest.mark.parametrize("obj", [airds])
def test_add_bounds(obj, dims):
    expected = {}
    expected["lat"] = xr.concat(
        [
            obj.lat.copy(data=np.arange(76.25, 16.0, -2.5)),
            obj.lat.copy(data=np.arange(73.75, 13.6, -2.5)),
        ],
        dim="bounds",
    )
    expected["lon"] = xr.concat(
        [
            obj.lon.copy(data=np.arange(198.75, 325 - 1.25, 2.5)),
            obj.lon.copy(data=np.arange(201.25, 325 + 1.25, 2.5)),
        ],
        dim="bounds",
    )
    t0 = pd.Timestamp("2013-01-01")
    t1 = pd.Timestamp("2013-01-01 18:00")
    dt = "6h"
    dtb2 = pd.Timedelta("3h")
    expected["time"] = xr.concat(
        [
            obj.time.copy(data=pd.date_range(start=t0 - dtb2, end=t1 - dtb2, freq=dt)),
            obj.time.copy(data=pd.date_range(start=t0 + dtb2, end=t1 + dtb2, freq=dt)),
        ],
        dim="bounds",
    )
    expected["lat"].attrs.clear()
    expected["lon"].attrs.clear()
    expected["time"].attrs.clear()

    added = obj.cf.add_bounds(dims)
    if isinstance(dims, str):
        dims = (dims,)

    for dim in dims:
        name = f"{dim}_bounds"
        assert name in added.coords
        assert added[dim].attrs["bounds"] == name
        assert_allclose(added[name].reset_coords(drop=True), expected[dim])


def test_bounds():
    ds = airds.copy(deep=True).cf.add_bounds("lat")
    actual = ds.cf[["lat"]]
    expected = ds[["lat", "lat_bounds"]]
    assert_identical(actual, expected)

    actual = ds.cf[["air"]]
    assert "lat_bounds" in actual.coords

    # can't associate bounds variable when providing scalar keys
    # i.e. when DataArrays are returned
    actual = ds.cf["lat"]
    expected = ds["lat"]
    assert_identical(actual, expected)

    actual = ds.cf.get_bounds("lat")
    expected = ds["lat_bounds"]
    assert_identical(actual, expected)

    # Do not attempt to get bounds when extracting a DataArray
    # raise a warning when extracting a Dataset and bounds do not exists
    ds["time"].attrs["bounds"] = "foo"
    with pytest.warns(None) as record:
        ds.cf["air"]
    assert len(record) == 0
    with pytest.warns(UserWarning, match="{'foo'} not found in object"):
        ds.cf[["air"]]


def test_bounds_to_vertices():
    # All available
    ds = airds.cf.add_bounds(["lon", "lat"])
    dsc = ds.cf.bounds_to_vertices()
    assert "lon_vertices" in dsc
    assert "lat_vertices" in dsc

    # Giving key
    dsc = ds.cf.bounds_to_vertices("longitude")
    assert "lon_vertices" in dsc
    assert "lat_vertices" not in dsc

    dsc = ds.cf.bounds_to_vertices(["longitude", "latitude"])
    assert "lon_vertices" in dsc
    assert "lat_vertices" in dsc

    # Error
    with pytest.raises(ValueError):
        dsc = ds.cf.bounds_to_vertices("T")

    # Words on datetime arrays to
    ds = airds.cf.add_bounds("time")
    dsc = ds.cf.bounds_to_vertices()
    assert "time_bounds" in dsc


def test_get_bounds_dim_name():
    ds = airds.copy(deep=True).cf.add_bounds("lat")
    assert ds.cf.get_bounds_dim_name("latitude") == "bounds"
    assert ds.cf.get_bounds_dim_name("lat") == "bounds"

    assert mollwds.cf.get_bounds_dim_name("longitude") == "bounds"
    assert mollwds.cf.get_bounds_dim_name("lon") == "bounds"


def test_docstring():
    assert "One of ('X'" in airds.cf.groupby.__doc__
    assert "Time variable accessor e.g. 'T.month'" in airds.cf.groupby.__doc__
    assert "One or more of ('X'" in airds.cf.mean.__doc__
    assert "present in .dims" in airds.cf.drop_dims.__doc__
    assert "present in .coords" in airds.cf.integrate.__doc__
    assert "present in .indexes" in airds.cf.resample.__doc__

    # Make sure docs are up to date
    get_all_doc = cf_xarray.accessor._get_all.__doc__
    all_keys = (
        cf_xarray.accessor._AXIS_NAMES
        + cf_xarray.accessor._COORD_NAMES
        + cf_xarray.accessor._CELL_MEASURES
    )
    expected = f"One or more of {all_keys!r}, or arbitrary measures, or standard names"
    assert get_all_doc.split() == expected.split()
    for name in ["dims", "indexes", "coords"]:
        actual = getattr(cf_xarray.accessor, f"_get_{name}").__doc__
        expected = get_all_doc + f" present in .{name}"
        assert actual.split() == expected.split()


def _make_names(prefixes):
    suffixes = ["", "a", "_a", "0", "_0", "a_0a"]
    return [
        f"{prefix}{suffix}" for prefix, suffix in itertools.product(prefixes, suffixes)
    ]


_TIME_NAMES = ["t"] + _make_names(
    [
        "time",
        "min",
        "hour",
        "day",
        "week",
        "month",
        "year",
    ]
)
_VERTICAL_NAMES = _make_names(
    [
        "z",
        "lv_1",
        "bottom_top",
        "sigma",
        "sigma_w",
        "hght",
        "height",
        "altitude",
        "depth",
        "isobaric",
        "pressure",
        "isotherm",
        "gdep",
        "nav_lev",
    ]
)
_X_NAMES = _make_names(["x"])
_Y_NAMES = _make_names(["y"])
_Z_NAMES = _VERTICAL_NAMES
_LATITUDE_NAMES = _make_names(["lat", "latitude", "gphi", "nav_lat"])
_LONGITUDE_NAMES = _make_names(["lon", "longitude", "glam", "nav_lon"])


@pytest.mark.parametrize(
    "kind, names",
    [
        ["X", _X_NAMES],
        ["Y", _Y_NAMES],
        ["Z", _Z_NAMES],
        ["T", _TIME_NAMES],
        ["latitude", _LATITUDE_NAMES],
        ["longitude", _LONGITUDE_NAMES],
    ],
)
def test_guess_coord_axis(kind, names):
    from cf_xarray.accessor import ATTRS

    for varname in names:
        ds = xr.Dataset()
        ds[varname] = (varname, [1, 2, 3, 4, 5])
        dsnew = ds.cf.guess_coord_axis()
        assert dsnew[varname].attrs == ATTRS[kind]

        varname = varname.upper()
        ds[varname] = (varname, [1, 2, 3, 4, 5])
        dsnew = ds.cf.guess_coord_axis()
        assert dsnew[varname].attrs == ATTRS[kind]


def test_guess_coord_axis_datetime():
    ds = xr.Dataset()
    ds["time"] = ("time", pd.date_range("2001-01-01", "2001-04-01"))
    dsnew = ds.cf.guess_coord_axis()
    assert dsnew.time.attrs == {"standard_name": "time", "axis": "T"}


def test_attributes():
    actual = airds.cf.sizes
    expected = {"X": 50, "Y": 25, "T": 4, "longitude": 50, "latitude": 25, "time": 4}
    assert actual == expected

    assert popds.cf.sizes == {"X": 30, "Y": 20}

    with pytest.raises(AttributeError):
        multiple.cf.sizes

    assert airds.cf.chunks == {}

    expected = {
        "X": (50,),
        "Y": (5, 5, 5, 5, 5),
        "T": (4,),
        "longitude": (50,),
        "latitude": (5, 5, 5, 5, 5),
        "time": (4,),
    }
    assert airds.chunk({"lat": 5}).cf.chunks == expected

    with pytest.raises(AttributeError):
        airds.da.cf.chunks

    airds2 = airds.copy(deep=True)
    airds2.lon.attrs = {}
    actual = airds2.cf.sizes
    expected = {"lon": 50, "Y": 25, "T": 4, "latitude": 25, "time": 4}
    assert actual == expected

    actual = popds.cf.data_vars
    expected = {
        "sea_water_x_velocity": popds.cf["UVEL"],
        "sea_water_potential_temperature": popds.cf["TEMP"],
    }
    assert_dicts_identical(actual, expected)

    actual = multiple.cf.data_vars
    expected = dict(multiple.data_vars)
    assert_dicts_identical(actual, expected)

    # check that data_vars contains ancillary variables
    assert_identical(anc.cf.data_vars["specific_humidity"], anc.cf["specific_humidity"])

    # clash between var name and "special" CF name
    # Regression test for #126
    data = np.random.rand(4, 3)
    times = pd.date_range("2000-01-01", periods=4)
    locs = [30, 60, 90]
    coords = [("time", times, {"axis": "T"}), ("space", locs)]
    foo = xr.DataArray(data, coords, dims=["time", "space"])
    ds1 = xr.Dataset({"T": foo})
    assert_identical(ds1.cf.data_vars["T"], ds1["T"])


def test_missing_variable_in_coordinates():
    airds.air.attrs["coordinates"] = "lat lon time"
    with xr.set_options(keep_attrs=True):
        # keep bad coordinates attribute after mean
        assert_identical(airds.time, airds.air.cf.mean(["X", "Y"]).cf["time"])


def test_Z_vs_vertical_ROMS():
    from ..datasets import romsds

    assert_identical(romsds.s_rho.reset_coords(drop=True), romsds.temp.cf["Z"])
    assert_identical(
        romsds.z_rho_dummy.reset_coords(drop=True), romsds.temp.cf["vertical"]
    )

    romsds = romsds.copy(deep=True)

    romsds.temp.attrs.clear()
    # look in encoding
    assert_identical(romsds.s_rho.reset_coords(drop=True), romsds.temp.cf["Z"])
    with pytest.raises(KeyError):
        # z_rho is not in .encoding["coordinates"]
        # so this won't work
        romsds.temp.cf["vertical"]

    # use .coords if coordinates attribute is not available
    romsds.temp.encoding.clear()
    assert_identical(romsds.s_rho.reset_coords(drop=True), romsds.temp.cf["Z"])
    assert_identical(
        romsds.z_rho_dummy.reset_coords(drop=True), romsds.temp.cf["vertical"]
    )


def test_param_vcoord_ocean_s_coord():
    romsds.s_rho.attrs["standard_name"] = "ocean_s_coordinate_g2"
    Zo_rho = (romsds.hc * romsds.s_rho + romsds.Cs_r * romsds.h) / (
        romsds.hc + romsds.h
    )
    expected = romsds.zeta + (romsds.zeta + romsds.h) * Zo_rho
    romsds.cf.decode_vertical_coords()
    assert_allclose(
        romsds.z_rho.reset_coords(drop=True), expected.reset_coords(drop=True)
    )

    romsds.s_rho.attrs["standard_name"] = "ocean_s_coordinate_g1"
    Zo_rho = romsds.hc * (romsds.s_rho - romsds.Cs_r) + romsds.Cs_r * romsds.h
    expected = Zo_rho + romsds.zeta * (1 + Zo_rho / romsds.h)
    romsds.cf.decode_vertical_coords()
    assert_allclose(
        romsds.z_rho.reset_coords(drop=True), expected.reset_coords(drop=True)
    )

    romsds.cf.decode_vertical_coords(prefix="ZZZ")
    assert "ZZZ_rho" in romsds.coords

    copy = romsds.copy(deep=True)
    del copy["zeta"]
    with pytest.raises(KeyError):
        copy.cf.decode_vertical_coords()

    copy = romsds.copy(deep=True)
    copy.s_rho.attrs["formula_terms"] = "s: s_rho C: Cs_r depth: h depth_c: hc"
    with pytest.raises(KeyError):
        copy.cf.decode_vertical_coords()


def test_standard_name_mapper():
    da = xr.DataArray(
        np.arange(6),
        dims="time",
        coords={
            "label": (
                "time",
                ["A", "B", "B", "A", "B", "C"],
                {"standard_name": "standard_label"},
            )
        },
    )

    actual = da.cf.groupby("standard_label").mean()
    expected = da.cf.groupby("label").mean()
    assert_identical(actual, expected)

    actual = da.cf.sortby("standard_label")
    expected = da.sortby("label")
    assert_identical(actual, expected)

    assert cf_xarray.accessor._get_with_standard_name(da, None) == []


@pytest.mark.parametrize("obj", objects)
@pytest.mark.parametrize("attr", ["drop_vars", "set_coords"])
def test_drop_vars_and_set_coords(obj, attr):

    # DataArray object has no attribute set_coords
    if not isinstance(obj, Dataset) and attr == "set_coords":
        return

    # Get attribute
    expected = getattr(obj, attr)
    actual = getattr(obj.cf, attr)

    # Axis
    assert_identical(expected("lon"), actual("X"))
    # Coordinate
    assert_identical(expected("lon"), actual("longitude"))
    # Cell measure
    assert_identical(expected("cell_area"), actual("area"))
    # Variables
    if isinstance(obj, Dataset):
        assert_identical(expected("air"), actual("air_temperature"))
        assert_identical(expected(obj.variables), actual(obj.cf.keys()))


@pytest.mark.parametrize("obj", objects)
def test_drop_sel_and_reset_coords(obj):

    # Axis
    assert_identical(obj.drop_sel(lat=75), obj.cf.drop_sel(Y=75))
    # Coordinate
    assert_identical(obj.drop_sel(lat=75), obj.cf.drop_sel(latitude=75))

    # Cell measure
    assert_identical(obj.reset_coords("cell_area"), obj.cf.reset_coords("area"))
    # Variable
    if isinstance(obj, Dataset):
        assert_identical(
            obj.reset_coords("air"), obj.cf.reset_coords("air_temperature")
        )


@pytest.mark.parametrize("ds", datasets)
def test_drop_dims(ds):

    # Add data_var and coord to test _get_dims
    ds["lon_var"] = ds["lon"]
    ds = ds.assign_coords(lon_coord=ds["lon"])

    # Axis and coordinate
    for cf_name in ["X", "longitude"]:
        assert_identical(ds.drop_dims("lon"), ds.cf.drop_dims(cf_name))


@pytest.mark.parametrize("ds", datasets)
def test_differentiate(ds):

    # Add data_var and coord to test _get_coords
    ds["lon_var"] = ds["lon"]
    ds = ds.assign_coords(lon_coord=ds["lon"])

    # Coordinate
    assert_identical(ds.differentiate("lon"), ds.cf.differentiate("lon"))

    # Multiple coords (test error raised by _single)
    with pytest.raises(KeyError, match=".*I expected only one."):
        assert_identical(ds.differentiate("lon"), ds.cf.differentiate("X"))


def test_new_standard_name_mappers():
    assert_identical(forecast.cf.mean("realization"), forecast.mean("M"))
    assert_identical(
        forecast.cf.mean(["realization", "forecast_period"]), forecast.mean(["M", "L"])
    )
    assert_identical(forecast.cf.chunk({"realization": 1}), forecast.chunk({"M": 1}))
    assert_identical(forecast.cf.isel({"realization": 1}), forecast.isel({"M": 1}))
    assert_identical(forecast.cf.isel(**{"realization": 1}), forecast.isel(**{"M": 1}))
    assert_identical(
        forecast.cf.groupby("forecast_reference_time.month").mean(),
        forecast.groupby("S.month").mean(),
    )


def test_possible_x_y_plot():
    from ..accessor import _possible_x_y_plot

    # choose axes
    assert _possible_x_y_plot(airds.air.isel(time=1), "x") == "lon"
    assert _possible_x_y_plot(airds.air.isel(time=1), "y") == "lat"
    assert _possible_x_y_plot(airds.air.isel(lon=1), "y") == "lat"
    assert _possible_x_y_plot(airds.air.isel(lon=1), "x") == "time"

    # choose coordinates over axes
    assert _possible_x_y_plot(popds.UVEL, "x") == "ULONG"
    assert _possible_x_y_plot(popds.UVEL, "y") == "ULAT"
    assert _possible_x_y_plot(popds.TEMP, "x") == "TLONG"
    assert _possible_x_y_plot(popds.TEMP, "y") == "TLAT"

    assert _possible_x_y_plot(popds.UVEL.drop_vars("ULONG"), "x") == "nlon"

    # choose X over T, Y over Z
    def makeds(*dims):
        coords = {dim: (dim, np.arange(3), {"axis": dim}) for dim in dims}
        return xr.DataArray(np.zeros((3, 3)), dims=dims, coords=coords)

    yzds = makeds("Y", "Z")
    assert _possible_x_y_plot(yzds, "y") == "Z"
    assert _possible_x_y_plot(yzds, "x") is None

    xtds = makeds("X", "T")
    assert _possible_x_y_plot(xtds, "y") is None
    assert _possible_x_y_plot(xtds, "x") == "X"


def test_groupby_special_ops():
    cfgrouped = airds.cf.groupby_bins("latitude", np.arange(20, 50, 10))
    grouped = airds.groupby_bins("lat", np.arange(20, 50, 10))

    # __iter__
    for (label, group), (cflabel, cfgroup) in zip(grouped, cfgrouped):
        assert label == cflabel
        assert_identical(group, cfgroup)

    # arithmetic
    expected = grouped - grouped.mean()
    actual = grouped - cfgrouped.mean()
    assert_identical(expected, actual)
