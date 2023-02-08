import itertools
import pickle
import warnings
from textwrap import dedent
from urllib.request import urlopen

import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from xarray import Dataset
from xarray.testing import assert_allclose, assert_identical

import cf_xarray  # noqa
from cf_xarray.helpers import vertices_to_bounds
from cf_xarray.utils import parse_cf_standard_name_table

from ..datasets import (
    airds,
    ambig,
    anc,
    basin,
    ds_no_attrs,
    ds_with_tuple,
    dsg,
    forecast,
    mollwds,
    multiple,
    pomds,
    popds,
    romsds,
    rotds,
    sgrid_delft,
    sgrid_delft3,
    sgrid_roms,
    vert,
)
from . import (
    raise_if_dask_computes,
    requires_cftime,
    requires_pint,
    requires_regex,
    requires_rich,
    requires_scipy,
)

mpl.use("Agg")

ds = airds
datasets = [airds, airds.chunk({"lat": 5})]
dataarrays = [airds.air, airds.air.chunk({"lat": 5})]
objects = datasets + dataarrays


def assert_dicts_identical(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    for k in dict1:
        assert_identical(dict1[k], dict2[k])


def test_repr() -> None:

    assert "air_temperature: [(1, 2, 3)]" in ds_with_tuple.cf.__repr__()

    # Dataset.
    # Stars: axes, coords, and std names
    actual = airds.cf.__repr__()
    expected = """\
    Coordinates:
                 CF Axes: * X: ['lon']
                          * Y: ['lat']
                          * T: ['time']
                            Z: n/a

          CF Coordinates: * longitude: ['lon']
                          * latitude: ['lat']
                          * time: ['time']
                            vertical: n/a

           Cell Measures:   area: ['cell_area']
                            volume: n/a

          Standard Names: * latitude: ['lat']
                          * longitude: ['lon']
                          * time: ['time']

                  Bounds:   n/a

           Grid Mappings:   n/a

    Data Variables:
           Cell Measures:   area, volume: n/a

          Standard Names:   air_temperature: ['air']

                  Bounds:   n/a

           Grid Mappings:   n/a"""

    assert actual == dedent(expected)

    # DataArray (Coordinates section same as Dataset)
    assert airds.cf.__repr__().startswith(airds["air"].cf.__repr__())
    actual = airds["air"].cf.__repr__()
    expected = """\
    Coordinates:
                 CF Axes: * X: ['lon']
                          * Y: ['lat']
                          * T: ['time']
                            Z: n/a

          CF Coordinates: * longitude: ['lon']
                          * latitude: ['lat']
                          * time: ['time']
                            vertical: n/a

           Cell Measures:   area: ['cell_area']
                            volume: n/a

          Standard Names: * latitude: ['lat']
                          * longitude: ['lon']
                          * time: ['time']

                  Bounds:   n/a

           Grid Mappings:   n/a"""
    assert actual == dedent(expected)

    # Empty Standard Names
    actual = popds.cf.__repr__()
    expected = """\
    Coordinates:
                 CF Axes: * X: ['nlon']
                          * Y: ['nlat']
                            Z, T: n/a

          CF Coordinates:   longitude: ['TLONG', 'ULONG']
                            latitude: ['TLAT', 'ULAT']
                            vertical, time: n/a

           Cell Measures:   area, volume: n/a

          Standard Names:   n/a

                  Bounds:   n/a

           Grid Mappings:   n/a

    Data Variables:
           Cell Measures:   area, volume: n/a

          Standard Names:   sea_water_potential_temperature: ['TEMP']
                            sea_water_x_velocity: ['UVEL']

                  Bounds:   n/a

           Grid Mappings:   n/a"""
    assert actual == dedent(expected)

    # Flag DataArray
    assert "Flag Variable" in repr(basin.cf)

    # "Temp" dataset
    actual = airds["air"]._to_temp_dataset().cf.__repr__()
    expected = """\
    Coordinates:
                 CF Axes: * X: ['lon']
                          * Y: ['lat']
                          * T: ['time']
                            Z: n/a

          CF Coordinates: * longitude: ['lon']
                          * latitude: ['lat']
                          * time: ['time']
                            vertical: n/a

           Cell Measures:   area: ['cell_area']
                            volume: n/a

          Standard Names: * latitude: ['lat']
                          * longitude: ['lon']
                          * time: ['time']

                  Bounds:   n/a

           Grid Mappings:   n/a

    Data Variables:
           Cell Measures:   area, volume: n/a

          Standard Names:   air_temperature: [<this-array>]

                  Bounds:   n/a

           Grid Mappings:   n/a"""
    assert actual == dedent(expected)

    # CF roles
    actual = dsg.cf.__repr__()
    expected = """\
    Discrete Sampling Geometry:
                CF Roles: * profile_id: ['profile']
                          * trajectory_id: ['trajectory']

    Coordinates:
                 CF Axes:   X, Y, Z, T: n/a

          CF Coordinates:   longitude, latitude, vertical, time: n/a

           Cell Measures:   area, volume: n/a

          Standard Names:   n/a

                  Bounds:   n/a

           Grid Mappings:   n/a

    Data Variables:
           Cell Measures:   area, volume: n/a

          Standard Names:   n/a

                  Bounds:   n/a

           Grid Mappings:   n/a"""
    assert actual == dedent(expected)


def test_axes() -> None:
    expected = dict(T=["time"], X=["lon"], Y=["lat"])
    actual = airds.cf.axes
    assert actual == expected

    expected = dict(X=["nlon"], Y=["nlat"])
    actual = popds.cf.axes
    assert actual == expected


def test_coordinates() -> None:
    expected = dict(latitude=["lat"], longitude=["lon"], time=["time"])
    actual = airds.cf.coordinates
    assert actual == expected

    expected = dict(latitude=["TLAT", "ULAT"], longitude=["TLONG", "ULONG"])
    actual = popds.cf.coordinates
    assert actual == expected


@requires_pint
def test_coordinates_quantified() -> None:
    # note: import order is important
    from .. import units  # noqa

    pytest.importorskip("pint_xarray")

    quantified = popds.pint.quantify()
    assert_identical(
        quantified.cf[["latitude"]].pint.dequantify(), popds.cf[["latitude"]]
    )


def test_cell_measures() -> None:
    ds = airds.copy(deep=False)
    ds["foo"] = xr.DataArray(ds["cell_area"], attrs=dict(standard_name="foo_std_name"))
    ds["air"].attrs["cell_measures"] += " foo_measure: foo"
    assert ("foo_std_name" in ds.cf["air_temperature"].cf) and ("foo_measure" in ds.cf)

    ds["air"].attrs["cell_measures"] += " volume: foo"
    ds["foo"].attrs["cell_measures"] = ds["air"].attrs["cell_measures"]
    expected = dict(area=["cell_area"], foo_measure=["foo"], volume=["foo"])
    actual_air = ds.cf["air"].cf.cell_measures
    actual_foo = ds.cf["foo_measure"].cf.cell_measures
    assert actual_air == actual_foo == expected

    actual = ds.cf.cell_measures
    assert actual == expected

    # Additional cell measure in repr
    actual = ds.cf.__repr__()
    expected_repr = """\
    Data Variables:
           Cell Measures:   foo_measure: ['foo']
                            volume: ['foo']
                            area: n/a

          Standard Names:   air_temperature: ['air']
                            foo_std_name: ['foo']

                  Bounds:   n/a

           Grid Mappings:   n/a"""
    assert actual.endswith(dedent(expected_repr))


def test_standard_names() -> None:
    expected = dict(
        air_temperature=["air"], latitude=["lat"], longitude=["lon"], time=["time"]
    )
    actual = airds.cf.standard_names
    assert actual == expected

    dsnew = xr.Dataset()
    dsnew["a"] = ("a", np.arange(10), {"standard_name": "a"})
    dsnew["b"] = ("a", np.arange(10), {"standard_name": "a"})
    assert dsnew.cf.standard_names == dict(a=["a", "b"])


def test_drop_bounds() -> None:
    assert ambig.cf["latitude"].name == "lat"
    assert ambig.cf["longitude"].name == "lon"
    assert ambig.cf.bounds["latitude"] == ["vertices_latitude"]
    assert ambig.cf.bounds["longitude"] == ["vertices_longitude"]


def test_accessor_getattr_and_describe() -> None:
    ds_verta = vert.set_coords(
        (
            "ps",
            "areacella",
        )
    )
    ds_vertb = xr.decode_cf(vert, decode_coords="all")  # type: ignore

    assert ds_verta.cf.cell_measures == ds_vertb.cf.cell_measures
    assert ds_verta.o3.cf.cell_measures == ds_vertb.o3.cf.cell_measures
    assert ds_verta.cf.formula_terms == ds_vertb.cf.formula_terms
    assert ds_verta.o3.cf.formula_terms == ds_vertb.o3.cf.formula_terms
    assert ds_verta.cf.bounds == ds_vertb.cf.bounds
    assert ds_verta.cf.grid_mapping_names == ds_vertb.cf.grid_mapping_names
    assert str(ds_verta.cf) == str(ds_vertb.cf)


def test_accessor_getattr_coordinate_Nonetype() -> None:
    ds_vert = vert
    ds_vert["o3"].encoding["coordinates"] = None
    assert ds_vert.o3.cf["latitude"].name == "lat"
    ds_vert["lat"].encoding["coordinates"] = None
    assert ds_vert.cf["latitude"].name == "lat"


def test_getitem_standard_name() -> None:
    actual = airds.cf["air_temperature"]
    expected = airds["air"]
    assert_identical(actual, expected)

    actual = airds.lat.cf["latitude"]
    expected = airds["lat"]
    assert_identical(actual, expected)

    ds = airds.copy(deep=False)
    ds["air2"] = ds.air
    with pytest.raises(KeyError):
        ds.cf["air_temperature"]
    actual = ds.cf[["air_temperature"]]
    expected_ds = ds[["air", "air2"]]
    assert_identical(actual, expected_ds)

    with pytest.raises(KeyError):
        ds.air.cf["air_temperature"]


def test_getitem_ancillary_variables() -> None:
    expected = anc.set_coords(["q_error_limit", "q_detection_limit"])["q"]
    assert_identical(anc.cf["q"], expected)
    assert_identical(anc.cf["specific_humidity"], expected)

    with pytest.warns(UserWarning):
        anc[["q"]].cf["q"]

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with cf_xarray.set_options(warn_on_missing_variables=False):
            anc[["q"]].cf["q"]

    for k in ["ULONG", "ULAT"]:
        assert k not in popds.cf["TEMP"].coords

    for k in ["TLONG", "TLAT"]:
        assert k not in popds.cf["UVEL"].coords


def test_rename_like() -> None:
    # it'll match for axis: X (lon, nlon) and coordinate="longitude" (lon, TLONG)
    # so delete the axis attributes
    original = popds
    newair = airds.copy(deep=False)
    del newair.lon.attrs["axis"]
    del newair.lat.attrs["axis"]

    renamed = popds.cf["TEMP"].cf.rename_like(newair)
    assert original.TEMP.attrs["coordinates"] == "TLONG TLAT"
    for k in ["TLONG", "TLAT"]:
        assert k in original.coords
        assert k not in renamed.coords

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

    # skip conflicting variables
    da = popds.cf["TEMP"]
    with pytest.warns(UserWarning, match="Conflicting variables skipped:.*"):
        expected = {"longitude": ["TLONG"], "latitude": ["TLAT"]}
        actual = da.cf.rename_like(airds).cf.coordinates
        assert expected == actual
    expected = {"longitude": ["lon"], "latitude": ["lat"]}
    actual = da.cf.rename_like(airds, skip="axes").cf.coordinates
    assert expected == actual

    # rename bounds
    original = airds.cf[["air"]].cf.add_bounds("lon")
    other = popds.cf[["TEMP"]].cf.add_bounds("nlon")
    renamed = original.cf.rename_like(other, skip="coordinates")
    assert renamed.cf.bounds["nlon"] == ["nlon_bounds"]

    # rename cell measures
    other = airds.cf["air"].cf.rename(area="CELL_AREA")
    other.attrs["cell_measures"] = other.attrs["cell_measures"].replace(
        "cell_area", "CELL_AREA"
    )
    renamed = airds.cf["air"].cf.rename_like(other)
    assert renamed.cf.cell_measures["area"] == ["CELL_AREA"]

    renamed = ds_with_tuple.cf.rename_like(airds)
    assert renamed.identical(airds)


@pytest.mark.parametrize("obj", objects)
@pytest.mark.parametrize(
    "attr, xrkwargs, cfkwargs",
    (
        ("resample", {"time": "M"}, {"T": "M"}),
        ("rolling", {"lat": 5}, {"Y": 5}),
        ("groupby", {"group": "time"}, {"group": "T"}),
        ("groupby", {"group": "time.month"}, {"group": "T.month"}),
        ("groupby_bins", {"group": "lat", "bins": 5}, {"group": "latitude", "bins": 5}),
        ("coarsen", {"lon": 2, "lat": 5}, {"X": 2, "Y": 5}),
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
        actuals = [
            obj.cf.weighted("area").sum("Y"),
            obj.cf.weighted(obj["cell_area"]).sum("Y"),
            obj.cf.weighted(weights=obj["cell_area"]).sum("Y"),
        ]
    for actual in actuals:
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


def test_pos_args_methods() -> None:
    expected = airds.transpose("lon", "time", "lat")
    actual = airds.cf.transpose("longitude", "T", "latitude")
    assert_identical(actual, expected)

    actual = airds.cf.transpose("longitude", ...)
    assert_identical(actual, expected)

    expected = multiple.transpose("y2", "y1", "x1", "x2")
    actual = multiple.cf.transpose("Y", "X")
    assert_identical(actual, expected)


def test_preserve_unused_keys() -> None:

    ds = airds.copy(deep=False)
    ds.time.attrs.clear()
    actual = ds.cf.sel(X=260, Y=40, time=airds.time[:2], method="nearest")
    expected = ds.sel(lon=260, lat=40, time=airds.time[:2], method="nearest")
    assert_identical(actual, expected)


def test_kwargs_expand_key_to_multiple_keys() -> None:

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


def test_dataarray_getitem() -> None:

    air = airds.air.copy(deep=False)
    air.name = None

    assert_identical(air.cf["longitude"], air["lon"])
    with pytest.raises(KeyError):
        air.cf[["longitude"]]
    with pytest.raises(KeyError):
        air.cf[["longitude", "latitude"]]

    air["cell_area"].attrs["standard_name"] = "area_grid_cell"
    assert_identical(air.cf["area_grid_cell"], air.cell_area.reset_coords(drop=True))


def test_dataarray_plot() -> None:

    obj = airds.air.copy(deep=False)

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

    obj.lon.attrs["positive"] = "down"
    rv = obj.cf.isel(T=1, Y=[0, 1, 2]).cf.plot(hue="Y")
    np.testing.assert_equal(rv[0].get_xdata(), obj.lon.data)
    xlim = rv[0].axes.get_xlim()
    assert xlim[0] > xlim[1]
    plt.close()
    del obj.lon.attrs["positive"]

    rv = obj.cf.isel(T=1, Y=[0, 1, 2]).cf.plot.line()
    np.testing.assert_equal(rv[0].get_xdata(), obj.lon.data)
    plt.close()

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
        ("area", "cell_area"),
    ),
)
def test_getitem(obj, key, expected_key):
    assert key in obj.cf

    actual = obj.cf[key]
    expected = obj[expected_key].reset_coords(drop=True)
    assert_identical(actual, expected)


@pytest.mark.parametrize("obj", objects)
def test_getitem_errors(obj):
    with pytest.raises(KeyError):
        obj.cf["XX"]
    obj2 = obj.copy(deep=False)
    obj2.lon.attrs = {}
    with pytest.raises(KeyError):
        obj2.cf["X"]


def test_bad_cell_measures_attribute() -> None:
    air2 = airds.copy(deep=False)
    air2.air.attrs["cell_measures"] = "--OPT"
    with pytest.warns(UserWarning):
        air2.cf["air"]
    with pytest.warns(UserWarning):
        assert_identical(air2.air.drop_vars("cell_area"), air2.cf["air"])
    with pytest.warns(UserWarning):
        assert air2.cf.cell_measures == {}
    with pytest.warns(UserWarning):
        assert air2.air.cf.cell_measures == {}
    with pytest.raises(ValueError):
        air2.cf.get_associated_variable_names("air", error=True)
    with pytest.warns(UserWarning):
        air2.cf.get_associated_variable_names("air", error=False)

    #  GH216
    repr(air2.cf)

    # GH340
    air2.cf.axes
    air2.cf.coordinates
    with pytest.warns(UserWarning):
        air2.cf.cell_measures


def test_getitem_clash_standard_name() -> None:
    ds = xr.Dataset()
    ds.coords["area"] = xr.DataArray(np.ones(10), attrs={"standard_name": "cell_area"})
    assert_identical(ds.cf["cell_area"], ds["area"].reset_coords(drop=True))

    ds = xr.Dataset()
    ds["time"] = (
        "time",
        np.arange(10),
        {"standard_name": "time", "bounds": "time_bounds"},
    )
    ds["time_bounds"] = (
        ("time", "bounds"),
        np.ones((10, 2)),
        {"standard_name": "time"},
    )

    ds["lat"] = (
        "lat",
        np.arange(10),
        {"units": "degrees_north", "bounds": "lat_bounds"},
    )
    ds["lat_bounds"] = (
        ("lat", "bounds"),
        np.ones((10, 2)),
        {"units": "degrees_north"},
    )
    assert_identical(ds["lat"], ds.cf["latitude"])


def test_getitem_uses_coordinates() -> None:
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


def test_getitem_uses_dimension_names_when_coordinates_attr() -> None:
    # POP-like dataset
    ds = popds
    assert_identical(ds.cf["X"], ds["nlon"])
    assert_identical(ds.cf["Y"], ds["nlat"])
    assert_identical(ds.UVEL.cf["X"], ds["nlon"])
    assert_identical(ds.TEMP.cf["Y"], ds["nlat"])


def test_plot_xincrease_yincrease() -> None:
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4), lon=slice(50))
    ds.lon.attrs["positive"] = "down"
    ds.lat.attrs["positive"] = "down"

    _, ax = plt.subplots(1, 1)
    ds.air.isel(time=1).cf.plot(ax=ax, x="X", y="Y")

    for lim in [ax.get_xlim(), ax.get_ylim()]:
        assert lim[0] > lim[1]


@pytest.mark.parametrize("dims", ["time2", "lat", "time", ["lat", "lon"]])
def test_add_bounds(dims):
    ds = airds
    original = ds.copy(deep=True)

    expected = {}
    expected["lat"] = xr.concat(
        [
            ds.lat.copy(data=np.arange(76.25, 16.0, -2.5)),
            ds.lat.copy(data=np.arange(73.75, 13.6, -2.5)),
        ],
        dim="bounds",
    )
    expected["lon"] = xr.concat(
        [
            ds.lon.copy(data=np.arange(198.75, 325 - 1.25, 2.5)),
            ds.lon.copy(data=np.arange(201.25, 325 + 1.25, 2.5)),
        ],
        dim="bounds",
    )
    t0 = pd.Timestamp("2013-01-01")
    t1 = pd.Timestamp("2013-01-01 18:00")
    dt = "6h"
    dtb2 = pd.Timedelta("3h")
    expected["time"] = xr.concat(
        [
            ds.time.copy(data=pd.date_range(start=t0 - dtb2, end=t1 - dtb2, freq=dt)),
            ds.time.copy(data=pd.date_range(start=t0 + dtb2, end=t1 + dtb2, freq=dt)),
        ],
        dim="bounds",
    )
    expected["time2"] = expected["time"]
    expected["lat"].attrs.clear()
    expected["lon"].attrs.clear()
    expected["time"].attrs.clear()

    added = ds.copy(deep=False)
    added.coords["time2"] = ds.time
    added = added.cf.add_bounds(dims)
    if isinstance(dims, str):
        dims = (dims,)

    for dim in dims:
        name = f"{dim}_bounds"
        assert name in added.coords
        assert added[dim].attrs["bounds"] == name
        assert_allclose(
            added[name].reset_coords(drop=True), expected[dim].transpose(..., "bounds")
        )

    _check_unchanged(original, ds)


def test_add_bounds_multiple() -> None:
    # Test multiple dimensions
    assert not {"x1_bounds", "x2_bounds"} <= set(multiple.variables)
    assert {"x1_bounds", "x2_bounds"} <= set(multiple.cf.add_bounds("X").variables)


def test_add_bounds_nd_variable() -> None:
    ds = xr.Dataset(
        {"z": (("x", "y"), np.arange(12).reshape(4, 3))},
        coords={"x": np.arange(4), "y": np.arange(3)},
    )

    # 2D
    expected = (
        vertices_to_bounds(
            np.arange(0, 13, 3).reshape(5, 1) + np.arange(-2, 2).reshape(1, 4)  # type: ignore
        )
        .rename("z_bounds")
        .assign_coords(**ds.coords)
    )
    actual = ds.cf.add_bounds("z").z_bounds.reset_coords(drop=True)
    xr.testing.assert_identical(actual, expected)

    # 2D rotated ds
    lon_bounds = (
        rotds.drop_vars(["lon_bounds"])
        .assign(x=rotds.cf["X"], y=rotds.cf["Y"])
        .cf.add_bounds(["lon"])
        .lon_bounds
    )
    # upper left of cell must be the EXACT same as the lower left of the cell above
    assert lon_bounds[0, 1, 1] == lon_bounds[0, 2, 0]
    # upper right of cell must be the EXACT same as the lower right of the cell above
    assert lon_bounds[0, 1, 2] == lon_bounds[0, 2, 3]

    # 1D
    expected = (
        xr.concat([ds.z - 1.5, ds.z + 1.5], dim="bounds")
        .rename("z_bounds")
        .transpose("bounds", "y", "x")
    )

    actual = ds.cf.add_bounds("z", dim="x").z_bounds.reset_coords(drop=True)
    xr.testing.assert_identical(expected.transpose(..., "bounds"), actual)

    # Requesting bounds on a non-variable dimension
    with pytest.raises(ValueError, match="are dimensions with no index."):
        ds.drop_vars("x").cf.add_bounds("x")

    with pytest.raises(ValueError, match="The `bounds` dimension already exists"):
        ds.cf.add_bounds("z").cf.add_bounds("x")


def test_bounds() -> None:
    ds = airds.copy(deep=False).cf.add_bounds("lat")

    actual = ds.cf.bounds
    expected = {"Y": ["lat_bounds"], "lat": ["lat_bounds"], "latitude": ["lat_bounds"]}
    assert ds.cf.bounds == expected

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
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ds.cf["air"]

    with pytest.warns(UserWarning, match="{'foo'} not found in object"):
        ds.cf[["air"]]

    # Dataset has bounds
    expected_repr = """\
              Bounds:   Y: ['lat_bounds']
                        lat: ['lat_bounds']
                        latitude: ['lat_bounds']"""
    assert expected_repr in ds.cf.__repr__()

    # DataArray does not have bounds
    expected_repr = airds.cf["air"].cf.__repr__()
    actual = ds.cf["air"].cf.__repr__()
    assert actual == expected_repr


def test_bounds_to_vertices() -> None:
    # All available
    ds = airds
    original = ds.copy(deep=True)
    dsb = ds.cf.add_bounds(["lon", "lat"])
    dsv = dsb.cf.bounds_to_vertices()
    assert "lon_vertices" in dsv
    assert "lat_vertices" in dsv

    # Giving key
    dsv = dsb.cf.bounds_to_vertices("longitude")
    assert "lon_vertices" in dsv
    assert "lat_vertices" not in dsv

    dsv = dsb.cf.bounds_to_vertices(["longitude", "latitude"])
    assert "lon_vertices" in dsv
    assert "lat_vertices" in dsv

    # Error
    with pytest.raises(ValueError):
        dsv = dsb.cf.bounds_to_vertices("T")

    # Words on datetime arrays to
    dsb = dsb.cf.add_bounds("time")
    dsv = dsb.cf.bounds_to_vertices()
    assert "time_bounds" in dsv

    _check_unchanged(original, ds)


def test_get_bounds_dim_name() -> None:
    ds = airds.cf.add_bounds("lat")
    assert ds.cf.get_bounds_dim_name("latitude") == "bounds"
    assert ds.cf.get_bounds_dim_name("lat") == "bounds"

    assert mollwds.cf.get_bounds_dim_name("longitude") == "bounds"
    assert mollwds.cf.get_bounds_dim_name("lon") == "bounds"


def test_grid_mappings():
    ds = rotds.copy(deep=False)

    actual = ds.cf.grid_mapping_names
    expected = {"rotated_latitude_longitude": ["rotated_pole"]}
    assert actual == expected

    expected = ds.rotated_pole
    actual = ds.cf["rotated_latitude_longitude"]
    assert_identical(actual, expected)

    expected = ds.rotated_pole
    actual = ds.cf["grid_mapping"]
    assert_identical(actual, expected)

    # Propagation
    actual = ds.cf[["temp"]]
    assert "rotated_pole" in actual.coords

    actual = ds.cf["temp"]
    assert "rotated_pole" in actual.coords

    actual = ds.cf["temp"].cf["grid_mapping"]
    assert_identical(actual, expected)

    actual = ds.cf["temp"].coords["rotated_pole"].drop_vars("rotated_pole")
    assert_identical(actual, expected)

    actual = ds.cf["temp"].cf["rotated_latitude_longitude"]
    assert_identical(actual, expected)

    # Test repr
    expected = """\
           Grid Mappings:   rotated_latitude_longitude: ['rotated_pole']"""
    assert dedent(expected) in ds.cf.__repr__()
    # assert dedent(expected) in ds.cf["temp"].cf.__repr__()

    # grid_mapping_name
    assert ds.cf["temp"].cf.grid_mapping_name == "rotated_latitude_longitude"

    # what if there are really 2 grid mappins?
    ds["temp2"] = ds.temp
    ds["temp2"].attrs["grid_mapping"] = "rotated_pole2"
    ds["rotated_pole2"] = ds.rotated_pole
    expected = """\
           Grid Mappings:   rotated_latitude_longitude: ['rotated_pole', 'rotated_pole2']"""
    assert dedent(expected) in ds.cf.__repr__()

    with pytest.raises(KeyError):
        ds.cf["grid_mapping"]

    assert "rotated_latitude_longitude" in ds.cf.keys()

    with pytest.raises(KeyError):
        ds.cf["grid_mapping"]
    actual = ds.cf[["grid_mapping"]]
    expected = ds[["rotated_pole", "rotated_pole2"]].reset_coords()
    assert_identical(expected, actual)

    expected = ds.rotated_pole
    actual = ds.cf["temp"].cf["grid_mapping"]
    assert_identical(actual, expected)
    expected = ds.rotated_pole2
    actual = ds.cf["temp2"].cf["grid_mapping"]
    assert_identical(actual, expected)

    # test _get_all with grid_mapping_var mapper
    ds = ds.cf.set_coords("grid_mapping")
    assert "rotated_pole" in ds.coords


def test_bad_grid_mapping_attribute():
    ds = rotds.copy(deep=False)
    ds.temp.attrs["grid_mapping"] = "foo"
    # warning when extracting a Datarray and grid_mapping does not exist
    with pytest.warns(UserWarning):
        ds.cf["temp"]
    # warning when extracting a Dataset and grid_mapping does not exist
    with pytest.warns(UserWarning):
        ds.cf[["temp"]]
    # this should probably also raise a warning (like cell_measures)
    # with pytest.warns(UserWarning):
    #    assert ds.cf.grid_mappings == {}
    with pytest.warns(UserWarning):
        ds.cf.get_associated_variable_names("temp", error=False)


def test_docstring() -> None:
    assert "One of ('X'" in airds.cf.groupby.__doc__
    assert "Time variable accessor e.g. 'T.month'" in airds.cf.groupby.__doc__
    assert "One or more of ('X'" in airds.cf.mean.__doc__
    assert "present in .dims" in airds.cf.drop_dims.__doc__
    assert "present in .coords" in airds.cf.integrate.__doc__
    assert "present in .indexes" in airds.cf.resample.__doc__

    # Make sure docs are up to date
    get_all_doc: str = cf_xarray.accessor._get_all.__doc__  # type: ignore
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


def _check_unchanged(old, new):
    # Check data array attributes or global dataset attributes
    def _check_attrs_equal(o, n):
        # Compare values and keys,
        # But not ids : allow for copied values (see #365)
        assert o.keys() == n.keys()
        for k, v in o.items():
            if isinstance(v, np.ndarray):
                assert np.all(v == n[k])
            else:
                assert v == n[k]

    assert type(old) == type(new)
    _check_attrs_equal(old.attrs, new.attrs)

    # Check coordinate attributes and data variable attributes
    dicts = [(old.coords, new.coords)]
    if isinstance(old, xr.Dataset):
        dicts.append((old.data_vars, new.data_vars))
    for old_dict, new_dict in dicts:
        assert old_dict.keys() == new_dict.keys()  # set comparison
        for key, old_obj in old_dict.items():
            new_obj = new_dict[key]
            _check_attrs_equal(old_obj.attrs, new_obj.attrs)


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
_X_NAMES = _make_names(["x", "nlon", "i", "ni"])
_Y_NAMES = _make_names(["y", "nlat", "j", "nj"])
_Z_NAMES = _VERTICAL_NAMES + ["olevel", "level", "zlevel"]
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


def test_guess_coord_axis_datetime() -> None:
    ds = xr.Dataset()
    ds["time"] = ("time", pd.date_range("2001-01-01", "2001-04-01"))
    dsnew = ds.cf.guess_coord_axis()
    assert dsnew.time.attrs == {"standard_name": "time", "axis": "T"}


def test_attributes() -> None:
    actual = airds.cf.sizes
    expected = {"X": 50, "Y": 25, "T": 4, "longitude": 50, "latitude": 25, "time": 4}
    assert actual == expected

    assert popds.cf.sizes == {"X": 30, "Y": 20}

    with pytest.raises(AttributeError):
        multiple.cf.sizes

    assert airds.cf.chunks == {}

    expected_chunks = {
        "X": (50,),
        "Y": (5, 5, 5, 5, 5),
        "T": (4,),
        "longitude": (50,),
        "latitude": (5, 5, 5, 5, 5),
        "time": (4,),
    }
    assert airds.chunk({"lat": 5}).cf.chunks == expected_chunks
    assert airds.chunk({"lat": 5}).cf.chunksizes == expected_chunks

    with pytest.raises(AttributeError):
        airds.da.cf.chunks

    airds2 = airds.copy(deep=False)
    airds2.lon.attrs = {}
    actual = airds2.cf.sizes
    expected_sizes = {"lon": 50, "Y": 25, "T": 4, "latitude": 25, "time": 4}
    assert actual == expected_sizes

    actual = popds.cf.data_vars
    expected_data_vars = {
        "sea_water_x_velocity": popds.cf["UVEL"],
        "sea_water_potential_temperature": popds.cf["TEMP"],
    }
    assert_dicts_identical(actual, expected_data_vars)

    actual = multiple.cf.data_vars
    assert_dicts_identical(actual, dict(multiple.data_vars))

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

    # multiple latitudes but only one latitude data_var
    ds = popds.copy(deep=False)
    for var in ["ULAT", "TLAT"]:
        ds[var].attrs["standard_name"] = "latitude"
    ds = ds.reset_coords("ULAT")
    assert_identical(ds.cf.data_vars["latitude"], ds.cf["ULAT"])


def test_missing_variable_in_coordinates() -> None:
    airds.air.attrs["coordinates"] = "lat lon time"
    with xr.set_options(keep_attrs=True):
        # keep bad coordinates attribute after mean
        assert_identical(airds.time, airds.air.cf.mean(["X", "Y"]).cf["time"])


def test_Z_vs_vertical_ROMS() -> None:
    from ..datasets import romsds

    assert_identical(romsds.s_rho.reset_coords(drop=True), romsds.temp.cf["Z"])
    assert_identical(
        romsds.z_rho_dummy.reset_coords(drop=True), romsds.temp.cf["vertical"]
    )

    romsds = romsds.copy(deep=False)

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


def test_param_vcoord_ocean_s_coord() -> None:
    romsds.s_rho.attrs["standard_name"] = "ocean_s_coordinate_g2"
    Zo_rho = (romsds.hc * romsds.s_rho + romsds.Cs_r * romsds.h) / (
        romsds.hc + romsds.h
    )
    expected = romsds.zeta + (romsds.zeta + romsds.h) * Zo_rho
    romsds.cf.decode_vertical_coords(outnames={"s_rho": "z_rho"})
    assert_allclose(
        romsds.z_rho.reset_coords(drop=True), expected.reset_coords(drop=True)
    )

    romsds.s_rho.attrs["standard_name"] = "ocean_s_coordinate_g1"
    Zo_rho = romsds.hc * (romsds.s_rho - romsds.Cs_r) + romsds.Cs_r * romsds.h

    expected = Zo_rho + romsds.zeta * (1 + Zo_rho / romsds.h)
    romsds.cf.decode_vertical_coords(outnames={"s_rho": "z_rho"})
    assert_allclose(
        romsds.z_rho.reset_coords(drop=True), expected.reset_coords(drop=True)
    )

    romsds.cf.decode_vertical_coords(outnames={"s_rho": "ZZZ_rho"})
    assert "ZZZ_rho" in romsds.coords

    copy = romsds.copy(deep=False)
    del copy["zeta"]
    with pytest.raises(KeyError):
        copy.cf.decode_vertical_coords(outnames={"s_rho": "z_rho"})

    copy = romsds.copy(deep=False)
    copy.s_rho.attrs["formula_terms"] = "s: s_rho C: Cs_r depth: h depth_c: hc"
    with pytest.raises(KeyError):
        copy.cf.decode_vertical_coords(outnames={"s_rho": "z_rho"})


def test_param_vcoord_ocean_sigma_coordinate() -> None:
    expected = pomds.zeta + pomds.sigma * (pomds.depth + pomds.zeta)
    pomds.cf.decode_vertical_coords(outnames={"sigma": "z"})
    assert_allclose(pomds.z.reset_coords(drop=True), expected.reset_coords(drop=True))

    copy = pomds.copy(deep=False)
    del copy["zeta"]
    with pytest.raises(AssertionError):
        copy.cf.decode_vertical_coords()

    with pytest.raises(KeyError):
        copy.cf.decode_vertical_coords(outnames={})


def test_formula_terms() -> None:
    srhoterms = {
        "s": "s_rho",
        "C": "Cs_r",
        "eta": "zeta",
        "depth": "h",
        "depth_c": "hc",
    }
    assert romsds.cf.formula_terms == {"s_rho": srhoterms}
    assert romsds["temp"].cf.formula_terms == srhoterms
    assert romsds["s_rho"].cf.formula_terms == srhoterms

    s_rho = romsds["s_rho"].copy(deep=False)
    del s_rho.attrs["standard_name"]
    del s_rho.s_rho.attrs["standard_name"]  # TODO: xarray bug
    assert s_rho.cf.formula_terms == srhoterms

    with pytest.raises(KeyError):
        # x,y,t variable
        romsds["zeta"].cf.formula_terms


def test_standard_name_mapper() -> None:
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
    if isinstance(obj, Dataset) and "air" in obj.data_vars:
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
    if isinstance(obj, Dataset) and "air" in obj.data_vars:
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


@pytest.mark.parametrize("obj", objects)
def test_rename(obj):

    cf_dict = {
        "air_temperature" if isinstance(obj, Dataset) else "longitude": "renamed"
    }
    xr_dict = {
        "air"
        if isinstance(obj, Dataset) and "air" in obj.data_vars
        else "lon": "renamed"
    }
    assert_identical(obj.rename(xr_dict), obj.cf.rename(cf_dict))
    assert_identical(obj.rename(**xr_dict), obj.cf.rename(**cf_dict))


def test_rename_tuple():
    obj = ds_with_tuple

    cf_dict = {"air_temperature": "renamed"}
    xr_dict = {(1, 2, 3): "renamed"}
    assert_identical(obj.rename(xr_dict), obj.cf.rename(cf_dict))


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


def test_new_standard_name_mappers() -> None:
    assert_identical(forecast.cf.mean("realization"), forecast.mean("M"))
    assert_identical(
        forecast.cf.mean(["realization", "forecast_period"]), forecast.mean(["M", "L"])
    )
    assert_identical(forecast.cf.chunk({"realization": 1}), forecast.chunk({"M": 1}))
    assert_identical(forecast.cf.isel({"realization": 1}), forecast.isel({"M": 1}))
    assert_identical(forecast.cf.isel(**{"realization": 1}), forecast.isel(**{"M": 1}))  # type: ignore
    assert_identical(
        forecast.cf.groupby("forecast_reference_time.month").mean(),
        forecast.groupby("S.month").mean(),
    )


def test_possible_x_y_plot() -> None:
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

    xtds.coords["lon"] = ("X", [1, 2, 3], {"standard_name": "longitude"})
    # skip lon (which is 1D on X) if user passes hue="X"
    # choose T instead which is a different dimension
    # (and so a more meaningful plot)
    assert _possible_x_y_plot(xtds, "x", skip="X") == "T"
    # now with hue="lon", skip "X"
    assert _possible_x_y_plot(xtds, "x", skip="lon") == "T"


def test_groupby_special_ops() -> None:
    cfgrouped = airds.cf.groupby_bins("latitude", np.arange(20, 50, 10))
    grouped = airds.groupby_bins("lat", np.arange(20, 50, 10))

    # __iter__
    for (label, group), (cflabel, cfgroup) in zip(grouped, cfgrouped):
        assert label == cflabel
        assert_identical(group, cfgroup)

    # arithmetic
    # TODO: Extremely buggy!
    # 1. cfgrouped - cfgrouped.mean() raises RecursionError
    # 2. use_flox=True doesn't preserve attributes for some reason
    with xr.set_options(keep_attrs=True):
        expected = grouped - grouped.mean()
        actual = grouped - cfgrouped.mean()
    assert_identical(expected, actual)


@pytest.mark.parametrize("obj", objects)
def test_stack(obj):
    expected = obj.stack(latlon=["lat", "lon"])
    actual = obj.cf.stack(latlon=["latitude", "longitude"])
    assert_identical(expected, actual)

    actual = obj.cf.stack({"latlon": ["latitude", "longitude"]})
    assert_identical(expected, actual)


da = xr.DataArray(
    np.arange(10)[::-1],  # like ocean temperature
    dims="z",
    coords={"z": ("z", np.arange(10))},
    name="test",
)


@pytest.mark.parametrize("obj", [da, da.to_dataset()])
def test_differentiate_positive_upward(obj):
    obj.z.attrs["positive"] = "down"
    expected = obj.differentiate("z", 2)
    actual = obj.cf.differentiate("z", 2)
    assert_identical(expected, actual)

    obj.z.attrs["positive"] = "up"
    expected = obj.differentiate("z", 2)
    actual = obj.cf.differentiate("z", 2, positive_upward=True)
    assert_identical(expected, actual)

    obj.z.attrs["positive"] = "down"
    expected = -1 * obj.differentiate("z", 2)
    actual = obj.cf.differentiate("z", 2, positive_upward=True)
    assert_identical(expected, actual)

    obj = obj.isel(z=slice(None, None, -1))
    expected = -1 * obj.differentiate("z", 2)
    actual = obj.cf.differentiate("z", 2, positive_upward=True)
    assert_identical(expected, actual)
    obj = obj.isel(z=slice(None, None, -1))

    with xr.set_options(keep_attrs=True):
        da["z"] = obj.z * -1
    expected = -1 * obj.differentiate("z", 2)
    actual = obj.cf.differentiate("z", 2, positive_upward=True)
    assert_identical(expected, actual)

    obj = obj.isel(z=slice(None, None, -1))
    expected = -1 * obj.differentiate("z", 2)
    actual = obj.cf.differentiate("z", 2, positive_upward=True)
    assert_identical(expected, actual)

    del obj.z.attrs["positive"]
    with pytest.raises(ValueError):
        obj.cf.differentiate("z", positive_upward=True)

    obj.z.attrs["positive"] = "zzz"
    with pytest.raises(ValueError):
        obj.cf.differentiate("z", positive_upward=True)


def test_cmip6_attrs() -> None:
    da = xr.DataArray(
        np.ones((10, 10)),
        dims=("nlon", "nlat"),
        coords={
            "nlon": (
                "nlon",
                np.arange(10),
                {"long_name": "cell index along first dimension"},
            ),
            "nlat": (
                "nlat",
                np.arange(10),
                {"long_name": "cell index along second dimension"},
            ),
        },
    )
    assert da.cf.axes["X"] == ["nlon"]
    assert da.cf.axes["Y"] == ["nlat"]


def test_custom_criteria() -> None:
    my_custom_criteria = {
        "ssh": {
            "standard_name": "sea_surface_elev*|sea_surface_height",
            "name": "sea_surface_elevation$",  # variable name
        },
        "salt": {
            "standard_name": "salinity",
            "name": "sal*",
        },
        "wind_speed": {
            "standard_name": "wind_speed$",
        },
    }
    my_custom_criteria2 = {"temp": {"name": "temperature"}}
    my_custom_criteria_list = [my_custom_criteria, my_custom_criteria2]
    my_custom_criteria_tuple = (my_custom_criteria, my_custom_criteria2)

    cf_xarray.set_options(custom_criteria=my_custom_criteria)

    # Match by name regex match
    ds = xr.Dataset()
    ds["salinity"] = ("dim", np.arange(10))
    assert_identical(ds.cf["salt"], ds["salinity"])

    # match by name of coords for DataArray (#378)
    da = xr.DataArray([0, 1], coords={"salinity": [2, 3]})
    assert_identical(da.cf["salt"], da["salinity"])

    # Match by standard_name regex match
    ds = xr.Dataset()
    ds["elev"] = ("dim", np.arange(10), {"standard_name": "sea_surface_elevBLAH"})
    assert_identical(ds.cf["ssh"], ds["elev"])

    # Match by standard_name exact match
    ds = xr.Dataset()
    ds["salinity"] = ("dim", np.arange(10), {"standard_name": "salinity"})
    assert_identical(ds.cf["salt"], ds["salinity"])

    # If not exact name, won't match
    ds = xr.Dataset()
    ds["sea_surface_elevation123"] = ("dim", np.arange(10))
    # Since this will not match, this should error
    with pytest.raises(KeyError):
        ds.cf["ssh"]

    # will select only one variable here since exact match
    ds = xr.Dataset()
    ds["winds"] = ("dim", np.arange(10), {"standard_name": "wind_speed"})
    ds["gusts"] = ("dim", np.arange(10), {"standard_name": "wind_speed_of_gust"})
    assert_identical(ds.cf["wind_speed"], ds["winds"])

    # Match by exact name
    ds = xr.Dataset()
    ds["sea_surface_elevation"] = ("dim", np.arange(10))
    ds["sea_surface_height"] = (
        "dim",
        np.arange(10),
        {"standard_name": "sea_surface_elevBLAH"},
    )
    # Since there are two variables, this should error
    with pytest.raises(KeyError):
        ds.cf["ssh"]
    # But the following should work instead given the two ssh variables
    assert_identical(
        ds.cf[["ssh"]], ds[["sea_surface_elevation", "sea_surface_height"]]
    )

    # test criteria list of dicts
    with cf_xarray.set_options(custom_criteria=my_custom_criteria_list):
        ds = xr.Dataset()
        ds["temperature"] = ("dim", np.arange(10))
        assert_identical(ds.cf["temp"], ds["temperature"])

    # test criteria tuple of dicts
    with cf_xarray.set_options(custom_criteria=my_custom_criteria_tuple):
        ds = xr.Dataset()
        ds["temperature"] = ("dim", np.arange(10))
        assert_identical(ds.cf["temp"], ds["temperature"])


@requires_regex
def test_regex_match():
    # test that having a global regex expression flag later in the expression will work if
    # regex is found
    vocab = {"temp": {"name": "tem|(?i)temp"}}
    ds = xr.Dataset()
    ds["Tempblah"] = [0, 1, 2]
    with cf_xarray.set_options(custom_criteria=vocab):
        assert_identical(ds.cf["temp"], ds["Tempblah"])


def test_cf_standard_name_table_version() -> None:

    url = (
        "https://raw.githubusercontent.com/cf-convention/cf-convention.github.io/"
        "master/Data/cf-standard-names/current/src/cf-standard-name-table.xml"
    )
    expected_info, _, _ = parse_cf_standard_name_table(urlopen(url))
    actual_info, _, _ = parse_cf_standard_name_table()
    assert expected_info == actual_info


def test_add_canonical_attributes_0_dim() -> None:
    """test if works for variables with 0 dimension"""
    xr.DataArray(
        0, attrs={"standard_name": "sea_water_potential_temperature"}
    ).cf.add_canonical_attributes()


@requires_cftime
@pytest.mark.parametrize("reshape", [False, True])
def test_datetime_like(reshape):
    """test for 0 or >= 2 time dimensions"""
    import cftime

    data = cftime.datetime(2022, 1, 12)
    if reshape:
        data = [[data]]
    da = xr.DataArray(
        data,
        attrs={"standard_name": "sea_water_age_since_surface_contact"},
    )
    new_attrs = da.cf.add_canonical_attributes().attrs
    assert "units" not in new_attrs and "description" in new_attrs


@pytest.mark.parametrize("override", [True, False])
@pytest.mark.parametrize("skip", ["units", None])
@pytest.mark.parametrize("verbose", [True, False])
def test_add_canonical_attributes(override, skip, verbose, capsys):

    ds = airds
    original = ds.copy(deep=True)
    cf_ds = ds.cf.add_canonical_attributes(
        override=override, skip=skip, verbose=verbose
    )

    # Catch print
    captured = capsys.readouterr()
    if not verbose:
        captured.out == ""

    # Attributes have been added
    for var in sum(ds.cf.standard_names.values(), []):
        assert set(ds[var].attrs) < set(cf_ds[var].attrs)

    # Time units did not change
    assert ds["time"].attrs.get("units") is cf_ds["time"].attrs.get("units") is None

    # Check override, skip, and verbose
    if not override or skip:
        assert cf_ds["lat"].attrs["units"] == "degrees_north"
        assert "* units" not in captured.out
    else:
        assert cf_ds["lat"].attrs["units"] == "degree_north"
        if verbose:
            assert "* units: degree_north" in captured.out

    # History
    assert (
        f"cf.add_canonical_attributes(override={override!r}, skip={skip!r}, verbose={verbose!r}, source=None)"
        in cf_ds.attrs["history"]
    )

    # DataArray (test only once)
    if override and skip and verbose:
        cf_da = ds["air"].cf.add_canonical_attributes(
            override=override, skip=skip, verbose=verbose
        )

        time_stamp_size = 24
        assert (
            cf_da.attrs["history"][time_stamp_size:]
            == cf_ds.attrs["history"][time_stamp_size:]
        )

        cf_da.attrs.pop("history")
        assert_identical(cf_da, cf_ds["air"])

    _check_unchanged(original, ds)


@pytest.mark.parametrize("op", ["ge", "gt", "eq", "ne", "le", "lt"])
def test_flag_features(op):
    actual = getattr(basin.cf, f"__{op}__")("atlantic_ocean")
    expected = getattr(basin, f"__{op}__")(1)
    assert_identical(actual, expected)


def test_flag_isin() -> None:
    actual = basin.cf.isin(["atlantic_ocean", "pacific_ocean"])
    expected = basin.isin([1, 2])
    assert_identical(actual, expected)


def test_flag_errors() -> None:
    with pytest.raises(ValueError):
        basin.cf.isin(["arctic_ocean"])

    with pytest.raises(ValueError):
        basin.cf == "arctic_ocean"

    ds = xr.Dataset({"basin": basin})
    with pytest.raises(ValueError):
        ds.cf.isin(["atlantic_ocean"])

    basin_ = basin.copy(deep=True)
    basin_.attrs.pop("flag_values")
    with pytest.raises(ValueError):
        basin_.cf.isin(["pacific_ocean"])

    with pytest.raises(ValueError):
        basin_.cf == "pacific_ocean"


def test_missing_variables() -> None:

    # Bounds
    ds = mollwds.copy(deep=False)
    ds = ds.drop_vars("lon_bounds")
    assert ds.cf.bounds == {"lat": ["lat_bounds"], "latitude": ["lat_bounds"]}

    with pytest.raises(KeyError, match=r"No results found for 'longitude'."):
        ds.cf.get_bounds("longitude")

    # Cell measures
    ds = airds.copy(deep=False)
    ds = ds.drop_vars("cell_area")
    assert ds.cf.cell_measures == {}

    # Formula terms
    ds = vert.copy(deep=False)
    ds = ds.drop_vars("ap")
    assert ds.cf.formula_terms == {"lev": {"b": "b", "ps": "ps"}}


def test_pickle() -> None:
    da = xr.DataArray([1.0], name="a")
    ds = da.to_dataset()
    pickle.loads(pickle.dumps(da.cf))
    pickle.loads(pickle.dumps(ds.cf))


def test_cf_role() -> None:
    for name in ["profile_id", "trajectory_id"]:
        assert name in dsg.cf.keys()

    assert dsg.cf.cf_roles == {
        "profile_id": ["profile"],
        "trajectory_id": ["trajectory"],
    }

    dsg.foo.cf.plot(x="profile_id")
    dsg.foo.cf.plot(x="trajectory_id")


def test_grid_topology() -> None:
    ds = xr.Dataset(
        data_vars={},
        coords={
            "mesh": (tuple(), 1, {"cf_role": "mesh_topology"}),
            "grid": (tuple(), 1, {"cf_role": "grid_topology"}),
        },
    )
    assert_identical(ds.cf["grid_topology"], ds.grid.reset_coords(drop=True))
    assert_identical(ds.cf["mesh_topology"], ds.mesh.reset_coords(drop=True))


@requires_scipy
def test_curvefit() -> None:
    from cf_xarray.datasets import airds

    def line(time, slope):
        t = (time - time[0]).astype(float)
        return slope * t

    actual = airds.air.cf.isel(lat=4, lon=5).curvefit(coords=("time",), func=line)
    expected = airds.air.cf.isel(lat=4, lon=5).cf.curvefit(coords="T", func=line)
    assert_identical(expected, actual)

    def plane(coords, slopex, slopey):
        x, y = coords
        return slopex * (x - x.mean()) + slopey * (y - y.mean())

    actual = airds.air.isel(time=0).curvefit(coords=("lat", "lon"), func=plane)
    expected = airds.air.isel(time=0).cf.curvefit(
        coords=["latitude", "longitude"], func=plane
    )
    assert_identical(expected, actual)


@requires_rich
@pytest.mark.parametrize(
    "obj, contains",
    (
        [airds.air, "Coordinates"],
        [popds, "Data Variables"],
        [basin, "Flag Variable"],
        [dsg, "Discrete Sampling Geometry"],
    ),
)
def test_rich_repr(obj, contains):
    from rich.console import Console

    console = Console()
    console.begin_capture()
    console.print(obj.cf)
    printed = console.end_capture()

    assert contains in printed


def test_sgrid():
    from ..sgrid import parse_axes

    positions = ["psi", "u", "v", "rho"]
    expected = {
        "X": {f"xi_{pos}" for pos in positions},
        "Y": {f"eta_{pos}" for pos in positions},
        "Z": {"s_rho", "s_w"},
    }
    assert parse_axes(sgrid_roms) == expected
    assert sgrid_roms.cf.axes == {"X": ["xi_u"], "Y": ["eta_u"]}

    assert "grid" in sgrid_roms.cf["u"].coords
    assert sgrid_roms.cf.get_associated_variable_names("u")["grid"] == ["grid"]

    assert_identical(sgrid_roms.cf["X"], sgrid_roms.xi_u)
    assert_identical(sgrid_roms.cf["Y"], sgrid_roms.eta_u)

    with pytest.raises(KeyError):
        sgrid_roms.u.cf["X"]
    with pytest.raises(KeyError):
        sgrid_roms.u.cf["Y"]

    roms_ = sgrid_roms.set_coords("grid")
    assert roms_.u.cf.axes == {"X": ["xi_u"], "Y": ["eta_u"]}
    assert_identical(roms_.u.cf["X"], sgrid_roms.xi_u)
    assert_identical(roms_.u.cf["Y"], sgrid_roms.eta_u)

    for obj in [roms_, roms_.u]:
        assert "xi_u" in obj.cf.__repr__()
        assert "eta_u" in obj.cf.__repr__()

    assert parse_axes(sgrid_delft) == {"X": {"icell", "inode"}, "Y": {"jcell", "jnode"}}
    assert parse_axes(sgrid_delft3) == {
        "X": {"iface", "inode"},
        "Y": {"jface", "jnode"},
        "Z": {"kface", "knode"},
    }
