import matplotlib as mpl
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from xarray import Dataset
from xarray.testing import assert_allclose, assert_identical

import cf_xarray  # noqa

from . import raise_if_dask_computes
from .datasets import airds, anc, ds_no_attrs, multiple, popds

mpl.use("Agg")

ds = airds
datasets = [airds, airds.chunk({"lat": 5})]
dataarrays = [airds.air, airds.air.chunk({"lat": 5})]
objects = datasets + dataarrays


def assert_dicts_identical(dict1, dict2):
    assert dict1.keys() == dict2.keys()
    for k in dict1:
        assert_identical(dict1[k], dict2[k])


def test_describe(capsys):
    airds.cf.describe()
    actual = capsys.readouterr().out
    expected = (
        "Axes:\n\tX: ['lon']\n\tY: ['lat']\n\tZ: []\n\tT: ['time']\n"
        "\nCoordinates:\n\tlongitude: ['lon']\n\tlatitude: ['lat']"
        "\n\tvertical: []\n\ttime: ['time']\n"
        "\nCell Measures:\n\tarea: ['cell_area']\n\tvolume: []\n"
        "\nStandard Names:\n\tair_temperature: ['air']\n\n"
    )
    assert actual == expected


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


def test_cell_measures(capsys):
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

    ds.cf.describe()
    actual = capsys.readouterr().out
    expected = (
        "\nCell Measures:\n\tarea: ['cell_area']\n\tfoo_measure: ['foo']\n\tvolume: ['foo']\n"
        "\nStandard Names:\n\tair_temperature: ['air']\n\tfoo_std_name: ['foo']\n\n"
    )
    assert actual.endswith(expected)


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

    ds = airds.copy(deep=True)
    ds["air2"] = ds.air
    with pytest.raises(ValueError):
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

    with pytest.raises(KeyError):
        popds.cf.rename_like(airds)

    renamed = popds.cf["TEMP"].cf.rename_like(airds)
    for k in ["TLONG", "TLAT"]:
        assert k not in renamed.coords
        assert k in original.coords
        assert original.TEMP.attrs["coordinates"] == "TLONG TLAT"

    assert "lon" in renamed.coords
    assert "lat" in renamed.coords
    assert renamed.attrs["coordinates"] == "lon lat"


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
        # groupby("time.day")?
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

    f, ax = plt.subplots(1, 1)
    ds.air.isel(time=1).cf.plot(ax=ax, x="X", y="Y")

    for lim in [ax.get_xlim(), ax.get_ylim()]:
        assert lim[0] > lim[1]


@pytest.mark.parametrize("dims", ["lat", "time", ["lat", "lon"]])
@pytest.mark.parametrize("obj", [airds])
def test_add_bounds(obj, dims):
    expected = dict()
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


def test_docstring():
    assert "One of ('X'" in airds.cf.groupby.__doc__
    assert "One or more of ('X'" in airds.cf.mean.__doc__


def test_guess_coord_axis():
    ds = xr.Dataset()
    ds["time"] = ("time", pd.date_range("2001-01-01", "2001-04-01"))
    ds["lon_rho"] = ("lon_rho", [1, 2, 3, 4, 5])
    ds["lat_rho"] = ("lat_rho", [1, 2, 3, 4, 5])
    ds["x1"] = ("x1", [1, 2, 3, 4, 5])
    ds["y1"] = ("y1", [1, 2, 3, 4, 5])

    dsnew = ds.cf.guess_coord_axis()
    assert dsnew.time.attrs == {"standard_name": "time", "axis": "T"}
    assert dsnew.lon_rho.attrs == {
        "standard_name": "longitude",
        "units": "degrees_east",
    }
    assert dsnew.lat_rho.attrs == {
        "standard_name": "latitude",
        "units": "degrees_north",
    }
    assert dsnew.x1.attrs == {"axis": "X"}
    assert dsnew.y1.attrs == {"axis": "Y"}


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
    from .datasets import romsds

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
    from .datasets import romsds

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


@pytest.mark.parametrize("obj", objects)
@pytest.mark.parametrize("attr", ["drop", "drop_vars", "set_coords"])
@pytest.mark.filterwarnings("ignore:dropping .* using `drop` .* deprecated")
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
