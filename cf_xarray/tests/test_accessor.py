import matplotlib as mpl
import pytest
import xarray as xr
from matplotlib import pyplot as plt
from xarray.testing import assert_identical

import cf_xarray  # noqa

from . import raise_if_dask_computes
from .datasets import airds, anc, ds_no_attrs, multiple, popds

mpl.use("Agg")

ds = airds
datasets = [airds, airds.chunk({"lat": 5})]
dataarrays = [airds.air, airds.air.chunk({"lat": 5})]
objects = datasets + dataarrays


def test_dicts():
    from .datasets import airds

    actual = airds.cf.sizes
    expected = {"X": 50, "Y": 25, "T": 4, "longitude": 50, "latitude": 25, "time": 4}
    assert actual == expected

    assert popds.cf.sizes == popds.sizes

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

    airds = airds.copy(deep=True)
    airds.lon.attrs = {}
    actual = airds.cf.sizes
    expected = {"lon": 50, "Y": 25, "T": 4, "latitude": 25, "time": 4}
    assert actual == expected


def test_describe():
    actual = airds.cf._describe()
    expected = (
        "Axes:\n\tX: ['lon']\n\tY: ['lat']\n\tZ: [None]\n\tT: ['time']\n"
        "\nCoordinates:\n\tlongitude: ['lon']\n\tlatitude: ['lat']"
        "\n\tvertical: [None]\n\ttime: ['time']\n"
        "\nCell Measures:\n\tarea: unsupported\n\tvolume: unsupported\n"
        "\nStandard Names:\n\t['air_temperature', 'latitude', 'longitude', 'time']"
    )
    assert actual == expected


def test_getitem_standard_name():
    actual = airds.cf["air_temperature"]
    expected = airds["air"]
    assert_identical(actual, expected)

    ds = airds.copy(deep=True)
    ds["air2"] = ds.air
    actual = ds.cf["air_temperature"]
    expected = ds[["air", "air2"]]
    assert_identical(actual, expected)

    with pytest.raises(KeyError):
        ds.air.cf["air_temperature"]


def test_getitem_ancillary_variables():
    expected = anc.set_coords(["q_error_limit", "q_detection_limit"])["q"]
    assert_identical(anc.cf["q"], expected)
    assert_identical(anc.cf["specific_humidity"], expected)


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


def test_kwargs_expand_key_to_multiple_keys():

    actual = multiple.cf.isel(X=5, Y=3)
    expected = multiple.isel(x1=5, y1=3, x2=5, y2=3)
    assert_identical(actual, expected)

    actual = multiple.cf.mean("X")
    expected = multiple.mean(["x1", "x2"])
    assert_identical(actual, expected)

    actual = multiple.cf.coarsen(X=10, Y=5)
    expected = multiple.coarsen(x1=10, y1=5, x2=10, y2=5)
    assert_identical(actual.mean(), expected.mean())


@pytest.mark.parametrize(
    "obj, expected",
    [
        (ds, set(("latitude", "longitude", "time", "X", "Y", "T", "air_temperature"))),
        (ds.air, set(("latitude", "longitude", "time", "X", "Y", "T", "area"))),
        (ds_no_attrs.air, set()),
    ],
)
def test_get_valid_keys(obj, expected):
    actual = obj.cf.get_valid_keys()
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
        air.cf[["longitude", "latitude"]],


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
    ds = popds
    assert_identical(
        ds.cf["X"], ds.reset_coords()[["ULONG", "TLONG"]].set_coords(["ULONG", "TLONG"])
    )
    assert_identical(
        ds.cf["Y"], ds.reset_coords()[["ULAT", "TLAT"]].set_coords(["ULAT", "TLAT"])
    )
    assert_identical(ds.UVEL.cf["X"], ds["ULONG"].reset_coords(drop=True))
    assert_identical(ds.TEMP.cf["X"], ds["TLONG"].reset_coords(drop=True))


def test_plot_xincrease_yincrease():
    ds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4), lon=slice(50))
    ds.lon.attrs["positive"] = "down"
    ds.lat.attrs["positive"] = "down"

    f, ax = plt.subplots(1, 1)
    ds.air.isel(time=1).cf.plot(ax=ax, x="X", y="Y")

    for lim in [ax.get_xlim(), ax.get_ylim()]:
        assert lim[0] > lim[1]
