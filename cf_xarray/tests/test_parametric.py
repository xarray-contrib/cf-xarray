import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from cf_xarray import parametric

ps = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"), name="ps")

p0 = xr.DataArray(
    10.0,
    dims=(),
    coords={},
    name="p0",
)

a = xr.DataArray([0, 1, 2], dims=("lev",), name="a")

b = xr.DataArray([6, 7, 8], dims=("lev",), name="b")

sigma = xr.DataArray(
    [0, 1, 2], dims=("lev",), name="sigma", coords={"k": (("lev",), [0, 1, 2])}
)

eta = xr.DataArray(
    np.ones((2, 2, 2)),
    dims=("time", "lat", "lon"),
    name="eta",
    attrs={"standard_name": "sea_surface_height_above_geoid"},
)

depth = xr.DataArray(
    np.ones((2, 2)),
    dims=("lat", "lon"),
    name="depth",
    attrs={"standard_name": "sea_floor_depth_below_geoid"},
)

depth_c = xr.DataArray(30.0, dims=(), coords={}, name="depth_c")

s = xr.DataArray([0, 1, 2], dims=("lev"), name="s")


def test_atmosphere_ln_pressure_coordinate():
    lev = xr.DataArray(
        [0, 1, 2],
        dims=("lev",),
        name="lev",
    )

    transform = parametric.AtmosphereLnPressure.from_terms(
        {
            "p0": p0,
            "lev": lev,
        }
    )

    output = transform.decode()

    expected = xr.DataArray([10.0, 3.678794, 1.353353], dims=("lev",), name="p")

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "air_pressure"


def test_atmosphere_sigma_coordinate():
    ptop = xr.DataArray(0.98692327, dims=(), coords={}, name="ptop")

    transform = parametric.AtmosphereSigma.from_terms(
        {
            "sigma": sigma,
            "ps": ps,
            "ptop": ptop,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [[0.986923, 0.986923], [0.986923, 0.986923]],
                [[0.986923, 0.986923], [0.986923, 0.986923]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            [
                [[1.013077, 1.013077], [1.013077, 1.013077]],
                [[1.013077, 1.013077], [1.013077, 1.013077]],
            ],
        ],
        dims=("lev", "time", "lat", "lon"),
        coords={"k": (("lev",), [0, 1, 2])},
        name="p",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "air_pressure"


def test_atmosphere_hybrid_sigma_pressure_coordinate():
    ap = xr.DataArray([3, 4, 5], dims=("lev",), name="ap")

    transform = parametric.AtmosphereHybridSigmaPressure.from_terms(
        {
            "b": b,
            "ps": ps,
            "a": a,
            "p0": p0,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [[6.0, 6.0], [6.0, 6.0]],
                [[6.0, 6.0], [6.0, 6.0]],
            ],
            [
                [[17.0, 17.0], [17.0, 17.0]],
                [[17.0, 17.0], [17.0, 17.0]],
            ],
            [
                [[28.0, 28.0], [28.0, 28.0]],
                [[28.0, 28.0], [28.0, 28.0]],
            ],
        ],
        dims=("lev", "time", "lat", "lon"),
        name="p",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "air_pressure"

    transform = parametric.AtmosphereHybridSigmaPressure.from_terms(
        {
            "b": b,
            "ps": ps,
            "ap": ap,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [[9.0, 9.0], [9.0, 9.0]],
                [[9.0, 9.0], [9.0, 9.0]],
            ],
            [
                [[11.0, 11.0], [11.0, 11.0]],
                [[11.0, 11.0], [11.0, 11.0]],
            ],
            [
                [[13.0, 13.0], [13.0, 13.0]],
                [[13.0, 13.0], [13.0, 13.0]],
            ],
        ],
        dims=("lev", "time", "lat", "lon"),
        name="p",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "air_pressure"


def test_atmosphere_hybrid_height_coordinate():
    orog = xr.DataArray(
        np.zeros((2, 2, 2)),
        dims=("time", "lat", "lon"),
        attrs={"standard_name": "surface_altitude"},
    )

    transform = parametric.AtmosphereHybridHeight.from_terms(
        {
            "a": a,
            "b": b,
            "orog": orog,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            [
                [[2.0, 2.0], [2.0, 2.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
        ],
        dims=("lev", "time", "lat", "lon"),
        name="p",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_atmosphere_sleve_coordinate():
    b1 = xr.DataArray([0, 0, 1], dims=("lev",), name="b1")

    b2 = xr.DataArray([1, 1, 0], dims=("lev",), name="b2")

    ztop = xr.DataArray(
        30.0,
        dims=(),
        coords={},
        name="ztop",
        attrs={"standard_name": "altitude_at_top_of_atmosphere_model"},
    )

    zsurf1 = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"))

    zsurf2 = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"))

    transform = parametric.AtmosphereSleve.from_terms(
        {
            "a": a,
            "b1": b1,
            "b2": b2,
            "ztop": ztop,
            "zsurf1": zsurf1,
            "zsurf2": zsurf2,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            [
                [[31.0, 31.0], [31.0, 31.0]],
                [[31.0, 31.0], [31.0, 31.0]],
            ],
            [
                [[61.0, 61.0], [61.0, 61.0]],
                [[61.0, 61.0], [61.0, 61.0]],
            ],
        ],
        dims=("lev", "time", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_ocean_sigma_coordinate():
    transform = parametric.OceanSigma.from_terms(
        {
            "sigma": sigma,
            "eta": eta,
            "depth": depth,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
            ],
            [
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
            ],
        ],
        dims=("time", "lat", "lon", "lev"),
        name="z",
        coords={"k": (("lev",), [0, 1, 2])},
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_ocean_s_coordinate():
    _a = xr.DataArray(1, dims=(), coords={}, name="a")

    _b = xr.DataArray(1, dims=(), coords={}, name="b")

    transform = parametric.OceanS.from_terms(
        {
            "s": s,
            "eta": eta,
            "depth": depth,
            "a": _a,
            "b": _b,
            "depth_c": depth_c,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [
                    [12.403492, 40.434874, 70.888995],
                    [12.403492, 40.434874, 70.888995],
                ],
                [
                    [12.403492, 40.434874, 70.888995],
                    [12.403492, 40.434874, 70.888995],
                ],
            ],
            [
                [
                    [12.403492, 40.434874, 70.888995],
                    [12.403492, 40.434874, 70.888995],
                ],
                [
                    [12.403492, 40.434874, 70.888995],
                    [12.403492, 40.434874, 70.888995],
                ],
            ],
        ],
        dims=("time", "lat", "lon", "lev"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_ocean_s_coordinate_g1():
    C = xr.DataArray([0, 1, 2], dims=("lev",), name="C")

    transform = parametric.OceanSG2.from_terms(
        {
            "s": s,
            "c": C,
            "eta": eta,
            "depth": depth,
            "depth_c": depth_c,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
            ],
            [
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
            ],
        ],
        dims=("time", "lat", "lon", "lev"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_ocean_s_coordinate_g2():
    C = xr.DataArray([0, 1, 2], dims=("lev",), name="C")

    transform = parametric.OceanSG2.from_terms(
        {
            "s": s,
            "c": C,
            "eta": eta,
            "depth": depth,
            "depth_c": depth_c,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
            ],
            [
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
                [
                    [1.0, 3.0, 5.0],
                    [1.0, 3.0, 5.0],
                ],
            ],
        ],
        dims=("time", "lat", "lon", "lev"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_ocean_sigma_z_coordinate():
    zlev = xr.DataArray(
        [0, 1, np.nan], dims=("lev",), name="zlev", attrs={"standard_name": "altitude"}
    )

    _sigma = xr.DataArray([np.nan, np.nan, 3], dims=("lev",), name="sigma")

    transform = parametric.OceanSigmaZ.from_terms(
        {
            "sigma": _sigma,
            "eta": eta,
            "depth": depth,
            "depth_c": depth_c,
            "nsigma": 10,
            "zlev": zlev,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.0, 1.0], [1.0, 1.0]],
            ],
            [
                [[7.0, 7.0], [7.0, 7.0]],
                [[7.0, 7.0], [7.0, 7.0]],
            ],
        ],
        dims=("lev", "time", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


def test_ocean_double_sigma_coordinate():
    k_c = xr.DataArray(
        1,
        dims=(),
        coords={},
        name="k_c",
    )

    href = xr.DataArray(
        20.0,
        dims=(),
        coords={},
        name="href",
    )

    z1 = xr.DataArray(
        10.0,
        dims=(),
        coords={},
        name="z1",
    )

    z2 = xr.DataArray(
        30.0,
        dims=(),
        coords={},
        name="z2",
    )

    a = xr.DataArray(
        2.0,
        dims=(),
        coords={},
        name="a",
    )

    transform = parametric.OceanDoubleSigma.from_terms(
        {
            "sigma": sigma,
            "depth": depth,
            "z1": z1,
            "z2": z2,
            "a": a,
            "href": href,
            "k_c": k_c,
        }
    )

    output = transform.decode()

    expected = xr.DataArray(
        [
            [[0.0, 0.0], [0.0, 0.0]],
            [[10.010004, 10.010004], [10.010004, 10.010004]],
            [[1.0, 1.0], [1.0, 1.0]],
        ],
        dims=("lev", "lat", "lon"),
        coords={"k": (("lev",), [0, 1, 2])},
        name="z",
    )

    assert_allclose(output, expected)

    assert output.attrs["standard_name"] == "altitude"


@pytest.mark.parametrize(
    "input,expected",
    [
        ({"zlev": {"standard_name": "altitude"}}, "altitude"),
        (
            {
                "zlev": {"standard_name": "altitude"},
                "eta": {"standard_name": "sea_surface_height_above_geoid"},
            },
            "altitude",
        ),
        (
            {
                "zlev": {"standard_name": "altitude"},
                "eta": {"standard_name": "sea_surface_height_above_geoid"},
                "depth": {"standard_name": "sea_floor_depth_below_geoid"},
            },
            "altitude",
        ),
        (
            {
                "eta": {"standard_name": "sea_surface_height_above_geoid"},
                "depth": {"standard_name": "sea_floor_depth_below_geoid"},
            },
            "altitude",
        ),
    ],
)
def test_derive_ocean_stdname(input, expected):
    output = parametric._derive_ocean_stdname(**input)

    assert output == expected


def test_derive_ocean_stdname_no_standard_name():
    with pytest.raises(
        ValueError, match="The standard name for the 'zlev' variable is not available."
    ):
        parametric._derive_ocean_stdname(zlev={})


def test_derive_ocean_stdname_no_match():
    with pytest.raises(
        ValueError,
        match="Could not derive standard name from combination of not in any list.",
    ):
        parametric._derive_ocean_stdname(zlev={"standard_name": "not in any list"})
