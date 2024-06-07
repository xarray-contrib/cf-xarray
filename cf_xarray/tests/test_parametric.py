import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_allclose

from cf_xarray import parametric

ps = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"), name="ps")

p0 = xr.DataArray(
    [
        10,
    ],
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

depth_c = xr.DataArray([30.0], name="depth_c")

s = xr.DataArray([0, 1, 2], dims=("lev"), name="s")


def test_atmosphere_ln_pressure_coordinate():
    lev = xr.DataArray(
        [0, 1, 2],
        dims=("lev",),
        name="lev",
    )

    output = parametric.atmosphere_ln_pressure_coordinate(p0, lev)

    expected = xr.DataArray([10.0, 3.678794, 1.353353], dims=("lev",), name="p")

    assert_allclose(output, expected)

    assert output.name == "p"
    assert output.attrs["standard_name"] == "air_pressure"


def test_atmosphere_sigma_coordinate():
    ptop = xr.DataArray([0.98692327], name="ptop")

    output = parametric.atmosphere_sigma_coordinate(sigma, ps, ptop)

    expected = xr.DataArray(
        [
            [
                [[0.986923, 0.986923], [0.986923, 0.986923]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.013077, 1.013077], [1.013077, 1.013077]],
            ],
            [
                [[0.986923, 0.986923], [0.986923, 0.986923]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[1.013077, 1.013077], [1.013077, 1.013077]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        coords={"k": (("lev",), [0, 1, 2])},
        name="p",
    )

    assert_allclose(output, expected)

    assert output.name == "p"
    assert output.attrs["standard_name"] == "air_pressure"


def test_atmosphere_hybrid_sigma_pressure_coordinate():
    ap = xr.DataArray([3, 4, 5], dims=("lev",), name="ap")

    output = parametric.atmosphere_hybrid_sigma_pressure_coordinate(b, ps, p0, a=a)

    expected = xr.DataArray(
        [
            [
                [[6.0, 6.0], [6.0, 6.0]],
                [[17.0, 17.0], [17.0, 17.0]],
                [[28.0, 28.0], [28.0, 28.0]],
            ],
            [
                [[6.0, 6.0], [6.0, 6.0]],
                [[17.0, 17.0], [17.0, 17.0]],
                [[28.0, 28.0], [28.0, 28.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="p",
    )

    assert_allclose(output, expected)

    assert output.name == "p"
    assert output.attrs["standard_name"] == "air_pressure"

    output = parametric.atmosphere_hybrid_sigma_pressure_coordinate(b, ps, p0, ap=ap)

    expected = xr.DataArray(
        [
            [
                [[9.0, 9.0], [9.0, 9.0]],
                [[11.0, 11.0], [11.0, 11.0]],
                [[13.0, 13.0], [13.0, 13.0]],
            ],
            [
                [[9.0, 9.0], [9.0, 9.0]],
                [[11.0, 11.0], [11.0, 11.0]],
                [[13.0, 13.0], [13.0, 13.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="p",
    )

    assert_allclose(output, expected)

    assert output.name == "p"
    assert output.attrs["standard_name"] == "air_pressure"


def test_atmosphere_hybrid_height_coordinate():
    orog = xr.DataArray(
        np.zeros((2, 2, 2)),
        dims=("time", "lat", "lon"),
        attrs={"standard_name": "surface_altitude"},
    )

    output = parametric.atmosphere_hybrid_height_coordinate(a, b, orog)

    expected = xr.DataArray(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[2.0, 2.0], [2.0, 2.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="p",
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_atmosphere_sleve_coordinate():
    b1 = xr.DataArray([0, 0, 1], dims=("lev",), name="b1")

    b2 = xr.DataArray([1, 1, 0], dims=("lev",), name="b2")

    ztop = xr.DataArray(
        [30.0],
        name="ztop",
        attrs={"standard_name": "altitude_at_top_of_atmosphere_model"},
    )

    zsurf1 = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"))

    zsurf2 = xr.DataArray(np.ones((2, 2, 2)), dims=("time", "lat", "lon"))

    output = parametric.atmosphere_sleve_coordinate(a, b1, b2, ztop, zsurf1, zsurf2)

    expected = xr.DataArray(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[31.0, 31.0], [31.0, 31.0]],
                [[61.0, 61.0], [61.0, 61.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[31.0, 31.0], [31.0, 31.0]],
                [[61.0, 61.0], [61.0, 61.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_ocean_sigma_coordinate():
    output = parametric.ocean_sigma_coordinate(sigma, eta, depth)

    expected = xr.DataArray(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="z",
        coords={"k": (("lev",), [0, 1, 2])},
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_ocean_s_coordinate():
    _a = xr.DataArray([1], name="a")

    _b = xr.DataArray([1], name="b")

    output = parametric.ocean_s_coordinate(s, eta, depth, _a, _b, depth_c)

    expected = xr.DataArray(
        [
            [
                [[12.403492, 12.403492], [12.403492, 12.403492]],
                [[40.434874, 40.434874], [40.434874, 40.434874]],
                [[70.888995, 70.888995], [70.888995, 70.888995]],
            ],
            [
                [[12.403492, 12.403492], [12.403492, 12.403492]],
                [[40.434874, 40.434874], [40.434874, 40.434874]],
                [[70.888995, 70.888995], [70.888995, 70.888995]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_ocean_s_coordinate_g1():
    C = xr.DataArray([0, 1, 2], dims=("lev",), name="C")

    output = parametric.ocean_s_coordinate_g1(s, C, eta, depth, depth_c)

    expected = xr.DataArray(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_ocean_s_coordinate_g2():
    C = xr.DataArray([0, 1, 2], dims=("lev",), name="C")

    output = parametric.ocean_s_coordinate_g2(s, C, eta, depth, depth_c)

    expected = xr.DataArray(
        [
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
            [
                [[1.0, 1.0], [1.0, 1.0]],
                [[3.0, 3.0], [3.0, 3.0]],
                [[5.0, 5.0], [5.0, 5.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_ocean_sigma_z_coordinate():
    zlev = xr.DataArray([0, 1, np.nan], dims=("lev",), name="zlev", attrs={"standard_name": "altitude"})

    _sigma = xr.DataArray([np.nan, np.nan, 3], dims=("lev",), name="sigma")

    output = parametric.ocean_sigma_z_coordinate(_sigma, eta, depth, depth_c, 10, zlev)

    expected = xr.DataArray(
        [
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[7.0, 7.0], [7.0, 7.0]],
            ],
            [
                [[0.0, 0.0], [0.0, 0.0]],
                [[1.0, 1.0], [1.0, 1.0]],
                [[7.0, 7.0], [7.0, 7.0]],
            ],
        ],
        dims=("time", "lev", "lat", "lon"),
        name="z",
    )

    assert_allclose(output, expected)

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


def test_ocean_double_sigma_coordinate():
    k_c = xr.DataArray(
        [
            1,
        ],
        name="k_c",
    )

    href = xr.DataArray(
        [
            20.0,
        ],
        name="href",
    )

    z1 = xr.DataArray(
        [
            10.0,
        ],
        name="z1",
    )

    z2 = xr.DataArray(
        [
            30.0,
        ],
        name="z2",
    )

    a = xr.DataArray(
        [
            2.0,
        ],
        name="a",
    )

    output = parametric.ocean_double_sigma_coordinate(
        sigma, depth, z1, z2, a, href, k_c
    )

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

    assert output.name == "z"
    assert output.attrs["standard_name"] == "altitude"


@pytest.mark.parametrize("input,expected", [
    ({"zlev": {"standard_name": "altitude"}}, "altitude"),
    ({"zlev": {"standard_name": "altitude"}, "eta": {"standard_name": "sea_surface_height_above_geoid"}}, "altitude"),
    ({"zlev": {"standard_name": "altitude"}, "eta": {"standard_name": "sea_surface_height_above_geoid"}, "depth": {"standard_name": "sea_floor_depth_below_geoid"}}, "altitude"),
    ({"eta": {"standard_name": "sea_surface_height_above_geoid"}, "depth": {"standard_name": "sea_floor_depth_below_geoid"}}, "altitude"),
])
def test_derive_ocean_stdname(input, expected):
    output = parametric._derive_ocean_stdname(**input)

    assert output == expected


def test_derive_ocean_stdname_no_values():
    with pytest.raises(ValueError, match="Must provide atleast one of depth, eta, zlev."):
        parametric._derive_ocean_stdname()


def test_derive_ocean_stdname_empty_value():
    with pytest.raises(ValueError, match="The values for zlev cannot be `None`."):
        parametric._derive_ocean_stdname(zlev=None)


def test_derive_ocean_stdname_no_standard_name():
    with pytest.raises(ValueError, match="The standard name for the 'zlev' variable is not available."):
        parametric._derive_ocean_stdname(zlev={})


def test_derive_ocean_stdname_no_match():
    with pytest.raises(ValueError, match="Could not derive standard name from combination of not in any list."):
        parametric._derive_ocean_stdname(zlev={"standard_name": "not in any list"})


def test_func_from_stdname():
    with pytest.raises(AttributeError):
        parametric.func_from_stdname("test")

    func = parametric.func_from_stdname("atmosphere_ln_pressure_coordinate")

    assert func == parametric.atmosphere_ln_pressure_coordinate


def test_check_requirements():
    with pytest.raises(KeyError, match="'Required terms lev, p0 absent in dataset.'"):
        parametric.check_requirements(parametric.atmosphere_ln_pressure_coordinate, [])

    parametric.check_requirements(
        parametric.atmosphere_ln_pressure_coordinate, ["p0", "lev"]
    )

    with pytest.raises(
        KeyError,
        match=r"'Required terms b, p0 and atleast one optional term a, ap absent in dataset.'",
    ):
        parametric.check_requirements(
            parametric.atmosphere_hybrid_sigma_pressure_coordinate, ["ps"]
        )

    with pytest.raises(
        KeyError,
        match="'Atleast one of the optional terms a, ap is absent in dataset.'",
    ):
        parametric.check_requirements(
            parametric.atmosphere_hybrid_sigma_pressure_coordinate, ["ps", "p0", "b"]
        )

    with pytest.raises(
        KeyError,
        match="'Required terms b and atleast one optional term a, ap absent in dataset.'",
    ):
        parametric.check_requirements(
            parametric.atmosphere_hybrid_sigma_pressure_coordinate, ["ps", "p0", "a"]
        )
