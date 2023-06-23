import numpy as np
import xarray as xr

airds = xr.tutorial.open_dataset("air_temperature").isel(time=slice(4), lon=slice(50))
airds.air.attrs["cell_measures"] = "area: cell_area"
airds.air.attrs["standard_name"] = "air_temperature"
airds.coords["cell_area"] = (
    xr.DataArray(np.cos(airds.lat * np.pi / 180))
    * xr.ones_like(airds.lon)
    * 105e3
    * 110e3
)

ds_no_attrs = airds.copy(deep=False)
for _variable in ds_no_attrs.variables:
    ds_no_attrs[_variable].attrs = {}


ds_with_tuple = airds.copy(deep=False)
ds_with_tuple = airds.rename({"air": (1, 2, 3)})

# POM dataset
pomds = xr.Dataset()
pomds["sigma"] = (
    # fmt: off
    "sigma",
    [-0.983333, -0.95    , -0.916667, -0.883333, -0.85    , -0.816667,
     -0.783333, -0.75    , -0.716667, -0.683333, -0.65    , -0.616667,
     -0.583333, -0.55    , -0.516667, -0.483333, -0.45    , -0.416667,
     -0.383333, -0.35    , -0.316667, -0.283333, -0.25    , -0.216667,
     -0.183333, -0.15    , -0.116667, -0.083333, -0.05    , -0.016667],
    # fmt: on
    {
        "units": "sigma_level",
        "long_name": "Sigma Stretched Vertical Coordinate at Nodes",
        "positive": "down",
        "standard_name": "ocean_sigma_coordinate",
        "formula_terms": "sigma: sigma eta: zeta depth: depth",
    }
)
pomds["depth"] = 175.0
pomds["zeta"] = ("ocean_time", [-0.155356, -0.127435])


popds = xr.Dataset()
popds.coords["TLONG"] = (
    ("nlat", "nlon"),
    np.ones((20, 30)),
    {"units": "degrees_east"},
)
popds.coords["TLAT"] = (
    ("nlat", "nlon"),
    2 * np.ones((20, 30)),
    {"units": "degrees_north"},
)
popds.coords["ULONG"] = (
    ("nlat", "nlon"),
    0.5 * np.ones((20, 30)),
    {"units": "degrees_east"},
)
popds.coords["ULAT"] = (
    ("nlat", "nlon"),
    2.5 * np.ones((20, 30)),
    {"units": "degrees_north"},
)
popds["UVEL"] = (
    ("nlat", "nlon"),
    np.ones((20, 30)) * 15,
    {"coordinates": "ULONG ULAT", "standard_name": "sea_water_x_velocity"},
)
popds["TEMP"] = (
    ("nlat", "nlon"),
    np.ones((20, 30)) * 15,
    {"coordinates": "TLONG TLAT", "standard_name": "sea_water_potential_temperature"},
)
popds["nlon"] = ("nlon", np.arange(popds.sizes["nlon"]), {"axis": "X"})
popds["nlat"] = ("nlat", np.arange(popds.sizes["nlat"]), {"axis": "Y"})

# This dataset has ancillary variables

anc = xr.Dataset()
anc["q"] = (
    ("x", "y"),
    np.random.randn(10, 20),
    dict(
        standard_name="specific_humidity",
        units="g/g",
        ancillary_variables="q_error_limit q_detection_limit",
    ),
)
anc["q_error_limit"] = (
    ("x", "y"),
    np.random.randn(10, 20),
    dict(standard_name="specific_humidity standard_error", units="g/g"),
)
anc["q_detection_limit"] = xr.DataArray(
    1e-3, attrs=dict(standard_name="specific_humidity detection_minimum", units="g/g")
)


multiple = xr.Dataset()
multiple.coords["x1"] = ("x1", range(30), {"axis": "X"})
multiple.coords["y1"] = ("y1", range(20), {"axis": "Y"})
multiple.coords["x2"] = ("x2", range(10), {"axis": "X"})
multiple.coords["y2"] = ("y2", range(5), {"axis": "Y"})

multiple["v1"] = (("x1", "y1"), np.ones((30, 20)) * 15)
multiple["v2"] = (("x2", "y2"), np.ones((10, 5)) * 15)


romsds = xr.Dataset()
romsds["s_rho"] = (
    # fmt: off
    "s_rho",
    [-0.983333, -0.95    , -0.916667, -0.883333, -0.85    , -0.816667,
     -0.783333, -0.75    , -0.716667, -0.683333, -0.65    , -0.616667,
     -0.583333, -0.55    , -0.516667, -0.483333, -0.45    , -0.416667,
     -0.383333, -0.35    , -0.316667, -0.283333, -0.25    , -0.216667,
     -0.183333, -0.15    , -0.116667, -0.083333, -0.05    , -0.016667],
    # fmt: on
    {
        "long_name": "S-coordinate at RHO-points",
        "valid_min": -1.0,
        "valid_max": 0.0,
        "standard_name": "ocean_s_coordinate_g2",
        "formula_terms": "s: s_rho C: Cs_r eta: zeta depth: h depth_c: hc",
        "field": "s_rho, scalar",
    }
)
romsds.coords["hc"] = 20.0
romsds.coords["h"] = 603.9
romsds.coords["Vtransform"] = 2.0
romsds.coords["Cs_r"] = (
    # fmt: off
    "s_rho",
    [-9.33010396e-01, -8.09234736e-01, -6.98779853e-01, -6.01008926e-01,
     -5.15058562e-01, -4.39938913e-01, -3.74609181e-01, -3.18031817e-01,
     -2.69209327e-01, -2.27207488e-01, -1.91168387e-01, -1.60316097e-01,
     -1.33957253e-01, -1.11478268e-01, -9.23404709e-02, -7.60741092e-02,
     -6.22718662e-02, -5.05823390e-02, -4.07037635e-02, -3.23781605e-02,
     -2.53860004e-02, -1.95414261e-02, -1.46880431e-02, -1.06952600e-02,
     -7.45515186e-03, -4.87981407e-03, -2.89916971e-03, -1.45919898e-03,
     -5.20560097e-04, -5.75774004e-05],
    # fmt: on
)
romsds["zeta"] = ("ocean_time", [-0.155356, -0.127435])
romsds["temp"] = (
    ("ocean_time", "s_rho"),
    [np.linspace(20, 30, 30)] * 2,
    {"coordinates": "z_rho_dummy", "standard_name": "sea_water_potential_temperature"},
)
romsds["temp"].encoding["coordinates"] = "s_rho"
romsds.coords["z_rho_dummy"] = (
    ("ocean_time", "s_rho"),
    np.random.randn(2, 30),
    {"positive": "up"},
)


def _create_mollw_ds():
    # Dataset with random data on a grid that is some sort of Mollweide projection
    YY, XX = np.mgrid[:11, :11] * 5 - 25
    YY_bnds, XX_bnds = np.mgrid[:12, :12] * 5 - 27.5

    R = 50
    theta = np.arcsin(YY / (R * np.sqrt(2)))
    lat = np.rad2deg(np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi))
    lon = np.rad2deg(XX * np.pi / (R * 2 * np.sqrt(2) * np.cos(theta)))

    theta_bnds = np.arcsin(YY_bnds / (R * np.sqrt(2)))
    lat_vertices = np.rad2deg(
        np.arcsin((2 * theta_bnds + np.sin(2 * theta_bnds)) / np.pi)
    )
    lon_vertices = np.rad2deg(
        XX_bnds * np.pi / (R * 2 * np.sqrt(2) * np.cos(theta_bnds))
    )

    lon_bounds = np.stack(
        (
            lon_vertices[:-1, :-1],
            lon_vertices[:-1, 1:],
            lon_vertices[1:, 1:],
            lon_vertices[1:, :-1],
        ),
        axis=-1,
    )
    lat_bounds = np.stack(
        (
            lat_vertices[:-1, :-1],
            lat_vertices[:-1, 1:],
            lat_vertices[1:, 1:],
            lat_vertices[1:, :-1],
        ),
        axis=-1,
    )

    mollwds = xr.Dataset(
        coords=dict(
            lon=xr.DataArray(
                lon,
                dims=("y", "x"),
                attrs={"units": "degrees_east", "bounds": "lon_bounds"},
            ),
            lat=xr.DataArray(
                lat,
                dims=("y", "x"),
                attrs={"units": "degrees_north", "bounds": "lat_bounds"},
            ),
        ),
        data_vars=dict(
            lon_bounds=xr.DataArray(
                lon_bounds, dims=("y", "x", "bounds"), attrs={"units": "degrees_east"}
            ),
            lat_bounds=xr.DataArray(
                lat_bounds, dims=("y", "x", "bounds"), attrs={"units": "degrees_north"}
            ),
            lon_vertices=xr.DataArray(lon_vertices, dims=("y_vertices", "x_vertices")),
            lat_vertices=xr.DataArray(lat_vertices, dims=("y_vertices", "x_vertices")),
        ),
    )

    return mollwds


mollwds = _create_mollw_ds()


def _create_inexact_bounds():
    # Dataset that creates rotated pole curvilinear coordinates with CF bounds in
    # counterclockwise order that have precision issues.
    # dataset created using: https://gist.github.com/larsbuntemeyer/105d83c1eb39b1462150d3fabca0b66b
    rlon = np.array([17.935, 18.045, 18.155])
    rlat = np.array([21.615, 21.725, 21.835])

    lon = np.array(
        [
            [64.21746363939087, 64.42305921561967, 64.62774455060337],
            [64.38488380781622, 64.5906340869802, 64.79546745916944],
            [64.55349991538067, 64.75940009330206, 64.96437666717893],
        ]
    )

    lat = np.array(
        [
            [66.63852914622773, 66.57692364510149, 66.51510790697783],
            [66.72632651121215, 66.66454929137457, 66.60256232699675],
            [66.81394503895442, 66.75199541041104, 66.68983654206977],
        ]
    )

    lon_bounds = np.array(
        [
            [
                [64.03109895962405, 64.23707042152387, 64.44213290724275],
                [64.19784426308645, 64.40397614495, 64.60919235424163],
                [64.36578231818473, 64.57206989228206, 64.77743506492257],
            ],
            [
                [64.23707042152385, 64.44213290724275, 64.64629053434521],
                [64.40397614494998, 64.60919235424163, 64.81349710155956],
                [64.57206989228204, 64.77743506492257, 64.98188214131258],
            ],
            [
                [64.40397614494998, 64.60919235424161, 64.81349710155956],
                [64.57206989228204, 64.77743506492257, 64.98188214131258],
                [64.74136272095073, 64.94687197648351, 65.15145647131322],
            ],
            [
                [64.19784426308645, 64.40397614495, 64.60919235424161],
                [64.36578231818473, 64.57206989228206, 64.77743506492257],
                [64.53492430330854, 64.74136272095075, 64.94687197648351],
            ],
        ]
    )

    lat_bounds = np.array(
        [
            [
                [66.62524451217388, 66.56383049714253, 66.50220514181308],
                [66.71321640554079, 66.65163077691872, 66.58983429068188],
                [66.801010821918, 66.73925288286632, 66.67728458111449],
            ],
            [
                [66.56383049714253, 66.50220514181308, 66.44037016643905],
                [66.65163077691872, 66.58983429068188, 66.52782867442114],
                [66.73925288286632, 66.67728458111449, 66.61510765153199],
            ],
            [
                [66.6516307769187, 66.58983429068185, 66.52782867442114],
                [66.73925288286632, 66.67728458111449, 66.61510765153199],
                [66.82669478850752, 66.76455398820585, 66.702205074604],
            ],
            [
                [66.71321640554079, 66.6516307769187, 66.58983429068185],
                [66.801010821918, 66.73925288286632, 66.67728458111449],
                [66.88862573342278, 66.82669478850752, 66.76455398820585],
            ],
        ]
    )

    rotated = xr.Dataset(
        coords=dict(
            rlon=xr.DataArray(
                rlon,
                dims="rlon",
                attrs={
                    "units": "degrees",
                    "axis": "X",
                    "standard_name": "grid_longitude",
                },
            ),
            rlat=xr.DataArray(
                rlat,
                dims="rlat",
                attrs={
                    "units": "degrees",
                    "axis": "Y",
                    "standard_name": "grid_latitude",
                },
            ),
            lon=xr.DataArray(
                lon,
                dims=("rlon", "rlat"),
                attrs={
                    "units": "degrees_east",
                    "bounds": "lon_bounds",
                    "standard_name": "longitude",
                },
            ),
            lat=xr.DataArray(
                lat,
                dims=("rlon", "rlat"),
                attrs={
                    "units": "degrees_north",
                    "bounds": "lat_bounds",
                    "standard_name": "latitude",
                },
            ),
        ),
        data_vars=dict(
            lon_bounds=xr.DataArray(
                lon_bounds,
                dims=("bounds", "rlon", "rlat"),
                attrs={"units": "degrees_east"},
            ),
            lat_bounds=xr.DataArray(
                lat_bounds,
                dims=("bounds", "rlon", "rlat"),
                attrs={"units": "degrees_north"},
            ),
            rotated_pole=xr.DataArray(
                np.zeros((), dtype=np.int32),
                dims=None,
                attrs={
                    "grid_mapping_name": "rotated_latitude_longitude",
                    "grid_north_pole_latitude": 39.25,
                    "grid_north_pole_longitude": -162.0,
                },
            ),
            temp=xr.DataArray(
                np.random.rand(3, 3),
                dims=("rlat", "rlon"),
                attrs={
                    "standard_name": "air_temperature",
                    "grid_mapping": "rotated_pole",
                },
            ),
        ),
    )

    return rotated


rotds = _create_inexact_bounds()


forecast = xr.decode_cf(
    xr.Dataset.from_dict(
        {
            "coords": {
                "L": {
                    "dims": ("L",),
                    "attrs": {
                        "long_name": "Lead",
                        "standard_name": "forecast_period",
                        "pointwidth": 1.0,
                        "gridtype": 0,
                        "units": "months",
                    },
                    "data": [0, 1],
                },
                "M": {
                    "dims": ("M",),
                    "attrs": {
                        "standard_name": "realization",
                        "long_name": "Ensemble Member",
                        "pointwidth": 1.0,
                        "gridtype": 0,
                        "units": "unitless",
                    },
                    "data": [0, 1, 2],
                },
                "S": {
                    "dims": ("S",),
                    "attrs": {
                        "calendar": "360_day",
                        "long_name": "Forecast Start Time",
                        "standard_name": "forecast_reference_time",
                        "pointwidth": 0,
                        "gridtype": 0,
                        "units": "months since 1960-01-01",
                    },
                    "data": [0, 1, 2, 3],
                },
                "X": {
                    "dims": ("X",),
                    "attrs": {
                        "standard_name": "longitude",
                        "pointwidth": 1.0,
                        "gridtype": 1,
                        "units": "degree_east",
                    },
                    "data": [0, 1, 2, 3, 4],
                },
                "Y": {
                    "dims": ("Y",),
                    "attrs": {
                        "standard_name": "latitude",
                        "pointwidth": 1.0,
                        "gridtype": 0,
                        "units": "degree_north",
                    },
                    "data": [0, 1, 2, 3, 4, 5],
                },
            },
            "attrs": {"Conventions": "IRIDL"},
            "dims": {"L": 2, "M": 3, "S": 4, "X": 5, "Y": 6},
            "data_vars": {
                "sst": {
                    "dims": ("S", "L", "M", "Y", "X"),
                    "attrs": {
                        "pointwidth": 0,
                        "PDS_TimeRange": 3,
                        "center": "US Weather Service - National Met. Center",
                        "grib_name": "TMP",
                        "gribNumBits": 21,
                        "gribcenter": 7,
                        "gribparam": 11,
                        "gribleveltype": 1,
                        "GRIBgridcode": 3,
                        "process": 'Spectral Statistical Interpolation (SSI) analysis from "Final" run.',
                        "PTVersion": 2,
                        "gribfield": 1,
                        "units": "Celsius_scale",
                        "scale_min": -69.97389221191406,
                        "scale_max": 43.039306640625,
                        "long_name": "Sea Surface Temperature",
                        "standard_name": "sea_surface_temperature",
                    },
                    "data": np.arange(np.prod((4, 2, 3, 6, 5))).reshape(
                        (4, 2, 3, 6, 5)
                    ),
                }
            },
        }
    )
)

# Same as flags_excl but easier to read
basin = xr.DataArray(
    [1, 2, 1, 1, 2, 2, 3, 3, 3, 3],
    dims=("time",),
    attrs={
        "flag_values": [1, 2, 3],
        "flag_meanings": "atlantic_ocean pacific_ocean indian_ocean",
        "standard_name": "region",
    },
    name="basin",
)


flag_excl = xr.DataArray(
    np.array([1, 1, 2, 1, 2, 3, 3, 2], np.uint8),
    dims=("time",),
    coords={"time": [0, 1, 2, 3, 4, 5, 6, 7]},
    attrs={
        "flag_values": [1, 2, 3],
        "flag_meanings": "flag_1 flag_2 flag_3",
        "standard_name": "flag_mutual_exclusive",
    },
    name="flag_var",
)


flag_indep = xr.DataArray(
    np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=np.uint8),
    dims=("time",),
    attrs={
        "flag_masks": [1, 2, 4],
        "flag_meanings": "flag_1 flag_2 flag_4",
        "standard_name": "flag_independent",
    },
    name="flag_var",
)


flag_mix = xr.DataArray(
    np.array([4, 8, 13, 5, 10, 14, 7, 3], np.uint8),
    dims=("time",),
    attrs={
        "flag_values": [1, 2, 4, 8, 12],
        "flag_masks": [1, 2, 12, 12, 12],
        "flag_meanings": "flag_1 flag_2 flag_3 flag_4 flag_5",
        "standard_name": "flag_mix",
    },
    name="flag_var",
)


ambig = xr.Dataset(
    data_vars={},
    coords={
        "lat": ("lat", np.zeros(5)),
        "lon": ("lon", np.zeros(5)),
        "vertices_latitude": (["lat", "bnds"], np.zeros((5, 2))),
        "vertices_longitude": (["lon", "bnds"], np.zeros((5, 2))),
    },
)
ambig["lat"].attrs = {
    "bounds": "vertices_latitude",
    "units": "degrees_north",
    "standard_name": "latitude",
    "axis": "Y",
}
ambig["lon"].attrs = {
    "bounds": "vertices_longitude",
    "units": "degrees_east",
    "standard_name": "longitude",
    "axis": "X",
}
ambig["vertices_latitude"].attrs = {
    "units": "degrees_north",
}
ambig["vertices_longitude"].attrs = {
    "units": "degrees_east",
}


vert = xr.Dataset.from_dict(
    {
        "coords": {
            "lat": {
                "dims": ("lat",),
                "attrs": {
                    "standard_name": "latitude",
                    "axis": "Y",
                    "bounds": "lat_bnds",
                    "units": "degrees_north",
                },
                "data": [0.0, 1.0],
            },
            "lon": {
                "dims": ("lon",),
                "attrs": {
                    "standard_name": "longitude",
                    "axis": "X",
                    "bounds": "lon_bnds",
                    "units": "degrees_east",
                },
                "data": [0.0, 1.0],
            },
            "lev": {
                "dims": ("lev",),
                "attrs": {
                    "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
                    "formula": "p = ap + b*ps",
                    "formula_terms": "ap: ap b: b ps: ps",
                    "postitive": "down",
                    "axis": "Z",
                    "bounds": "lev_bnds",
                },
                "data": [0.0, 1.0],
            },
            "time": {
                "dims": ("time",),
                "attrs": {
                    "standard_name": "time",
                    "axis:": "T",
                    "bounds": "time_bnds",
                    "units": "days since 1850-01-01",
                    "calendar": "proleptic_gregorian",
                },
                "data": [0.5],
            },
            "lat_bnds": {
                "dims": (
                    "lat",
                    "bnds",
                ),
                "attrs": {
                    "units": "degrees_north",
                },
                "data": [[0.0, 0.5], [0.5, 1.0]],
            },
            "lon_bnds": {
                "dims": (
                    "lon",
                    "bnds",
                ),
                "attrs": {
                    "units": "degrees_east",
                },
                "data": [[0.0, 0.5], [0.5, 1.0]],
            },
            "lev_bnds": {
                "dims": (
                    "lev",
                    "bnds",
                ),
                "attrs": {
                    "standard_name": "atmosphere_hybrid_sigma_pressure_coordinate",
                    "formula": "p = ap + b*ps",
                    "formula_terms": "ap: ap b: b ps: ps",
                },
                "data": [[0.0, 0.5], [0.5, 1.0]],
            },
            "time_bnds": {
                "dims": ("time", "bnds"),
                "attrs": {
                    "units": "days since 1850-01-01",
                    "calendar": "proleptic_gregorian",
                },
                "data": [[0.0, 1.0]],
            },
            "ap": {
                "dims": ("lev",),
                "data": [0.0, 0.0],
            },
            "b": {
                "dims": ("lev",),
                "data": [1.0, 0.9],
            },
            "ap_bnds": {
                "dims": (
                    "lev",
                    "bnds",
                ),
                "data": [[0.0, 0.0], [0.0, 0.0]],
            },
            "b_bnds": {
                "dims": (
                    "lev",
                    "bnds",
                ),
                "data": [[1.0, 0.95], [0.95, 0.9]],
            },
        },
        "dims": {"time": 1, "lev": 2, "lat": 2, "lon": 2, "bnds": 2},
        "data_vars": {
            "o3": {
                "dims": ("time", "lev", "lat", "lon"),
                "attrs": {
                    "cell_methods": "area: time: mean",
                    "cell_measures": "area: areacella",
                    "missing_value": 1e20,
                    "_FillValue": 1e20,
                },
                "data": np.ones(8, dtype=np.float32).reshape((1, 2, 2, 2)),
            },
            "areacella": {
                "dims": ("lat", "lon"),
                "attrs": {
                    "standard_name": "cell_area",
                    "cell_methods": "area: sum",
                    "missing_value": 1e20,
                    "_FillValue": 1e20,
                },
                "data": np.ones(4, dtype=np.float32).reshape((2, 2)),
            },
            "ps": {
                "dims": ("time", "lat", "lon"),
                "data": np.ones(4, dtype=np.float32).reshape((1, 2, 2)),
            },
        },
    }
)


dsg = xr.Dataset(
    {"foo": (("trajectory", "profile"), [[1, 2, 3], [1, 2, 3]])},
    coords={
        "profile": ("profile", [0, 1, 2], {"cf_role": "profile_id"}),
        "trajectory": ("trajectory", [0, 1], {"cf_role": "trajectory_id"}),
    },
)


sgrid_roms = xr.Dataset()
sgrid_roms["grid"] = xr.DataArray(
    0,
    attrs=dict(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions="xi_psi eta_psi",
        face_dimensions="xi_rho: xi_psi (padding: both) eta_rho: eta_psi (padding: both)",
        edge1_dimensions="xi_u: xi_psi eta_u: eta_psi (padding: both)",
        edge2_dimensions="xi_v: xi_psi (padding: both) eta_v: eta_psi",
        node_coordinates="lon_psi lat_psi",
        face_coordinates="lon_rho lat_rho",
        edge1_coordinates="lon_u lat_u",
        edge2_coordinates="lon_v lat_v",
        vertical_dimensions="s_rho: s_w (padding: none)",
    ),
)
sgrid_roms["u"] = (("xi_u", "eta_u"), np.ones((2, 2)), {"grid": "grid"})

sgrid_delft = xr.Dataset()
sgrid_delft["grid"] = xr.DataArray(
    0,
    attrs=dict(
        cf_role="grid_topology",
        topology_dimension=2,
        node_dimensions="inode jnode",
        face_dimensions="icell: inode (padding: none) jcell: jnode (padding: none)",
        node_coordinates="node_lon node_lat",
    ),
)


sgrid_delft3 = xr.Dataset()
sgrid_delft3["grid"] = xr.DataArray(
    0,
    attrs=dict(
        cf_role="grid_topology",
        topology_dimension=3,
        node_dimensions="inode jnode knode",
        volume_dimensions="iface: inode (padding: none) jface: jnode (padding: none) kface: knode (padding: none)",
        node_coordinates="node_lon node_lat node_elevation",
    ),
)
