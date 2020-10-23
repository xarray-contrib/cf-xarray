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

ds_no_attrs = airds.copy(deep=True)
for variable in ds_no_attrs.variables:
    ds_no_attrs[variable].attrs = {}


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
anc


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
    {"coordinates": "z_rho_dummy"},
)
romsds["temp"].encoding["coordinates"] = "s_rho"
romsds.coords["z_rho_dummy"] = (
    ("ocean_time", "s_rho"),
    np.random.randn(2, 30),
    {"positive": "up"},
)


# Dataset with random data on a grid that is some sort of Mollweide projection
XX, YY = np.mgrid[:11, :11] * 5 - 25
XX_bnds, YY_bnds = np.mgrid[:12, :12] * 5 - 27.5

R = 50
theta = np.arcsin(YY / (R * np.sqrt(2)))
lat = np.rad2deg(np.arcsin((2 * theta + np.sin(2 * theta)) / np.pi))
lon = np.rad2deg(XX * np.pi / (R * 2 * np.sqrt(2) * np.cos(theta)))

theta_bnds = np.arcsin(YY_bnds / (R * np.sqrt(2)))
lat_corners = np.rad2deg(np.arcsin((2 * theta_bnds + np.sin(2 * theta_bnds)) / np.pi))
lon_corners = np.rad2deg(XX_bnds * np.pi / (R * 2 * np.sqrt(2) * np.cos(theta_bnds)))

lon_bounds = np.stack(
    (
        lon_corners[:-1, :-1],
        lon_corners[:-1, 1:],
        lon_corners[1:, 1:],
        lon_corners[1:, :-1],
    ),
    axis=0,
)
lat_bounds = np.stack(
    (
        lat_corners[:-1, :-1],
        lat_corners[:-1, 1:],
        lat_corners[1:, 1:],
        lat_corners[1:, :-1],
    ),
    axis=0,
)

mollwds = xr.Dataset(
    coords=dict(
        lon=xr.DataArray(
            lon,
            dims=("x", "y"),
            attrs={"units": "degrees_east", "bounds": "lon_bounds"},
        ),
        lat=xr.DataArray(
            lat,
            dims=("x", "y"),
            attrs={"units": "degrees_north", "bounds": "lat_bounds"},
        ),
    ),
    data_vars=dict(
        lon_bounds=xr.DataArray(
            lon_bounds, dims=("bounds", "x", "y"), attrs={"units": "degrees_east"}
        ),
        lat_bounds=xr.DataArray(
            lat_bounds, dims=("bounds", "x", "y"), attrs={"units": "degrees_north"}
        ),
        lon_corners=xr.DataArray(lon_corners, dims=("x_corners", "y_corners")),
        lat_corners=xr.DataArray(lat_corners, dims=("x_corners", "y_corners")),
    ),
)
