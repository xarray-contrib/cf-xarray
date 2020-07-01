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
    {"axis": "X", "units": "degrees_east"},
)
popds.coords["TLAT"] = (
    ("nlat", "nlon"),
    2 * np.ones((20, 30)),
    {"axis": "Y", "units": "degrees_north"},
)
popds.coords["ULONG"] = (
    ("nlat", "nlon"),
    0.5 * np.ones((20, 30)),
    {"axis": "X", "units": "degrees_east"},
)
popds.coords["ULAT"] = (
    ("nlat", "nlon"),
    2.5 * np.ones((20, 30)),
    {"axis": "Y", "units": "degrees_north"},
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
    1e-3, attrs=dict(standard_name="specific_humidity detection_minimum", units="g/g"),
)
anc


multiple = xr.Dataset()
multiple.coords["x1"] = ("x1", range(30), {"axis": "X"})
multiple.coords["y1"] = ("y1", range(20), {"axis": "Y"})
multiple.coords["x2"] = ("x2", range(10), {"axis": "X"})
multiple.coords["y2"] = ("y2", range(5), {"axis": "Y"})

multiple["v1"] = (("x1", "y1"), np.ones((30, 20)) * 15)
multiple["v2"] = (("x2", "y2"), np.ones((10, 5)) * 15)
