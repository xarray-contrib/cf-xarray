"""
Criteria for identifying axes and coordinate variables.
Reused with modification from MetPy under the terms of the BSD 3-Clause License.
Copyright (c) 2017 MetPy Developers.
"""


import re
from typing import Mapping, MutableMapping, Tuple

cf_role_criteria: Mapping[str, Mapping[str, str]] = {
    k: {"cf_role": k} for k in ("timeseries_id", "profile_id", "trajectory_id")
}

coordinate_criteria: MutableMapping[str, MutableMapping[str, Tuple]] = {
    "latitude": {
        "standard_name": ("latitude",),
        "units": (
            "degree_north",
            "degree_N",
            "degreeN",
            "degrees_north",
            "degrees_N",
            "degreesN",
        ),
        "_CoordinateAxisType": ("Lat",),
    },
    "longitude": {
        "standard_name": ("longitude",),
        "units": (
            "degree_east",
            "degree_E",
            "degreeE",
            "degrees_east",
            "degrees_E",
            "degreesE",
        ),
        "_CoordinateAxisType": ("Lon",),
    },
    "Z": {
        "standard_name": (
            "model_level_number",
            "atmosphere_ln_pressure_coordinate",
            "atmosphere_sigma_coordinate",
            "atmosphere_hybrid_sigma_pressure_coordinate",
            "atmosphere_hybrid_height_coordinate",
            "atmosphere_sleve_coordinate",
            "ocean_sigma_coordinate",
            "ocean_s_coordinate",
            "ocean_s_coordinate_g1",
            "ocean_s_coordinate_g2",
            "ocean_sigma_z_coordinate",
            "ocean_double_sigma_coordinate",
        ),
        "_CoordinateAxisType": (
            "GeoZ",
            "Height",
            "Pressure",
        ),
        "axis": ("Z",),
        "cartesian_axis": ("Z",),
        "grads_dim": ("z",),
    },
    "vertical": {
        "standard_name": (
            "air_pressure",
            "height",
            "depth",
            "geopotential_height",
            # computed dimensional coordinate name
            "altitude",
            "height_above_geopotential_datum",
            "height_above_reference_ellipsoid",
            "height_above_mean_sea_level",
        ),
        "positive": ("up", "down"),
    },
    "X": {
        "standard_name": ("projection_x_coordinate",),
        "_CoordinateAxisType": ("GeoX",),
        "axis": ("X",),
        "cartesian_axis": ("X",),
        "grads_dim": ("x",),
    },
    "Y": {
        "standard_name": ("projection_y_coordinate",),
        "_CoordinateAxisType": ("GeoY",),
        "axis": ("Y",),
        "cartesian_axis": ("Y",),
        "grads_dim": ("y",),
    },
    "T": {
        "standard_name": ("time",),
        "_CoordinateAxisType": ("Time",),
        "axis": ("T",),
        "cartesian_axis": ("T",),
        "grads_dim": ("t",),
    },
}

coordinate_criteria["time"] = coordinate_criteria["T"]

# "long_name" and "standard_name" criteria are the same. For convenience.
for coord, attrs in coordinate_criteria.items():
    coordinate_criteria[coord]["long_name"] = coordinate_criteria[coord][
        "standard_name"
    ]
coordinate_criteria["X"]["long_name"] += ("cell index along first dimension",)
coordinate_criteria["Y"]["long_name"] += ("cell index along second dimension",)


#: regular expressions for guess_coord_axis
regex = {
    "time": re.compile("\\bt\\b|(time|min|hour|day|week|month|year)[0-9]*"),
    "Z": re.compile(
        "(z|nav_lev|gdep|lv_|[o]*lev|bottom_top|sigma|h(ei)?ght|altitude|depth|"
        "isobaric|pres|isotherm)[a-z_]*[0-9]*"
    ),
    "Y": re.compile("y|j|nlat|nj"),
    "latitude": re.compile("y?(nav_lat|lat|gphi)[a-z0-9]*"),
    "X": re.compile("x|i|nlon|ni"),
    "longitude": re.compile("x?(nav_lon|lon|glam)[a-z0-9]*"),
}
regex["T"] = regex["time"]
