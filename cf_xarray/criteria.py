"""
Criteria for identifying axes and coordinate variables.
Reused with modification from MetPy under the terms of the BSD 3-Clause License.
Copyright (c) 2017 MetPy Developers.
"""


import copy
import re
from typing import MutableMapping, Tuple

coordinate_criteria: MutableMapping[str, MutableMapping[str, Tuple]] = {
    "standard_name": {
        "X": ("projection_x_coordinate",),
        "Y": ("projection_y_coordinate",),
        "T": ("time",),
        "time": ("time",),
        "vertical": (
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
        "Z": (
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
        "latitude": ("latitude",),
        "longitude": ("longitude",),
    },
    "_CoordinateAxisType": {
        "T": ("Time",),
        "Z": ("GeoZ", "Height", "Pressure"),
        "Y": ("GeoY",),
        "latitude": ("Lat",),
        "X": ("GeoX",),
        "longitude": ("Lon",),
    },
    "axis": {"T": ("T",), "Z": ("Z",), "Y": ("Y",), "X": ("X",)},
    "cartesian_axis": {"T": ("T",), "Z": ("Z",), "Y": ("Y",), "X": ("X",)},
    "positive": {"vertical": ("up", "down")},
    "units": {
        "latitude": (
            "degree_north",
            "degree_N",
            "degreeN",
            "degrees_north",
            "degrees_N",
            "degreesN",
        ),
        "longitude": (
            "degree_east",
            "degree_E",
            "degreeE",
            "degrees_east",
            "degrees_E",
            "degreesE",
        ),
    },
}

# "long_name" and "standard_name" criteria are the same. For convenience.
coordinate_criteria["long_name"] = copy.deepcopy(coordinate_criteria["standard_name"])
coordinate_criteria["long_name"]["X"] += ("cell index along first dimension",)
coordinate_criteria["long_name"]["Y"] += ("cell index along second dimension",)

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
