r"""Module to provide unit support via pint approximating UDUNITS/CF.

Reused with modification from MetPy under the terms of the BSD 3-Clause License.
Copyright (c) 2015,2017,2019 MetPy Developers.
"""
import re

import pint

UndefinedUnitError = pint.UndefinedUnitError
DimensionalityError = pint.DimensionalityError
UnitStrippedWarning = pint.UnitStrippedWarning

# Create registry, with preprocessors for UDUNITS-style powers (m2 s-2) and percent signs
units = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    preprocessors=[
        functools.partial(
            re.sub,
            r"(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])",
            "**",
        ),
        lambda string: string.replace("%", "percent"),
    ],
)

units.define(
    pint.unit.UnitDefinition("percent", "%", (), pint.converters.ScaleConverter(0.01))
)

# Define commonly encoutered units (both CF and non-CF) not defined by pint
units.define(
    "degrees_north = degree = degrees_N = degreesN = degree_north = degree_N = degreeN"
)
units.define(
    "degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE"
)
units.define("@alias meter = gpm")
units.define("practical_salinity_units = [] = psu")

# Enable pint's built-in matplotlib support
units.setup_matplotlib()

del pint
