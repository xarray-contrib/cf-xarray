"""Module to provide unit support via pint approximating UDUNITS/CF."""

import functools
import re

import pint
from packaging.version import Version

from .utils import emit_user_level_warning


@pint.register_unit_format("cf")
def short_formatter(unit, registry, **options):
    """Return a CF-compliant unit string from a `pint` unit.

    Parameters
    ----------
    unit : pint.UnitContainer
        Input unit.
    registry : pint.UnitRegistry
        The associated registry
    **options
        Additional options (may be ignored)

    Returns
    -------
    out : str
        Units following CF-Convention, using symbols.
    """
    # pint 0.24.1 gives {"dimensionless": 1} for non-shortened dimensionless units
    # CF uses "1" to denote fractions and dimensionless quantities
    if unit == {"dimensionless": 1} or not unit:
        return "1"

    # If u is a name, get its symbol (same as pint's "~" pre-formatter)
    # otherwise, assume a symbol (pint should have already raised on invalid units before this)
    unit = pint.util.UnitsContainer(
        {
            registry._get_symbol(u) if u in registry._units else u: exp
            for u, exp in unit.items()
        }
    )

    # Change in formatter signature in pint 0.24
    if Version(pint.__version__) < Version("0.24"):
        args = (unit.items(),)
    else:
        # Numerators splitted from denominators
        args = (
            ((u, e) for u, e in unit.items() if e >= 0),
            ((u, e) for u, e in unit.items() if e < 0),
        )

    out = pint.formatter(*args, as_ratio=False, product_fmt=" ", power_fmt="{}{}")
    # To avoid potentiel unicode problems in netCDF. In both cases, this unit is not recognized by udunits
    return out.replace("Δ°", "delta_deg")


# ------
# Reused with modification from MetPy under the terms of the BSD 3-Clause License.
# Copyright (c) 2015,2017,2019 MetPy Developers.
# Create registry, with preprocessors for UDUNITS-style powers (m2 s-2) and percent signs
units = pint.UnitRegistry(
    autoconvert_offset_to_baseunit=True,
    preprocessors=[
        functools.partial(
            re.compile(
                r"(?<=[A-Za-z])(?![A-Za-z])(?<![0-9\-][eE])(?<![0-9\-])(?=[0-9\-])"
            ).sub,
            "**",
        ),
        lambda string: string.replace("%", "percent"),
    ],
    force_ndarray_like=True,
)
# ----- end block copied from metpy

# need to insert to make sure this is the first preprocessor
# This ensures we convert integer `1` to string `"1"`, as needed by pint.
units.preprocessors.insert(0, str)

# -----
units.define("percent = 0.01 = %")

# Define commonly encountered units (both CF and non-CF) not defined by pint
units.define("@alias meter = gpm")
# ----- end block copied from metpy

# -----
# The following redefinitions were copied from xclim under the terms of their Apache-2 license
# In pint, the default symbol for year is "a" which is not CF-compliant (stands for "are")
units.define("year = 365.25 * day = yr")

# Define commonly encountered units not defined by pint
units.define("@alias degC = C = deg_C = Celsius = degrees_Celsius")
units.define("@alias degK = deg_K")
units.define("@alias day = d")
units.define("@alias hour = h")  # Not the Planck constant...
units.define(
    "degrees_north = degree = degrees_north = degrees_N = degreesN = degree_north = degree_N = degreeN"
)
units.define(
    "degrees_east = degree = degrees_east = degrees_E = degreesE = degree_east = degree_E = degreeE"
)
# degrees for grid_longitude / grid_latitude for grid_mappings
units.define("degrees = degree = degrees")
units.define("[speed] = [length] / [time]")
# ----- end block copied from xclim

# Add other specific aliases (by cf_xarray developers)
units.define("practical_salinity_unit = [] = psu = PSU")

# Enable pint's built-in matplotlib support
try:
    units.setup_matplotlib()
except ImportError:
    emit_user_level_warning(
        "Import(s) unavailable to set up matplotlib support...skipping this portion "
        "of the setup.",
        UserWarning,
    )
# end of vendored code from MetPy

# Set as application registry
pint.set_application_registry(units)
