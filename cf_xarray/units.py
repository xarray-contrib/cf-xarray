"""Module to provide unit support via pint approximating UDUNITS/CF."""
import functools
import re
import warnings

import pint
from pint import (  # noqa: F401
    DimensionalityError,
    UndefinedUnitError,
    UnitStrippedWarning,
)


# from `xclim`'s unit support module with permission of the maintainers
@pint.register_unit_format("cf")
def short_formatter(unit, registry, **options):
    """Return a CF-compliant unit string from a `pint` unit.

    Parameters
    ----------
    unit : pint.UnitContainer
        Input unit.
    registry : pint.UnitRegistry
        the associated registry
    **options
        Additional options (may be ignored)

    Returns
    -------
    out : str
        Units following CF-Convention, using symbols.
    """
    import re

    # convert UnitContainer back to Unit
    unit = registry.Unit(unit)
    # Print units using abbreviations (millimeter -> mm)
    s = f"{unit:~D}"

    # Search and replace patterns
    pat = r"(?P<inverse>(?:1 )?/ )?(?P<unit>\w+)(?: \*\* (?P<pow>\d))?"

    def repl(m):
        i, u, p = m.groups()
        p = p or (1 if i else "")
        neg = "-" if i else ""

        return f"{u}{neg}{p}"

    out, n = re.subn(pat, repl, s)

    # Remove multiplications
    out = out.replace(" * ", " ")
    # Delta degrees:
    out = out.replace("Δ°", "delta_deg")
    return out.replace("percent", "%")


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

units.define(
    pint.unit.UnitDefinition("percent", "%", (), pint.converters.ScaleConverter(0.01))
)

# Define commonly encountered units (both CF and non-CF) not defined by pint
units.define(
    "degrees_north = degree = degrees_N = degreesN = degree_north = degree_N = degreeN"
)
units.define(
    "degrees_east = degree = degrees_E = degreesE = degree_east = degree_E = degreeE"
)
units.define("@alias meter = gpm")
units.define("practical_salinity_unit = [] = psu = PSU")

# Enable pint's built-in matplotlib support
try:
    units.setup_matplotlib()
except ImportError:
    warnings.warn(
        "Import(s) unavailable to set up matplotlib support...skipping this portion "
        "of the setup."
    )
# end of vendored code from MetPy

# Set as application registry
pint.set_application_registry(units)
