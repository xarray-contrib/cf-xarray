r"""Tests the operation of cf_xarray's ported unit support code.

Reused with modification from MetPy under the terms of the BSD 3-Clause License.
Copyright (c) 2017 MetPy Developers.
"""

import pytest

pytest.importorskip("pint")

from ..units import units


def test_added_degrees_units():
    """Test that our added degrees units are present in the registry."""
    # Test equivalence of abbreviations/aliases to our defined names
    assert str(units("degrees_N").units) == "degrees_north"
    assert str(units("degreesN").units) == "degrees_north"
    assert str(units("degree_north").units) == "degrees_north"
    assert str(units("degree_N").units) == "degrees_north"
    assert str(units("degreeN").units) == "degrees_north"
    assert str(units("degrees_E").units) == "degrees_east"
    assert str(units("degreesE").units) == "degrees_east"
    assert str(units("degree_east").units) == "degrees_east"
    assert str(units("degree_E").units) == "degrees_east"
    assert str(units("degreeE").units) == "degrees_east"

    # Test equivalence of our defined units to base units
    assert units("degrees_north") == units("degrees")
    assert units("degrees_north").to_base_units().units == units.radian
    assert units("degrees_east") == units("degrees")
    assert units("degrees_east").to_base_units().units == units.radian


def test_gpm_unit():
    """Test that the gpm unit does alias to meters."""
    x = 1 * units("gpm")
    assert str(x.units) == "meter"


def test_psu_unit():
    """Test that the psu unit are present in the registry."""
    x = 1 * units("psu")
    assert str(x.units) == "practical_salinity_unit"


def test_percent_units():
    """Test that percent sign units are properly parsed and interpreted."""
    assert str(units("%").units) == "percent"


@pytest.mark.xfail(reason="not supported by pint, yet: hgrecco/pint#1295")
def test_udunits_power_syntax():
    """Test that UDUNITS style powers are properly parsed and interpreted."""
    assert units("m2 s-2").units == units.m ** 2 / units.s ** 2


def test_udunits_power_syntax_parse_units():
    """Test that UDUNITS style powers are properly parsed and interpreted."""
    assert units.parse_units("m2 s-2") == units.m ** 2 / units.s ** 2
