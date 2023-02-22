r"""Tests the operation of cf_xarray's ported unit support code.

Reused with modification from MetPy under the terms of the BSD 3-Clause License.
Copyright (c) 2017 MetPy Developers.
"""

import pytest

pytest.importorskip("pint")

from ..units import units as ureg


def test_added_degrees_units():
    """Test that our added degrees units are present in the registry."""
    # Test equivalence of abbreviations/aliases to our defined names
    assert str(ureg("degrees_N").units) == "degrees_north"
    assert str(ureg("degreesN").units) == "degrees_north"
    assert str(ureg("degree_north").units) == "degrees_north"
    assert str(ureg("degree_N").units) == "degrees_north"
    assert str(ureg("degreeN").units) == "degrees_north"
    assert str(ureg("degrees_E").units) == "degrees_east"
    assert str(ureg("degreesE").units) == "degrees_east"
    assert str(ureg("degree_east").units) == "degrees_east"
    assert str(ureg("degree_E").units) == "degrees_east"
    assert str(ureg("degreeE").units) == "degrees_east"

    # Test equivalence of our defined units to base units
    assert ureg("degrees_north") == ureg("degrees")
    assert ureg("degrees_north").to_base_units().units == ureg.radian
    assert ureg("degrees_east") == ureg("degrees")
    assert ureg("degrees_east").to_base_units().units == ureg.radian

    assert ureg("degrees").to_base_units().units == ureg.radian


def test_gpm_unit():
    """Test that the gpm unit does alias to meters."""
    x = 1 * ureg("gpm")
    assert str(x.units) == "meter"


def test_psu_unit():
    """Test that the psu unit are present in the registry."""
    x = 1 * ureg("psu")
    assert str(x.units) == "practical_salinity_unit"


def test_percent_units():
    """Test that percent sign units are properly parsed and interpreted."""
    assert str(ureg("%").units) == "percent"


@pytest.mark.xfail(reason="not supported by pint, yet: hgrecco/pint#1295")
def test_udunits_power_syntax():
    """Test that UDUNITS style powers are properly parsed and interpreted."""
    assert ureg("m2 s-2").units == ureg.m**2 / ureg.s**2


def test_udunits_power_syntax_parse_units():
    """Test that UDUNITS style powers are properly parsed and interpreted."""
    assert ureg.parse_units("m2 s-2") == ureg.m**2 / ureg.s**2


@pytest.mark.parametrize(
    ["units", "expected"],
    (
        ("kg ** 2", "kg2"),
        ("m ** -1", "m-1"),
        ("m ** 2 / s ** 2", "m2 s-2"),
        ("m ** 3 / (kg * s ** 2)", "m3 kg-1 s-2"),
    ),
)
def test_udunits_format(units, expected):
    u = ureg.parse_units(units)

    assert f"{u:cf}" == expected


@pytest.mark.parametrize(
    "alias",
    [ureg("Celsius"), ureg("degC"), ureg("C"), ureg("deg_C"), ureg("degrees_Celsius")],
)
def test_temperature_aliases(alias):
    assert alias == ureg("celsius")
