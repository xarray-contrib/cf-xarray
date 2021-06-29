"""
Tests OPTIONS logic brought in from xarray.
"""

import pytest

import cf_xarray as cfxr


def test_options():

    # test for inputting a nonexistent option
    with pytest.raises(ValueError):
        cfxr.set_options(DISPLAY_WIDTH=80)
