"""
Tests OPTIONS logic brought in from xarray.
"""

import pytest
import xarray as xr
import cf_xarray as cfxr


def test_options():

    # test for inputting a nonexistent option
    with pytest.raises(ValueError):
        cfxr.set_options(DISPLAY_WIDTH=80)


def test_tracker():
    da = xr.DataArray([1, 2, 3], dims="time", name="my_array", attrs={"comment": "A comment"})
    da = da.chunk({"time": 3})
    with xr.set_options(keep_attrs=cfxr.track_cf_attributes(cell_methods=True, history=True)):
        out = da.mean("time")
    print(out.__dask_graph__())
    assert out.attrs["cell_methods"] == "time: mean"
    assert out.attrs["history"] == "mean(dim='time', skipna=None)"
