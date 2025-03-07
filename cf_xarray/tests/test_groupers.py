import numpy as np
import pytest
import xarray as xr
from xarray.testing import assert_identical

pytest.importorskip("xarray", "2024.07.0")

from cf_xarray.datasets import flag_excl
from cf_xarray.groupers import FlagGrouper


def test_flag_grouper():
    ds = flag_excl.to_dataset().set_coords("flag_var").copy(deep=True)
    ds["foo"] = ("time", np.arange(8))
    actual = ds.groupby(flag_var=FlagGrouper()).mean()
    expected = ds.groupby("flag_var").mean()
    expected["flag_var"] = ["flag_1", "flag_2", "flag_3"]
    expected["flag_var"].attrs["standard_name"] = "flag_mutual_exclusive"
    assert_identical(actual, expected)

    del ds.flag_var.attrs["flag_values"]
    with pytest.raises(ValueError):
        ds.groupby(flag_var=FlagGrouper())

    ds.flag_var.attrs["flag_values"] = [0, 1, 2]
    del ds.flag_var.attrs["flag_meanings"]
    with pytest.raises(ValueError):
        ds.groupby(flag_var=FlagGrouper())


@pytest.mark.parametrize(
    "values",
    [
        [1, 2],
        [1, 2, 3],  # value out of range of flag_values
    ],
)
def test_flag_grouper_optimized(values):
    ds = xr.Dataset(
        {"foo": ("x", values, {"flag_values": [0, 1, 2], "flag_meanings": "a b c"})}
    )
    ret = FlagGrouper().factorize(ds.foo)
    expected = ds.foo
    expected.data[ds.foo.data > 2] = -1
    del ds.foo.attrs["flag_meanings"]
    del ds.foo.attrs["flag_values"]
    assert_identical(ret.codes, ds.foo)
