import numpy as np
from xarray.testing import assert_identical

from cf_xarray.datasets import flag_excl
from cf_xarray.groupers import FlagGrouper


def test_flag_grouper():
    ds = flag_excl.to_dataset().set_coords("flag_var")
    ds["foo"] = ("time", np.arange(8))
    actual = ds.groupby(flag_var=FlagGrouper()).mean()
    expected = ds.groupby("flag_var").mean()
    expected["flag_var"] = ["flag_1", "flag_2", "flag_3"]
    expected["flag_var"].attrs["standard_name"] = "flag_mutual_exclusive"
    assert_identical(actual, expected)
