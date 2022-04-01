import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cf_xarray as cfxr

ds1 = xr.Dataset(
    {"landsoilt": ("landpoint", np.random.randn(4))},
    {
        "landpoint": pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("lat", "lon")
        )
    },
)

ds2 = xr.Dataset(
    {"landsoilt": ("landpoint", np.random.randn(4))},
    {
        "landpoint": pd.MultiIndex.from_arrays(
            [["a", "b", "c", "d"], [1, 2, 4, 10]], names=("lat", "lon")
        )
    },
)

ds3 = xr.Dataset(
    {"landsoilt": ("landpoint", np.random.randn(4))},
    {
        "landpoint": pd.MultiIndex.from_arrays(
            [["a", "b", "b", "a"], [1, 2, 1, 2]], names=("lat", "lon")
        )
    },
)


@pytest.mark.parametrize("dataset", [ds1, ds2, ds3])
@pytest.mark.parametrize("idxnames", ["landpoint", ("landpoint",), None])
def test_compression_by_gathering_multi_index_roundtrip(dataset, idxnames):
    encoded = cfxr.encode_compress(dataset, idxnames)
    roundtrip = cfxr.decode_compress(encoded, idxnames)
    xr.testing.assert_identical(roundtrip, dataset)
