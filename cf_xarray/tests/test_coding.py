import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cf_xarray as cfxr


@pytest.mark.parametrize(
    "mindex",
    [
        pd.MultiIndex.from_product([["a", "b"], [1, 2]], names=("lat", "lon")),
        pd.MultiIndex.from_arrays(
            [["a", "b", "c", "d"], [1, 2, 4, 10]], names=("lat", "lon")
        ),
        pd.MultiIndex.from_arrays(
            [["a", "b", "b", "a"], [1, 2, 1, 2]], names=("lat", "lon")
        ),
    ],
)
@pytest.mark.parametrize("idxnames", ["landpoint", ("landpoint",), None])
def test_compression_by_gathering_multi_index_roundtrip(mindex, idxnames):
    dataset = xr.Dataset(
        {"landsoilt": ("landpoint", np.random.randn(4), {"foo": "bar"})},
        {"landpoint": mindex},
    )
    encoded = cfxr.encode_compress(dataset, idxnames)
    roundtrip = cfxr.decode_compress(encoded, idxnames)
    assert "compress" in roundtrip["landpoint"].encoding
    xr.testing.assert_identical(roundtrip, dataset)
