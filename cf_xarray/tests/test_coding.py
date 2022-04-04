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
@pytest.mark.parametrize("idxnames", ["foo", "landpoint", ("landpoint",), None])
def test_compression_by_gathering_multi_index_roundtrip(mindex, idxnames):
    dim = "foo" if idxnames == "foo" else "landpoint"
    dataset = xr.Dataset(
        data_vars={"landsoilt": (dim, np.random.randn(4), {"foo": "bar"})},
        coords={
            dim: (dim, mindex, {"long_name": "land point number"}),
            "coord1": (dim, [1, 2, 3, 4], {"foo": "baz"}),
        },
        attrs={"dataset": "test dataset"},
    )
    dataset.lat.attrs["standard_name"] = "latitude"
    dataset.lon.attrs["standard_name"] = "longitude"

    encoded = cfxr.encode_multi_index_as_compress(dataset, idxnames)
    roundtrip = cfxr.decode_compress_to_multi_index(encoded, idxnames)
    assert "compress" not in roundtrip[dim].encoding
    xr.testing.assert_identical(roundtrip, dataset)

    dataset[dim].attrs["compress"] = "lat lon"
    with pytest.raises(ValueError):
        cfxr.encode_multi_index_as_compress(dataset, idxnames)
