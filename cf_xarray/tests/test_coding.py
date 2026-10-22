import numpy as np
import pandas as pd
import pytest
import xarray as xr

import cf_xarray as cfxr


@pytest.mark.parametrize(
    "name", ["lat name", " leading", "trailing ", "tab\tname", "line\nname", ""]
)
@pytest.mark.parametrize("level", [0, 1])
def test_compression_rejects_ambiguous_level_names(name, level):
    names = ["lat", "lon"]
    names[level] = name
    index = pd.MultiIndex.from_product([[1, 2], [3, 4]], names=names)
    dataset = xr.Dataset(
        {"data": ("point", [1, 2, 3, 4])},
        coords=xr.Coordinates.from_pandas_multiindex(index, "point"),
    )
    before = dataset.copy(deep=True)
    with pytest.raises(ValueError, match="nonempty strings without whitespace"):
        cfxr.encode_multi_index_as_compress(dataset)
    xr.testing.assert_identical(dataset, before)


@pytest.mark.parametrize("with_data", [False, True])
def test_compression_preserves_unrelated_coordinates(with_data):
    index = pd.MultiIndex.from_product([[1, 2], [3, 4]], names=("x", "y"))
    dataset = xr.Dataset(
        coords=xr.Coordinates.from_pandas_multiindex(index, "point"),
        attrs={"title": "coordinate preservation"},
    ).assign_coords(
        auxiliary=("point", [5, 6, 7, 8], {"long_name": "auxiliary coordinate"}),
        independent=("other", [9, 10], {"units": "m"}),
        scalar=xr.DataArray(42, attrs={"long_name": "scalar coordinate"}),
    )
    dataset.independent.encoding["dtype"] = "int32"
    if with_data:
        dataset["data"] = ("point", [1, 2, 3, 4])

    encoded = cfxr.encode_multi_index_as_compress(dataset)
    before = encoded.copy(deep=True)
    decoded = cfxr.decode_compress_to_multi_index(encoded)
    xr.testing.assert_identical(decoded, dataset)
    assert decoded.independent.encoding == dataset.independent.encoding
    xr.testing.assert_identical(encoded, before)


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
