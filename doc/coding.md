---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---

```{eval-rst}
.. currentmodule:: cf_xarray
```

```{code-cell}
---
tags: [remove-cell]
---
import cf_xarray as cfxr
import numpy as np
import pandas as pd
import xarray as xr
xr.set_options(display_expand_data=False)
```

# Encoding and decoding

`cf_xarray` aims to support encoding and decoding variables using CF conventions not yet implemented by Xarray.

## Compression by gathering

The ["compression by gathering"](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering)
convention could be used for either {py:class}`pandas.MultiIndex` objects or `pydata/sparse` arrays.

### MultiIndex

`cf_xarray` provides {py:func}`encode_multi_index_as_compress` and {py:func}`decode_compress_to_multi_index` to encode MultiIndex-ed
dimensions using "compression by gethering".

Here's a test dataset

```{code-cell}
ds = xr.Dataset(
    {"landsoilt": ("landpoint", np.random.randn(4), {"foo": "bar"})},
    {
        "landpoint": pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("lat", "lon")
        )
    },
)
ds
```

First encode (note the `"compress"` attribute on the `landpoint` variable)

```{code-cell}
encoded = cfxr.encode_multi_index_as_compress(ds, "landpoint")
encoded
```

At this point, we can write `encoded` to a CF-compliant dataset using {py:func}`xarray.Dataset.to_netcdf` for example.
After reading that file, decode using

```{code-cell}
decoded = cfxr.decode_compress_to_multi_index(encoded, "landpoint")
decoded
```

We roundtrip perfectly

```{code-cell}
ds.identical(decoded)
```

### Sparse arrays

This is unsupported currently but a pull request is welcome!
