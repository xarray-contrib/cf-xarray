---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  display_name: Python 3
  name: python3
---
```{eval-rst}
.. currentmodule:: xarray
```
```{code-cell}
---
tags: [remove-cell]
---
import cf_xarray as cfxr
import numpy as np
import xarray as xr
xr.set_options(display_expand_data=False)
```


# Encoding and decoding

`cf_xarray` aims to support encoding and decoding variables using CF conventions not yet implemented by Xarray. For now, ``cf_xarray`` provides
:py:func:`encode_compress` and :py:func:`decode_compress` to encode MultiIndex-ed dimensions using the
["compression by gathering"](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering) convention.

First encode
```{code-cell}
ds = xr.Dataset(
    {"landsoilt": ("landpoint", np.random.randn(4))},
    {
        "landpoint": pd.MultiIndex.from_product(
            [["a", "b"], [1, 2]], names=("lat", "lon")
        )
    },
)
encoded = cfxr.encode_compress(ds, "landpoint")
encoded
```

At this point, we can write `encoded` to a CF-compliant dataset using :py:func:`xarray.to_netcdf` for example.
After reading that file, decode using
```{code-cell}
cfxr.decode_compress(encoded, "landpoint")
```
