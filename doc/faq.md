# Frequently Asked Questions

## I find `.cf` repr hard to read!

Install [rich](https://rich.readthedocs.io) and load the Jupyter extension for easier-to-read reprs.

```python
%load_ext rich

import cf_xarray
import xarray as xr

ds = xr.tutorial.open_dataset("air_temperature")
ds.cf
```

![rich repr](_static/rich-repr-example.png)
