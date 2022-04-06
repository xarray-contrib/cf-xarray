# History & provenance tracking

`cf_xarray` will eventually provide functions that add a [`cell_methods` attribute](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-methods), and a `history` attribute, that plug in to xarray's attribute tracking functionality.

Progress is blocked by a few pull requests:

1. An [Xarray pull request](https://github.com/pydata/xarray/pull/5668) to pass "context" to a custom `keep_attrs` handler.
1. Two cf_xarray pull requests that leverage the above: [1](https://github.com/xarray-contrib/cf-xarray/pull/253), [2](https://github.com/xarray-contrib/cf-xarray/pull/259)

```{tip}
If this capability is of interest, contributions toward finishing the above Pull Requests are very welcome.
```
