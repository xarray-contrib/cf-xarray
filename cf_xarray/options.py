"""
Started from xarray options.py
"""

from typing import Any, MutableMapping

OPTIONS: MutableMapping[str, Any] = {
    "custom_criteria": [],
}


class set_options:
    """Set options for cf-xarray in a controlled context.
    Currently supported options:
    - ``custom_critera``: Translate from axis, coord, or custom name to
      variable name optionally  using ``custom_criteria``. Default: [].

    You can use ``set_options`` either as a context manager:
    >>> ds = xr.Dataset({"x": np.arange(1000)})
    >>> with xr.set_options(display_width=40):
    ...     print(ds)
    ...
    <xarray.Dataset>
    Dimensions:  (x: 1000)
    Coordinates:
      * x        (x) int64 0 1 2 ... 998 999
    Data variables:
        *empty*
    Or to set global options:
    >>> xr.set_options(display_width=80)  # doctest: +ELLIPSIS
    <xarray.core.options.set_options object at 0x...>
    """

    def __init__(self, **kwargs):
        self.old = {}
        for k, v in kwargs.items():
            if k not in OPTIONS:
                raise ValueError(
                    f"argument name {k!r} is not in the set of valid options {set(OPTIONS)!r}"
                )
            self.old[k] = OPTIONS[k]
        self._apply_update(kwargs)

    def _apply_update(self, options_dict):
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
