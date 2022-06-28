"""
Started from xarray options.py
"""

import copy
from typing import Any, MutableMapping

from .utils import always_iterable

OPTIONS: MutableMapping[str, Any] = {
    "custom_criteria": [],
    "warn_on_missing_variables": True,
}


class set_options:
    """Set options for cf-xarray in a controlled context.

    Parameters
    ----------
    custom_criteria : dict
        Translate from axis, coord, or custom name to
        variable name optionally using ``custom_criteria``. Default: [].
    warn_on_missing_variables : bool
        Whether to raise a warning when variables referred to in attributes
        are not present in the object.

    Examples
    --------

    You can use ``set_options`` either as a context manager:

    >>> import numpy as np
    >>> import xarray as xr
    >>> my_custom_criteria = {"ssh": {"name": "elev$"}}
    >>> ds = xr.Dataset({"elev": np.arange(1000)})
    >>> with cf_xarray.set_options(custom_criteria=my_custom_criteria):
    ...     xr.testing.assert_identical(ds["elev"], ds.cf["ssh"])
    ...

    Or to set global options:

    >>> cf_xarray.set_options(custom_criteria=my_custom_criteria)
    >>> xr.testing.assert_identical(ds["elev"], ds.cf["ssh"])
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
        options_dict = copy.deepcopy(options_dict)
        for k, v in options_dict.items():
            if k == "custom_criteria":
                options_dict["custom_criteria"] = always_iterable(
                    options_dict["custom_criteria"], allowed=(tuple, list)
                )
        OPTIONS.update(options_dict)

    def __enter__(self):
        return

    def __exit__(self, type, value, traceback):
        self._apply_update(self.old)
