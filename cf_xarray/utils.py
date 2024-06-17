import inspect
import os
import warnings
from collections import defaultdict
from collections.abc import Iterable
from typing import Any
from xml.etree import ElementTree

import numpy as np
from xarray import DataArray

try:
    import cftime
except ImportError:
    cftime = None


def _is_duck_dask_array(x):
    """Return True if the input is a dask array."""
    # Code copied and simplified from xarray < 2024.02 (xr.core.pycompat.is_duck_dask_array)
    try:
        from dask.base import is_dask_collection
    except ImportError:
        return False

    return (
        is_dask_collection(x)
        and hasattr(x, "ndim")
        and hasattr(x, "shape")
        and hasattr(x, "dtype")
        and (
            (hasattr(x, "__array_function__") and hasattr(x, "__array_ufunc__"))
            or hasattr(x, "__array_namespace__")
        )
    )


def _contains_cftime_datetimes(array) -> bool:
    """Check if an array contains cftime.datetime objects"""
    # Copied / adapted from xarray.core.common
    if cftime is None:
        return False
    else:
        if array.dtype == np.dtype("O") and array.size > 0:
            sample = array.ravel()[0]
            if _is_duck_dask_array(sample):
                sample = sample.compute()
                if isinstance(sample, np.ndarray):
                    sample = sample.item()
            return isinstance(sample, cftime.datetime)
        else:
            return False


def _is_datetime_like(da: DataArray) -> bool:
    if np.issubdtype(da.dtype, np.datetime64) or np.issubdtype(
        da.dtype, np.timedelta64
    ):
        return True
    # if cftime was not imported, _contains_cftime_datetimes will return False
    if _contains_cftime_datetimes(da.data):
        return True

    return False


def parse_cell_methods_attr(attr: str) -> dict[str, str]:
    """
    Parse cell_methods attributes (format is 'measure: name').

    Parameters
    ----------
    attr : str
        String to parse

    Returns
    -------
    Dictionary mapping measure to name
    """
    strings = [s for scolons in attr.split(":") for s in scolons.split()]
    if len(strings) % 2 != 0:
        raise ValueError(f"attrs['cell_measures'] = {attr!r} is malformed.")

    return dict(zip(strings[slice(0, None, 2)], strings[slice(1, None, 2)]))


def invert_mappings(*mappings):
    """Takes a set of mappings and iterates through, inverting to make a
    new mapping of value: set(keys). Keys are deduplicated to avoid clashes between
    standard_name and coordinate names."""
    merged = defaultdict(set)
    for mapping in mappings:
        for k, v in mapping.items():
            for name in v:
                merged[name] |= {k}
    return merged


def always_iterable(obj: Any, allowed=(tuple, list, set, dict)) -> Iterable:
    return [obj] if not isinstance(obj, allowed) else obj


def parse_cf_standard_name_table(source=None):
    """"""

    if not source:
        import pooch

        source = pooch.retrieve(
            "https://raw.githubusercontent.com/cf-convention/cf-convention.github.io/"
            "master/Data/cf-standard-names/current/src/cf-standard-name-table.xml",
            known_hash=None,
        )
    root = ElementTree.parse(source).getroot()

    # Build dictionaries
    info = {}
    table = {}
    aliases = {}
    for child in root:
        if child.tag == "entry":
            key = child.attrib.get("id")
            table[key] = {}
            for item in ["canonical_units", "grib", "amip", "description"]:
                parsed = child.findall(item)
                attr = item.replace("canonical_", "")
                table[key][attr] = (parsed[0].text or "") if parsed else ""
        elif child.tag == "alias":
            alias = child.attrib.get("id")
            key = child.findall("entry_id")[0].text
            aliases[alias] = key
        else:
            info[child.tag] = child.text

    return info, table, aliases


def _get_version():
    __version__ = "unknown"
    try:
        from ._version import __version__
    except ImportError:
        pass
    return __version__


def find_stack_level(test_mode=False) -> int:
    """Find the first place in the stack that is not inside xarray.

    This is unless the code emanates from a test, in which case we would prefer
    to see the xarray source.

    This function is taken from pandas.

    Parameters
    ----------
    test_mode : bool
        Flag used for testing purposes to switch off the detection of test
        directories in the stack trace.

    Returns
    -------
    stacklevel : int
        First level in the stack that is not part of xarray.
    """
    import cf_xarray as cfxr

    pkg_dir = os.path.dirname(cfxr.__file__)
    test_dir = os.path.join(pkg_dir, "tests")

    # https://stackoverflow.com/questions/17407119/python-inspect-stack-is-slow
    frame = inspect.currentframe()
    n = 0
    while frame:
        fname = inspect.getfile(frame)
        if fname.startswith(pkg_dir) and (not fname.startswith(test_dir) or test_mode):
            frame = frame.f_back
            n += 1
        else:
            break
    return n


def emit_user_level_warning(message, category=None):
    """Emit a warning at the user level by inspecting the stack trace."""
    stacklevel = find_stack_level()
    warnings.warn(message, category=category, stacklevel=stacklevel)
