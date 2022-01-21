import os
from collections import defaultdict
from typing import Any, Dict, Iterable
from xml.etree import ElementTree

import numpy as np
from xarray import DataArray

try:
    import cftime
except ImportError:
    cftime = None


def _contains_cftime_datetimes(array) -> bool:
    """Check if an array contains cftime.datetime objects"""
    # Copied / adapted from xarray.core.common
    from xarray.core.pycompat import is_duck_dask_array

    if cftime is None:
        return False
    else:
        if array.dtype == np.dtype("O") and array.size > 0:
            sample = array.ravel()[0]
            if is_duck_dask_array(sample):
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


def parse_cell_methods_attr(attr: str) -> Dict[str, str]:
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
        source = os.path.join(
            os.path.dirname(__file__), "data", "cf-standard-name-table.xml"
        )
    root = ElementTree.parse(source).getroot()

    # Build dictionaries
    info = {}
    table: dict = {}
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
