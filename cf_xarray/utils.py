from collections import defaultdict
from typing import Dict

from xarray import DataArray


def _is_datetime_like(da: DataArray) -> bool:
    import numpy as np

    if np.issubdtype(da.dtype, np.datetime64) or np.issubdtype(
        da.dtype, np.timedelta64
    ):
        return True

    try:
        import cftime

        if isinstance(da.data[0], cftime.datetime):
            return True
    except ImportError:
        pass

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
