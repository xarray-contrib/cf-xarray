from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Union
from urllib.parse import urlsplit
from urllib.request import urlopen
from xml.etree import ElementTree

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


def always_iterable(obj: Any) -> Iterable:
    return [obj] if not isinstance(obj, (tuple, list, set, dict)) else obj


def parse_cf_table(uri: Union[str, Path], verbose=False):
    """
    Parse cf standard names table in xml format.

    Parameters
    ----------
    cf_table_uri: str, Path
        Location of the cf standard names table in xml format.
    verbose: bool
        Print table info to screen

    Returns
    -------
    tuple
        Dictionaries mapping:
            1. standard_name to attributes
            2. alias to standard_name
    """

    # Deal with urls
    if isinstance(uri, str) and all((urlsplit(uri).scheme, urlsplit(uri).netloc)):
        uri = urlopen(uri)

    tree = ElementTree.parse(uri)
    root = tree.getroot()

    # Construct table
    info = []
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
            info.append(f"{child.tag}: {child.text}")

    if verbose:
        print("\n  * ".join(["CF standard name table info:"] + info))

    return (
        table,
        aliases,
    )
