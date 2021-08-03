# This module provides functions for adding CF attribtues
# and tracking history, provenance using xarray's keep_attrs
# functionality

import copy
import functools
from datetime import datetime

CELL_METHODS = {
    "sum": "sum",
    "max": "maximum",
    "min": "minimum",
    "median": "median",
    "mean": "mean",
    "std": "standard_deviation",
    "var": "variance",
}


def add_cell_methods(attrs, context):
    """Add appropriate cell_methods attribute."""
    assert len(attrs) == 1
    cell_methods = attrs[0].get("cell_methods", "")
    # TODO: get dim_name from context
    return {"cell_methods": f"dim_name: {CELL_METHODS[context.func]} {cell_methods}"}


def add_history(attrs, context):
    """Adds a history attribute following the NetCDF User Guide convention."""

    # https://www.unidata.ucar.edu/software/netcdf/documentation/4.7.4-pre/attribute_conventions.html
    # A global attribute for an audit trail. This is a character array with a line
    # for each invocation of a program that has modified the dataset. Well-behaved
    # generic netCDF applications should append a line containing:
    #     date, time of day, user name, program name and command arguments.

    # nco uses the ctime format
    now = datetime.now().ctime()
    history = attrs[0].get("history", [])
    new_history = (
        f"{now}:"
        f" {context.func}(args)\n"
        # TODO: should we record software versions?
    )
    return {"history": history + [new_history]}


def _tracker(
    attrs,
    context,
    strict: bool = False,
    cell_methods: bool = True,
    history: bool = True,
):

    # can only handle single variable attrs for now
    assert len(attrs) == 1
    attrs_out = copy.deepcopy(attrs[0])

    if cell_methods and context.func in CELL_METHODS:
        attrs_out.update(add_cell_methods(attrs, context))
    if history:
        attrs_out.update(add_history(attrs, context))
        pass
    return attrs_out


def track_cf_attributes(
    *, strict: bool = False, cell_methods: bool = True, history: bool = True
):
    """Top-level user-facing function.

    Parameters
    ----------
    strict: bool
        Controls if an error is raised when an appropriate attribute cannot
        be added because of lack of information.
    cell_methods: bool
        Add cell_methods attribute when possible
    history: bool
        Adds a history attribute like NCO and follows the NUG convention.
    """

    # TODO: check xarray version here.
    return functools.partial(
        _tracker, strict=strict, cell_methods=cell_methods, history=history
    )
