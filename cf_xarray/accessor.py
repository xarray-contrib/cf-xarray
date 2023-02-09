from __future__ import annotations

import functools
import inspect
import itertools
import re
import warnings
from collections import ChainMap
from datetime import datetime
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Sequence,
    TypeVar,
    Union,
    cast,
    overload,
)

import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.arithmetic import SupportsArithmetic
from xarray.core.groupby import GroupBy
from xarray.core.resample import Resample
from xarray.core.rolling import Coarsen, Rolling
from xarray.core.weighted import Weighted

from . import sgrid
from .criteria import (
    _DSG_ROLES,
    cf_role_criteria,
    coordinate_criteria,
    grid_mapping_var_criteria,
    regex,
)
from .formatting import (
    _format_coordinates,
    # _format_conventions,
    _format_data_vars,
    _format_dsg_roles,
    _format_flags,
    _format_sgrid,
    _maybe_panel,
)
from .helpers import _guess_bounds_1d, _guess_bounds_2d, bounds_to_vertices
from .options import OPTIONS
from .utils import (
    _get_version,
    _is_datetime_like,
    always_iterable,
    invert_mappings,
    parse_cell_methods_attr,
    parse_cf_standard_name_table,
)

#: Classes wrapped by cf_xarray.
_WRAPPED_CLASSES = (Resample, GroupBy, Rolling, Coarsen, Weighted)

#:  `axis` names understood by cf_xarray
_AXIS_NAMES = ("X", "Y", "Z", "T")

#:  `coordinate` types understood by cf_xarray.
_COORD_NAMES = ("longitude", "latitude", "vertical", "time")

#:  Cell measures understood by cf_xarray.
_CELL_MEASURES = ("area", "volume")

ATTRS = {
    "X": {"axis": "X"},
    "T": {"axis": "T", "standard_name": "time"},
    "Y": {"axis": "Y"},
    "Z": {"axis": "Z"},
    "latitude": {"units": "degrees_north", "standard_name": "latitude"},
    "longitude": {"units": "degrees_east", "standard_name": "longitude"},
}
ATTRS["time"] = ATTRS["T"]
ATTRS["vertical"] = ATTRS["Z"]

# Type for Mapper functions
Mapper = Callable[[Union[DataArray, Dataset], Hashable], List[Hashable]]

# Type for decorators
F = TypeVar("F", bound=Callable[..., Any])


def sort_maybe_hashable(iterable: Iterable[Hashable]) -> list[Hashable]:
    only_str: list[str] = [elem for elem in iterable if isinstance(elem, str)]
    non_str: list[Hashable] = [elem for elem in iterable if not isinstance(elem, str)]
    return sorted(only_str) + non_str


def apply_mapper(
    mappers: Mapper | tuple[Mapper, ...],
    obj: DataArray | Dataset,
    key: Hashable,
    error: bool = True,
    default: Any = None,
) -> list[Any]:
    """
    Applies a mapping function; does error handling / returning defaults.

    Expects the mapper function to raise an error if passed a bad key.
    It should return a list in all other cases including when there are no
    results for a good key.
    """

    if not isinstance(key, Hashable):
        if default is None:
            raise ValueError(
                "`default` must be provided when `key` is not not a valid DataArray name (of hashable type)."
            )
        return list(always_iterable(default))

    default = [] if default is None else list(always_iterable(default))

    def _apply_single_mapper(mapper):

        try:
            results = mapper(obj, key)
        except (KeyError, ValueError) as e:
            if error or "I expected only one." in repr(e):
                raise e
            else:
                results = []
        return results

    if not isinstance(mappers, Iterable):
        mappers = (mappers,)

    # apply a sequence of mappers
    # if the mapper fails, it *should* return an empty list
    # if the mapper raises an error, that is processed based on `error`
    results = []
    for mapper in mappers:
        results.append(_apply_single_mapper(mapper))

    flat = list(itertools.chain(*results))
    # de-duplicate
    if all(not isinstance(r, DataArray) for r in flat):
        results = list(set(flat))
    else:
        results = flat

    nresults = any(bool(v) for v in [results])
    if not nresults:
        if error:
            raise KeyError(
                f"cf-xarray cannot interpret key {key!r}. Perhaps some needed attributes are missing."
            )
        else:
            # none of the mappers worked. Return the default
            return default
    return results


def _get_groupby_time_accessor(
    obj: DataArray | Dataset, key: Hashable
) -> list[Hashable]:
    # This first docstring is used by _build_docstring. Do not remove.
    """
    Time variable accessor e.g. 'T.month'
    """
    """
    Helper method for decoding datetime components "T.month".

    Parameters
    ----------
    obj : DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key : str, [e.g. "T.month"]
        key to check for.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that matches axis or coordinate `key` appended by the frequency extension (e.g. ".month")

    Notes
    -----
    Returns an empty list if there is no frequency extension specified.
    """

    assert isinstance(key, str)
    if "." in key:
        key, ext = key.split(".", 1)

        results = apply_mapper((_get_all,), obj, key, error=False)
        if len(results) > 1:
            raise KeyError(f"Multiple results received for {key}.")
        var = results[0]
        if not isinstance(var, str):
            raise KeyError(f"Resolved {key} to {var} but {var} is not a string.")
        return [var + "." + ext]

    else:
        return []


def _get_custom_criteria(
    obj: DataArray | Dataset, key: Hashable, criteria: Mapping | None = None
) -> list[Hashable]:
    """
    Translate from axis, coord, or custom name to variable name optionally
    using ``custom_criteria``

    Parameters
    ----------
    obj : DataArray, Dataset
    key : str
        key to check for.
    criteria : dict, optional
        Criteria to use to map from variable to attributes describing the
        variable. An example is coordinate_criteria which maps coordinates to
        their attributes and attribute values. If user has defined
        custom_criteria, this will be used by default.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that matches axis, coordinate, or custom `key`
    """

    try:
        from regex import match as regex_match
    except ImportError:
        from re import match as regex_match  # type: ignore

    if isinstance(obj, DataArray):
        obj = obj._to_temp_dataset()

    if criteria is None:
        if not OPTIONS["custom_criteria"]:
            return []
        criteria = OPTIONS["custom_criteria"]

    if criteria is not None:
        criteria_iter = always_iterable(criteria, allowed=(tuple, list, set))

    criteria_map = ChainMap(*criteria_iter)
    results: set = set()
    if key in criteria_map:
        for criterion, patterns in criteria_map[key].items():
            for var in obj.variables:
                if regex_match(patterns, obj[var].attrs.get(criterion, "")):
                    results.update((var,))
                # also check name specifically since not in attributes
                elif (
                    criterion == "name"
                    and isinstance(var, str)
                    and regex_match(patterns, var)
                ):
                    results.update((var,))
    return list(results)


def _get_axis_coord(obj: DataArray | Dataset, key: str) -> list[str]:
    """
    Translate from axis or coord name to variable name

    Parameters
    ----------
    obj : DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key : str, ["X", "Y", "Z", "T", "longitude", "latitude", "vertical", "time"]
        key to check for.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that matches axis or coordinate `key`

    Notes
    -----
    This functions checks for the following attributes in order
       - `standard_name` (CF option)
       - `_CoordinateAxisType` (from THREDDS)
       - `axis` (CF option)
       - `positive` (CF standard for non-pressure vertical coordinate)

    References
    ----------
    MetPy's parse_cf
    """

    valid_keys = _COORD_NAMES + _AXIS_NAMES
    if key not in valid_keys:
        raise KeyError(
            f"cf_xarray did not understand key {key!r}. Expected one of {valid_keys!r}"
        )

    search_in = set()
    attrs_or_encoding = ChainMap(obj.attrs, obj.encoding)
    coordinates = attrs_or_encoding.get("coordinates", None)
    # Handles case where the coordinates attribute is None
    # This is used to tell xarray to not write a coordinates attribute
    if coordinates:
        search_in.update(coordinates.split(" "))
    if not search_in:
        search_in = set(obj.coords)

    # maybe only do this for key in _AXIS_NAMES?
    search_in.update(obj.indexes)

    search_in = search_in & set(obj.coords)
    results: set = set()
    for coord in search_in:
        var = obj.coords[coord]
        if key in coordinate_criteria:
            for criterion, expected in coordinate_criteria[key].items():
                if var.attrs.get(criterion, None) in expected:
                    results.update((coord,))
                if criterion == "units":
                    # deal with pint-backed objects
                    units = getattr(var.data, "units", None)
                    if units in expected:
                        results.update((coord,))

    if key in _AXIS_NAMES and "grid_topology" in obj.cf.cf_roles:
        sgrid_axes = sgrid.parse_axes(obj)
        results.update((search_in | set(obj.dims)) & sgrid_axes[key])

    return list(results)


def _get_measure(obj: DataArray | Dataset, key: str) -> list[str]:
    """
    Translate from cell measures to appropriate variable name.
    This function interprets the ``cell_measures`` attribute on DataArrays.

    Parameters
    ----------
    obj : DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key : str
        key to check for.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that matches axis or coordinate `key`
    """

    if isinstance(obj, DataArray):
        obj = obj._to_temp_dataset()

    results = set()
    for var in obj.variables:
        da = obj[var]
        attrs_or_encoding = ChainMap(da.attrs, da.encoding)
        if "cell_measures" in attrs_or_encoding:
            attr = attrs_or_encoding["cell_measures"]
            try:
                measures = parse_cell_methods_attr(attr)
            except ValueError as e:
                raise ValueError(
                    f"{var} has malformed cell_measures attribute {attr}."
                ) from e
            if key in measures:
                results.update([measures[key]])

    if isinstance(results, str):
        return [results]
    return list(results)


def _get_bounds(obj: DataArray | Dataset, key: Hashable) -> list[Hashable]:
    """
    Translate from key (either CF key or variable name) to its bounds' variable names.
    This function interprets the ``bounds`` attribute on DataArrays.

    Parameters
    ----------
    obj : DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key : str
        key to check for.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that are bounds of `key`
    """

    results = set()
    for var in apply_mapper(_get_all, obj, key, error=False, default=[key]):
        attrs_or_encoding = ChainMap(obj[var].attrs, obj[var].encoding)
        if "bounds" in attrs_or_encoding:
            results |= {attrs_or_encoding["bounds"]}

    return list(results)


def _get_grid_mapping_name(obj: DataArray | Dataset, key: str) -> list[str]:
    """
    Translate from grid mapping name attribute to appropriate variable name.
    This function interprets the ``grid_mapping`` attribute on DataArrays.

    Parameters
    ----------
    obj : DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key : str
        key to check for.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that matches grid_mapping_name `key`
    """

    if isinstance(obj, DataArray):
        obj = obj._to_temp_dataset()

    results = set()
    for var in obj.variables:
        da = obj[var]
        attrs_or_encoding = ChainMap(da.attrs, da.encoding)
        if "grid_mapping" in attrs_or_encoding:
            grid_mapping_var_name = attrs_or_encoding["grid_mapping"]
            if grid_mapping_var_name not in obj.variables:
                raise ValueError(
                    f"{var} defines non-existing grid_mapping variable {grid_mapping_var_name}."
                )
            if key == obj[grid_mapping_var_name].attrs["grid_mapping_name"]:
                results.update([grid_mapping_var_name])
    return list(results)


def _get_with_standard_name(
    obj: DataArray | Dataset, name: Hashable | Iterable[Hashable]
) -> list[Hashable]:
    """returns a list of variable names with standard name == name."""
    if name is None:
        return []

    varnames: list[Hashable] = []
    if isinstance(obj, DataArray):
        obj = obj.coords.to_dataset()
    for vname, var in obj._variables.items():
        stdname = var.attrs.get("standard_name", None)
        if stdname == name:
            varnames.append(vname)

    return varnames


def _get_all(obj: DataArray | Dataset, key: Hashable) -> list[Hashable]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names
    """
    all_mappers: tuple[Mapper] = (
        _get_custom_criteria,
        functools.partial(_get_custom_criteria, criteria=cf_role_criteria),  # type: ignore
        functools.partial(_get_custom_criteria, criteria=grid_mapping_var_criteria),
        _get_axis_coord,
        _get_measure,
        _get_grid_mapping_name,
        _get_with_standard_name,
    )
    results = apply_mapper(all_mappers, obj, key, error=False, default=None)
    return results


def _get_dims(obj: DataArray | Dataset, key: Hashable) -> list[Hashable]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names present in .dims
    """
    return [k for k in _get_all(obj, key) if k in obj.dims]


def _get_indexes(obj: DataArray | Dataset, key: Hashable) -> list[Hashable]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names present in .indexes
    """
    return [k for k in _get_all(obj, key) if k in obj.indexes]


def _get_coords(obj: DataArray | Dataset, key: Hashable) -> list[Hashable]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names present in .coords
    """
    return [k for k in _get_all(obj, key) if k in obj.coords or k in obj.dims]


def _variables(func: F) -> F:
    @functools.wraps(func)
    def wrapper(obj: DataArray | Dataset, key: Hashable) -> list[DataArray]:
        return [obj[k] for k in func(obj, key)]  # type: ignore

    return cast(F, wrapper)


def _single(func: F) -> F:
    @functools.wraps(func)
    def wrapper(obj: DataArray | Dataset, key: Hashable):
        results = func(obj, key)
        if len(results) > 1:
            raise KeyError(
                f"Multiple results for {key!r} found: {results!r}. I expected only one."
            )
        elif len(results) == 0:
            raise KeyError(f"No results found for {key!r}.")
        return results

    wrapper.__doc__ = (
        func.__doc__.replace("One or more of", "One of")
        if func.__doc__
        else func.__doc__
    )

    return cast(F, wrapper)


#: Default mappers for common keys.
_DEFAULT_KEY_MAPPERS: Mapping[str, tuple[Mapper, ...]] = {
    "dim": (_get_dims,),
    "dims": (_get_dims,),  # transpose
    "drop_dims": (_get_dims,),  # drop_dims
    "dims_dict": (_get_dims,),  # swap_dims, rename_dims
    "shifts": (_get_dims,),  # shift, roll
    "pad_width": (_get_dims,),  # shift, roll
    "names": (_get_all,),  # set_coords, reset_coords, drop_vars
    "name_dict": (_get_all,),  # rename, rename_vars
    "new_name_or_name_dict": (_get_all,),  # rename
    "labels": (_get_indexes,),  # drop_sel
    "coords": (_get_dims,),  # interp
    "indexers": (_get_dims,),  # sel, isel, reindex
    #  "indexes": (_single(_get_dims),),  # set_index this decodes keys but not values
    "dims_or_levels": (_get_dims,),  # reset_index
    "window": (_get_dims,),  # rolling_exp
    "coord": (_single(_get_coords),),  # differentiate, integrate
    "group": (_single(_get_all), _get_groupby_time_accessor),  # groupby
    "indexer": (_single(_get_indexes),),  # resample
    "variables": (_get_all,),  # sortby
    "weights": (_variables(_single(_get_all)),),
    "chunks": (_get_dims,),  # chunk
}


def _guess_bounds(da, dim=None, out_dim="bounds"):
    """
    Guess bounds values given a 1D or 2D coordinate variable.
    Assumes equal spacing on either side of the coordinate label.
    This is a coarse approximation, especially for 2D bounds on curvilinear grids.
    """
    if dim is None:
        if da.ndim not in [1, 2]:
            raise ValueError(
                f"If dim is None, variable {da.name} must be 1D or 2D. Received {da.ndim}D variable instead."
            )
        dim = da.dims

    if not isinstance(dim, str):
        if len(dim) > 2:
            raise NotImplementedError(
                "Adding bounds with more than 2 dimensions is not supported."
            )
        elif len(dim) == 2:
            return _guess_bounds_2d(da, dim).rename(bounds=out_dim)
        else:
            dim = dim[0]

    return _guess_bounds_1d(da, dim).rename(bounds=out_dim)


def _build_docstring(func):
    """
    Builds a nice docstring for wrapped functions, stating what key words
    can be used for arguments.
    """

    sig = inspect.signature(func)
    string = ""
    for k in set(sig.parameters.keys()) & set(_DEFAULT_KEY_MAPPERS):
        mappers = _DEFAULT_KEY_MAPPERS.get(k, [])
        docstring = ";\n\t\t\t".join(
            mapper.__doc__ if mapper.__doc__ else "unknown. please open an issue."
            for mapper in mappers
        )
        string += f"\t\t{k}: {docstring} \n"

    for param in sig.parameters:
        if sig.parameters[param].kind is inspect.Parameter.VAR_KEYWORD:
            string += f"\t\t{param}: {_get_all.__doc__} \n\n"
    return (
        f"\n\tThe following arguments will be processed by cf_xarray: \n{string}"
        "\n\t----\n\t"
    )


def _getattr(
    obj: DataArray | Dataset,
    attr: str,
    accessor: CFAccessor,
    key_mappers: Mapping[str, Mapper],
    wrap_classes: bool = False,
    extra_decorator: Callable | None = None,
):
    """
    Common getattr functionality.

    Parameters
    ----------
    obj : DataArray, Dataset
    attr : Name of attribute in obj that will be shadowed.
    accessor : High level accessor object: CFAccessor
    key_mappers : dict
        dict(key_name: mapper)
    wrap_classes : bool
        Should we wrap the return value with _CFWrappedClass?
        Only True for the high level CFAccessor.
        Facilitates code reuse for _CFWrappedClass and _CFWrapppedPlotMethods
        For both of those, wrap_classes is False.
    extra_decorator : Callable (optional)
        An extra decorator, if necessary. This is used by _CFPlotMethods to set default
        kwargs based on CF attributes.
    """

    # UGH. this seems unavoidable because I'm overriding getattr
    if attr in ["_repr_html_", "__rich__", "__rich_console__"]:
        raise AttributeError

    try:
        attribute: Mapping | Callable = getattr(obj, attr)
    except AttributeError:
        if getattr(
            CFDatasetAccessor if isinstance(obj, DataArray) else CFDataArrayAccessor,
            attr,
            None,
        ):
            raise AttributeError(
                f"{obj.__class__.__name__+'.cf'!r} object has no attribute {attr!r}"
            )
        raise AttributeError(
            f"{attr!r} is not a valid attribute on the underlying xarray object."
        )

    if isinstance(attribute, Mapping):
        if not attribute:
            return dict(attribute)

        newmap = {}
        inverted = invert_mappings(
            accessor.axes,
            accessor.coordinates,
            accessor.cell_measures,
            accessor.standard_names,
        )
        unused_keys = set(attribute.keys()) - set(inverted)
        for key, value in attribute.items():
            for name in inverted[key]:
                if name in newmap:
                    raise AttributeError(
                        f"cf_xarray can't wrap attribute {attr!r} because there are multiple values for {name!r}. "
                        f"There is no unique mapping from {name!r} to a value in {attr!r}."
                    )
            newmap.update(dict.fromkeys(inverted[key], value))
        newmap.update({key: attribute[key] for key in unused_keys})

        skip: dict[str, list[Hashable] | None] = {
            "data_vars": ["coords"],
            "coords": None,
        }
        if attr in ["coords", "data_vars"]:
            for key in newmap:
                newmap[key] = _getitem(accessor, key, skip=skip[attr])
        return newmap

    elif isinstance(attribute, Callable):  # type: ignore
        func: Callable = attribute

    else:
        raise AttributeError(
            f"cf_xarray does not know how to wrap attribute '{type(obj).__name__}.{attr}'. "
            "Please file an issue if you have a solution."
        )

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        posargs, arguments = accessor._process_signature(
            func, args, kwargs, key_mappers
        )
        final_func = extra_decorator(func) if extra_decorator else func
        result = final_func(*posargs, **arguments)
        if wrap_classes and isinstance(result, _WRAPPED_CLASSES):
            result = _CFWrappedClass(result, accessor)

        return result

    # handle rich
    if wrapper.__doc__:
        wrapper.__doc__ = _build_docstring(func) + wrapper.__doc__

    return wrapper


@overload
def _getitem(
    accessor: CFAccessor,
    key: Hashable,
    skip: list[Hashable] | None = None,
) -> DataArray:
    ...


@overload
def _getitem(
    accessor: CFAccessor,
    key: Iterable[Hashable],
    skip: list[Hashable] | None = None,
) -> Dataset:
    ...


def _getitem(
    accessor,
    key,
    skip=None,
):
    """
    Index into obj using key. Attaches CF associated variables.

    Parameters
    ----------
    accessor : CFAccessor
    key : str, List[str]
    skip : str, optional
        One of ["coords", "measures"], avoid clashes with special coord names
    """

    obj = accessor._obj
    all_bounds = obj.cf.bounds if isinstance(obj, Dataset) else {}
    kind = str(type(obj).__name__)
    scalar_key = isinstance(key, Hashable)

    key_iter: Iterable[Hashable]
    if isinstance(key, Hashable):  # using scalar_key breaks mypy type narrowing
        key_iter = (key,)
    else:
        key_iter = key

    if skip is None:
        skip = []

    def drop_bounds(names):
        # sometimes bounds variables have the same standard_name as the
        # actual variable. It seems practical to ignore them when indexing
        # with a scalar key. Hopefully these will soon get decoded to IntervalIndex
        # and we can move on...
        if not isinstance(obj, DataArray) and scalar_key:
            bounds = set()
            for name in names:
                bounds.update(all_bounds.get(name, []))
            names = set(names) - bounds
        return names

    def check_results(names, key):
        if scalar_key and len(names) > 1:
            raise KeyError(
                f"Receive multiple variables for key {key!r}: {names}. "
                f"Expected only one. Please pass a list [{key!r}] "
                f"instead to get all variables matching {key!r}."
            )

    try:
        measures = accessor._get_all_cell_measures()
    except ValueError:
        measures = []
        warnings.warn("Ignoring bad cell_measures attribute.", UserWarning)

    if isinstance(obj, Dataset):
        grid_mapping_names = list(accessor.grid_mapping_names)
    else:
        try:
            grid_mapping_names = [accessor.grid_mapping_name]
        except ValueError:
            grid_mapping_names = []
    grid_mapping_names.append("grid_mapping")

    custom_criteria = ChainMap(*OPTIONS["custom_criteria"])

    varnames: list[Hashable] = []
    coords: list[Hashable] = []
    successful = dict.fromkeys(key_iter, False)
    for k in key_iter:
        if "coords" not in skip and k in _AXIS_NAMES + _COORD_NAMES:
            names = _get_all(obj, k)
            names = drop_bounds(names)
            check_results(names, k)
            successful[k] = bool(names)
            coords.extend(names)
        elif "measures" not in skip and k in measures:
            measure = _get_all(obj, k)
            check_results(measure, k)
            successful[k] = bool(measure)
            if measure:
                varnames.extend(measure)
        elif "grid_mapping_names" not in skip and k in grid_mapping_names:
            grid_mapping = _get_all(obj, k)
            check_results(grid_mapping, k)
            successful[k] = bool(grid_mapping)
            if grid_mapping:
                varnames.extend(grid_mapping)
        elif k in custom_criteria or k in cf_role_criteria:
            names = _get_all(obj, k)
            check_results(names, k)
            successful[k] = bool(names)
            varnames.extend(names)
        else:
            stdnames = set(_get_with_standard_name(obj, k))
            objcoords = set(obj.coords)
            stdnames = drop_bounds(stdnames)
            if "coords" in skip:
                stdnames -= objcoords
            check_results(stdnames, k)
            successful[k] = bool(stdnames)
            varnames.extend(stdnames - objcoords)
            coords.extend(stdnames & objcoords)

    # these are not special names but could be variable names in underlying object
    # we allow this so that we can return variables with appropriate CF auxiliary variables
    varnames.extend([k for k, v in successful.items() if not v])
    allnames = varnames + coords

    try:
        for name in allnames:
            extravars = accessor.get_associated_variable_names(
                name, skip_bounds=scalar_key, error=False
            )
            coords.extend(itertools.chain(*extravars.values()))

        if isinstance(obj, DataArray):
            ds = obj._to_temp_dataset()
        else:
            ds = obj

        if scalar_key:
            if len(allnames) == 1:
                da: DataArray = ds.reset_coords()[allnames[0]]  # type: ignore
                if allnames[0] in coords:
                    coords.remove(allnames[0])
                for k1 in coords:
                    da.coords[k1] = ds.variables[k1]
                return da
            else:
                raise KeyError(
                    f"Received scalar key {key!r} but multiple results: {allnames!r}. "
                    f"Please pass a list instead (['{key}']) to get back a Dataset "
                    f"with {allnames!r}."
                )

        ds = ds.reset_coords()[varnames + coords]
        if isinstance(obj, DataArray):
            if scalar_key and len(ds.variables) == 1:
                # single dimension coordinates
                assert coords
                assert not varnames

                return ds[coords[0]]

            elif scalar_key and len(ds.variables) > 1:
                raise NotImplementedError(
                    "Not sure what to return when given scalar key for DataArray and it has multiple values. "
                    "Please open an issue."
                )

        return ds.set_coords(coords)

    except KeyError:
        raise KeyError(
            f"{kind}.cf does not understand the key {k!r}. "
            f"Use 'repr({kind}.cf)' (or '{kind}.cf' in a Jupyter environment) to see a list of key names that can be interpreted."
        )


def _possible_x_y_plot(obj, key, skip=None):
    """Guesses a name for an x/y variable if possible."""
    # in priority order
    x_criteria = [
        ("coordinates", "longitude"),
        ("axes", "X"),
        ("coordinates", "time"),
        ("axes", "T"),
    ]
    y_criteria = [
        ("coordinates", "vertical"),
        ("axes", "Z"),
        ("coordinates", "latitude"),
        ("axes", "Y"),
    ]

    def _get_possible(accessor, criteria):
        # is_scalar depends on NON_NUMPY_SUPPORTED_TYPES
        # importing a private function seems better than
        # maintaining that variable!
        from xarray.core.utils import is_scalar

        for attr, key in criteria:
            values = getattr(accessor, attr).get(key)
            ax_coord_name = getattr(accessor, attr).get(key)
            if not values:
                continue
            elif ax_coord_name:
                values = [v for v in values if v in ax_coord_name]

            if skip is not None:
                skipvar = obj.cf[skip]
                bad_names = (skip, skipvar.name)
                bad_dims = ((skip,), skipvar.dims)
                values = [
                    v
                    for v in values
                    if v not in bad_names and obj[v].dims not in bad_dims
                ]
            if len(values) == 1 and not is_scalar(accessor._obj[values[0]]):
                return values[0]
            else:
                for v in values:
                    if not is_scalar(accessor._obj[v]):
                        return v
        return None

    if key == "x":
        return _get_possible(obj.cf, x_criteria)
    elif key == "y":
        return _get_possible(obj.cf, y_criteria)


class _CFWrappedClass(SupportsArithmetic):
    """
    This class is used to wrap any class in _WRAPPED_CLASSES.
    """

    def __init__(self, towrap, accessor: CFAccessor):
        """
        Parameters
        ----------
        towrap : Resample, GroupBy, Coarsen, Rolling, Weighted
            Instance of xarray class that is being wrapped.
        accessor : CFAccessor
            Parent accessor object
        """
        self.wrapped = towrap
        self.accessor = accessor

    def __repr__(self):
        return "--- CF-xarray wrapped \n" + repr(self.wrapped)

    def __getattr__(self, attr):
        return _getattr(
            obj=self.wrapped,
            attr=attr,
            accessor=self.accessor,
            key_mappers=_DEFAULT_KEY_MAPPERS,
        )

    def __iter__(self):
        return iter(self.wrapped)


class _CFWrappedPlotMethods:
    """
    This class wraps DataArray.plot
    """

    def __init__(self, obj, accessor):
        self._obj = obj
        self.accessor = accessor
        self._keys = ("x", "y", "hue", "col", "row")

    def _process_x_or_y(self, kwargs, key, skip=None):
        """Choose a default 'x' or 'y' variable name."""
        if key not in kwargs:
            kwargs[key] = _possible_x_y_plot(self._obj, key, skip)
        return kwargs

    def _set_axis_props(self, kwargs, key):
        value = kwargs.get(key)
        if value:
            if value in self.accessor.keys():
                var = self.accessor[value]
            else:
                var = self._obj[value]
            if "positive" in var.attrs:
                if var.attrs["positive"] == "down":
                    kwargs.setdefault(f"{key}increase", False)
                else:
                    kwargs.setdefault(f"{key}increase", True)
        return kwargs

    def _plot_decorator(self, func):
        """
        This decorator is used to set default kwargs on plotting functions.
        For now, this can
        1. set ``xincrease`` and ``yincrease``.
        2. automatically set ``x`` or ``y``.
        """

        @functools.wraps(func)
        def _plot_wrapper(*args, **kwargs):
            # First choose 'x' or 'y' if possible
            is_line_plot = (func.__name__ == "line") or (
                func.__name__ == "wrapper"
                and (kwargs.get("hue") or self._obj.ndim == 1)
            )
            if is_line_plot:
                hue = kwargs.get("hue")
                if "x" not in kwargs and "y" not in kwargs:
                    kwargs = self._process_x_or_y(kwargs, "x", skip=hue)
                    if not kwargs.get("x"):
                        kwargs = self._process_x_or_y(kwargs, "y", skip=hue)
            else:
                kwargs = self._process_x_or_y(kwargs, "x", skip=kwargs.get("y"))
                kwargs = self._process_x_or_y(kwargs, "y", skip=kwargs.get("x"))

            # Now set some nice properties
            kwargs = self._set_axis_props(kwargs, "x")
            kwargs = self._set_axis_props(kwargs, "y")

            return func(*args, **kwargs)

        return _plot_wrapper

    def __call__(self, *args, **kwargs):
        """
        Allows .plot()
        """
        plot = _getattr(
            obj=self._obj,
            attr="plot",
            accessor=self.accessor,
            key_mappers=dict.fromkeys(self._keys, (_single(_get_all),)),
        )
        return self._plot_decorator(plot)(*args, **kwargs)

    def __getattr__(self, attr):
        """
        Wraps .plot.contour() for example.
        """
        return _getattr(
            obj=self._obj.plot,
            attr=attr,
            accessor=self.accessor,
            key_mappers=dict.fromkeys(self._keys, (_single(_get_all),)),
            # TODO: "extra_decorator" is more complex than I would like it to be.
            # Not sure if there is a better way though
            extra_decorator=self._plot_decorator,
        )


def create_flag_dict(da):
    if not da.cf.is_flag_variable:
        raise ValueError(
            "Comparisons are only supported for DataArrays that represent CF flag variables."
            ".attrs must contain 'flag_values' and 'flag_meanings'"
        )

    flag_meanings = da.attrs["flag_meanings"].split(" ")
    flag_values = da.attrs["flag_values"]
    # TODO: assert flag_values is iterable
    assert len(flag_values) == len(flag_meanings)
    return dict(zip(flag_meanings, flag_values))


class CFAccessor:
    """
    Common Dataset and DataArray accessor functionality.
    """

    def __init__(self, obj):
        self._obj = obj
        self._all_cell_measures = None

    def __setstate__(self, d):
        self.__dict__ = d

    def _assert_valid_other_comparison(self, other):
        flag_dict = create_flag_dict(self._obj)
        if other not in flag_dict:
            raise ValueError(
                f"Did not find flag value meaning [{other}] in known flag meanings: [{flag_dict.keys()!r}]"
            )
        return flag_dict

    def __eq__(self, other):
        """
        Compare flag values against `other`.

        `other` must be in the 'flag_meanings' attribute.
        `other` is mapped to the corresponding value in the 'flag_values' attribute, and then
        compared.
        """
        flag_dict = self._assert_valid_other_comparison(other)
        return self._obj == flag_dict[other]

    def __ne__(self, other):
        """
        Compare flag values against `other`.

        `other` must be in the 'flag_meanings' attribute.
        `other` is mapped to the corresponding value in the 'flag_values' attribute, and then
        compared.
        """
        flag_dict = self._assert_valid_other_comparison(other)
        return self._obj != flag_dict[other]

    def __lt__(self, other):
        """
        Compare flag values against `other`.

        `other` must be in the 'flag_meanings' attribute.
        `other` is mapped to the corresponding value in the 'flag_values' attribute, and then
        compared.
        """
        flag_dict = self._assert_valid_other_comparison(other)
        return self._obj < flag_dict[other]

    def __le__(self, other):
        """
        Compare flag values against `other`.

        `other` must be in the 'flag_meanings' attribute.
        `other` is mapped to the corresponding value in the 'flag_values' attribute, and then
        compared.
        """
        flag_dict = self._assert_valid_other_comparison(other)
        return self._obj <= flag_dict[other]

    def __gt__(self, other):
        """
        Compare flag values against `other`.

        `other` must be in the 'flag_meanings' attribute.
        `other` is mapped to the corresponding value in the 'flag_values' attribute, and then
        compared.
        """
        flag_dict = self._assert_valid_other_comparison(other)
        return self._obj > flag_dict[other]

    def __ge__(self, other):
        """
        Compare flag values against `other`.

        `other` must be in the 'flag_meanings' attribute.
        `other` is mapped to the corresponding value in the 'flag_values' attribute, and then
        compared.
        """
        flag_dict = self._assert_valid_other_comparison(other)
        return self._obj >= flag_dict[other]

    def isin(self, test_elements):
        """Test each value in the array for whether it is in test_elements.

        Parameters
        ----------
        test_elements : array_like, 1D
            The values against which to test each value of `element`.
            These must be in "flag_meanings" attribute, and are mapped
            to the corresponding value in "flag_values" before passing
            that on to DataArray.isin.

        Returns
        -------
        isin : DataArray
            Has the same type and shape as this object, but with a bool dtype.
        """
        if not isinstance(self._obj, DataArray):
            raise ValueError(
                ".cf.isin is only supported on DataArrays that contain CF flag attributes."
            )
        flag_dict = create_flag_dict(self._obj)
        mapped_test_elements = []
        for elem in test_elements:
            if elem not in flag_dict:
                raise ValueError(
                    f"Did not find flag value meaning [{elem}] in known flag meanings: [{flag_dict.keys()!r}]"
                )
            mapped_test_elements.append(flag_dict[elem])
        return self._obj.isin(mapped_test_elements)

    def _drop_missing_variables(self, variables: list[Hashable]) -> list[Hashable]:
        if isinstance(self._obj, Dataset):
            good_names = set(self._obj._variables)
        elif isinstance(self._obj, DataArray):
            good_names = set(self._obj._coords)

        return [var for var in variables if var in good_names]

    def _get_all_cell_measures(self):
        """
        Get all cell measures defined in the object, adding CF pre-defined measures.
        """

        # get all_cell_measures only once
        if not self._all_cell_measures:
            self._all_cell_measures = set(_CELL_MEASURES + tuple(self.cell_measures))

        return self._all_cell_measures

    def curvefit(
        self,
        coords: Hashable | DataArray | Iterable[Hashable | DataArray],
        func: Callable[..., Any],
        reduce_dims: Hashable | Iterable[Hashable] | None = None,
        skipna: bool = True,
        p0: dict[str, Any] | None = None,
        bounds: dict[str, Any] | None = None,
        param_names: Sequence[str] | None = None,
        kwargs: dict[str, Any] | None = None,
    ):

        if coords is not None:
            if isinstance(coords, (Hashable, DataArray)):
                coords_iter: Iterable[Hashable | DataArray] = [coords]
            else:
                coords_iter = coords
            coords = [
                apply_mapper(
                    [_single(_get_coords)], self._obj, v, error=False, default=[v]  # type: ignore
                )[0]
                for v in coords_iter
            ]
        if reduce_dims is not None:
            if isinstance(reduce_dims, Hashable):
                reduce_dims_iter = [reduce_dims]
            else:
                reduce_dims_iter = list(reduce_dims)
            reduce_dims = [
                apply_mapper(
                    [_single(_get_dims)], self._obj, v, error=False, default=[v]  # type: ignore
                )[0]
                for v in reduce_dims_iter
            ]

        return self._obj.curvefit(
            coords=coords,
            func=func,
            reduce_dims=reduce_dims,
            skipna=skipna,
            p0=p0,
            bounds=bounds,
            param_names=param_names,
            kwargs=kwargs,
        )

    def _process_signature(
        self,
        func: Callable,
        args,
        kwargs,
        key_mappers: MutableMapping[str, tuple[Mapper, ...]],
    ):
        """
        Processes a function's signature, args, kwargs:
        1. Binds ``*args`` so that everything is a Mapping from kwarg name to values
        2. Calls ``_rewrite_values`` to rewrite any special CF names to normal xarray names.
           This uses ``key_mappers``
        3. Unpacks arguments if necessary before returning them.
        """
        sig = inspect.signature(func, follow_wrapped=False)

        # Catch things like .isel(T=5).
        # This assigns indexers_kwargs=dict(T=5).
        # and indexers_kwargs is of kind VAR_KEYWORD
        var_kws: list = []
        # capture *args, e.g. transpose
        var_args: list = []
        for param in sig.parameters:
            if sig.parameters[param].kind is inspect.Parameter.VAR_KEYWORD:
                var_kws.append(param)
            elif sig.parameters[param].kind is inspect.Parameter.VAR_POSITIONAL:
                var_args.append(param)

        posargs = []
        if args or kwargs:
            bound = sig.bind(*args, **kwargs)
            arguments = self._rewrite_values(
                bound.arguments, key_mappers, tuple(var_kws)
            )

            # unwrap the *args type arguments
            for arg in var_args:
                value = arguments.pop(arg, None)
                if value:
                    # value should always be Iterable
                    posargs.extend(value)
            # now unwrap the **kwargs type arguments
            for kw in var_kws:
                value = arguments.pop(kw, None)
                if value:
                    arguments.update(**value)
        else:
            arguments = {}

        return posargs, arguments

    def _rewrite_values(
        self,
        kwargs,
        key_mappers: MutableMapping[str, tuple[Mapper, ...]],
        var_kws: tuple[str, ...],
    ):
        """
        Rewrites the values in a Mapping from kwarg to value.

        Parameters
        ----------
        kwargs : Mapping
            Mapping from kwarg name to value
        key_mappers : Mapping
            Mapping from kwarg name to a Mapper function that will convert a
            given CF "special" name to an xarray name.
        var_kws : List[str]
            List of variable kwargs that need special treatment.
            e.g. **indexers_kwargs in isel

        Returns
        -------
        dict of kwargs with fully rewritten values.
        """
        updates: dict = {}

        # allow multiple return values here.
        # these are valid for .sel, .isel, .coarsen
        all_mappers = ChainMap(  # type: ignore
            key_mappers,
            dict.fromkeys(var_kws, (_get_all,)),
        )

        for key in set(all_mappers) & set(kwargs):
            value = kwargs[key]
            mappers = all_mappers[key]

            value = always_iterable(value)

            if isinstance(value, dict):
                # this for things like isel where **kwargs captures things like T=5
                # .sel, .isel, .rolling
                # Account for multiple names matching the key.
                # e.g. .isel(X=5) → .isel(xi_rho=5, xi_u=5, xi_v=5, xi_psi=5)
                # where xi_* have attrs["axis"] = "X"
                updates[key] = ChainMap(
                    *(
                        dict.fromkeys(
                            apply_mapper(mappers, self._obj, k, False, [k]), v
                        )
                        for k, v in value.items()
                    )
                )

            elif value is Ellipsis:
                pass

            else:
                # things like sum which have dim
                newvalue = [
                    apply_mapper(mappers, self._obj, v, error=False, default=[v])
                    for v in value
                ]
                # Mappers return list by default
                # for input dim=["lat", "X"], newvalue=[["lat"], ["lon"]],
                # so we deal with that here.
                unpacked = list(itertools.chain(*newvalue))
                if len(unpacked) == 1:
                    # handle 'group'
                    updates[key] = unpacked[0]
                else:
                    updates[key] = unpacked

        kwargs.update(updates)

        # TODO: is there a way to merge this with above?
        # maybe the keys we are looking for are in kwargs.
        # For example, this happens with DataArray.plot(),
        # where the signature is obscured and kwargs is
        #    kwargs = {"x": "X", "col": "T"}
        for vkw in var_kws:
            if vkw in kwargs:
                maybe_update = {
                    k: apply_mapper(
                        key_mappers[k], self._obj, v, error=False, default=[v]
                    )[0]
                    for k, v in kwargs[vkw].items()
                    if k in key_mappers
                }
                kwargs[vkw].update(maybe_update)

        return kwargs

    def __getattr__(self, attr):
        return _getattr(
            obj=self._obj,
            attr=attr,
            accessor=self,
            key_mappers=_DEFAULT_KEY_MAPPERS,
            wrap_classes=True,
        )

    def __contains__(self, item: str) -> bool:
        """
        Check whether item is a valid key for indexing with .cf
        """
        return item in self.keys()

    @property
    def plot(self):
        return _CFWrappedPlotMethods(self._obj, self)

    def describe(self):
        """
        Print a string repr to screen.
        """

        warnings.warn(
            "'obj.cf.describe()' will be removed in a future version. "
            "Use instead 'repr(obj.cf)' or 'obj.cf' in a Jupyter environment.",
            DeprecationWarning,
        )
        print(repr(self))

    def __repr__(self):
        return ("".join(self._generate_repr(rich=False))).rstrip()

    def __rich__(self):
        from rich.console import Group

        return Group(*self._generate_repr(rich=True))

    def _generate_repr(self, rich=False):
        dims = self._obj.dims
        coords = self._obj.coords

        # if self._obj._attrs:
        #     conventions = self._obj.attrs.pop("Conventions", None)
        #     if conventions:
        #         yield _format_conventions(conventions, rich)

        if isinstance(self._obj, DataArray) and self._obj.cf.is_flag_variable:
            yield _maybe_panel(
                _format_flags(self, rich), title="Flag Variable", rich=rich
            )

        roles = self.cf_roles
        if roles:
            if any(role in roles for role in _DSG_ROLES):
                yield _maybe_panel(
                    _format_dsg_roles(self, dims, rich),
                    title="Discrete Sampling Geometry",
                    rich=rich,
                )

            if "grid_topology" in self.cf_roles:
                axes = sgrid.parse_axes(self._obj)
                yield _maybe_panel(
                    _format_sgrid(self, axes, rich),
                    title="SGRID",
                    rich=rich,
                )

        yield _maybe_panel(
            _format_coordinates(self, dims, coords, rich),
            title="Coordinates",
            rich=rich,
        )
        if isinstance(self._obj, Dataset):
            yield _maybe_panel(
                _format_data_vars(self, self._obj.data_vars, rich),
                title="Data Variables",
                rich=rich,
            )

    def keys(self) -> set[Hashable]:
        """
        Utility function that returns valid keys for .cf[].

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.

        Returns
        -------
        set
            Set of valid key names that can be used with __getitem__ or .cf[key].
        """

        varnames = list(self.axes) + list(self.coordinates)
        varnames.extend(list(self.cell_measures))
        varnames.extend(list(self.standard_names))
        varnames.extend(list(self.cf_roles))
        if isinstance(self._obj, xr.Dataset):
            varnames.extend(list(self.grid_mapping_names))
        else:
            try:
                gmname = self.grid_mapping_name
                varnames.extend(list(gmname))
            except ValueError:
                pass

        return set(varnames)

    @property
    def axes(self) -> dict[str, list[Hashable]]:
        """
        Property that returns a dictionary mapping valid Axis standard names for ``.cf[]``
        to variable names.

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.
        However, it will only return the Axis names ``("X", "Y", "Z", "T")``
        present in ``.coords``, not in ``.data_vars``.

        Returns
        -------
        dict
            Dictionary with keys that can be used with ``__getitem__`` or as ``.cf[key]``.
            Keys will be the appropriate subset of ("X", "Y", "Z", "T").
            Values are lists of variable names that match that particular key.
        """
        vardict = {key: _get_coords(self._obj, key) for key in _AXIS_NAMES}

        return {k: sort_maybe_hashable(v) for k, v in vardict.items() if v}

    @property
    def coordinates(self) -> dict[str, list[Hashable]]:
        """
        Property that returns a dictionary mapping valid Coordinate standard names for ``.cf[]``
        to variable names.

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.
        However, it will only return the Coordinate names ``("latitude", "longitude", "vertical", "time")``
        present in ``.coords``, not in ``.data_vars``.

        Returns
        -------
        dict
            Dictionary of valid Coordinate names that can be used with ``__getitem__`` or ``.cf[key]``.
            Keys will be the appropriate subset of ``("latitude", "longitude", "vertical", "time")``.
            Values are lists of variable names that match that particular key.

        """
        vardict = {key: _get_coords(self._obj, key) for key in _COORD_NAMES}

        return {k: sort_maybe_hashable(v) for k, v in vardict.items() if v}

    @property
    def cell_measures(self) -> dict[str, list[Hashable]]:
        """
        Property that returns a dictionary mapping valid cell measure standard names for ``.cf[]``
        to variable names.

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.

        Returns
        -------
        dict
            Dictionary of valid cell measure names that can be used with ``__getitem__`` or ``.cf[key]``.
        """

        obj = self._obj
        all_attrs = [
            ChainMap(da.attrs, da.encoding).get("cell_measures", "")
            for da in obj.coords.values()
        ]
        if isinstance(obj, DataArray):
            all_attrs += [ChainMap(obj.attrs, obj.encoding).get("cell_measures", "")]
        elif isinstance(obj, Dataset):
            all_attrs += [
                ChainMap(da.attrs, da.encoding).get("cell_measures", "")
                for da in obj.data_vars.values()
            ]
        as_dataset = self._maybe_to_dataset().reset_coords()

        keys = {}
        for attr in set(all_attrs):
            try:
                keys.update(parse_cell_methods_attr(attr))
            except ValueError:
                bad_vars = list(
                    as_dataset.filter_by_attrs(cell_measures=attr).data_vars.keys()
                )
                warnings.warn(
                    f"Ignoring bad cell_measures attribute: {attr} on {bad_vars}.",
                    UserWarning,
                    stacklevel=2,
                )
        measures = {
            key: self._drop_missing_variables(_get_all(self._obj, key)) for key in keys
        }

        return {k: sort_maybe_hashable(set(v)) for k, v in measures.items() if v}

    @property
    def standard_names(self) -> dict[str, list[Hashable]]:
        """
        Returns a dictionary mapping standard names to variable names.

        Returns
        -------
        dict
            Dictionary mapping standard names to variable names.
        """
        if isinstance(self._obj, Dataset):
            variables = self._obj._variables
        elif isinstance(self._obj, DataArray):
            variables = self._obj._coords

        vardict: dict[str, list[Hashable]] = {}
        for k, v in variables.items():
            if "standard_name" in v.attrs:
                std_name = v.attrs["standard_name"]
                vardict[std_name] = vardict.setdefault(std_name, []) + [k]

        return {std: sort_maybe_hashable(v) for std, v in vardict.items()}

    @property
    def cf_roles(self) -> dict[str, list[Hashable]]:
        """
        Returns a dictionary mapping cf_role names to variable names.

        Returns
        -------
        dict
            Dictionary mapping cf_role names to variable names.

        References
        ----------
        Please refer to the CF conventions document : http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#coordinates-metadata

        Examples
        --------
        >>> import cf_xarray
        >>> from cf_xarray.datasets import dsg
        >>> dsg.cf.cf_roles
        {'profile_id': ['profile'], 'trajectory_id': ['trajectory']}
        """
        if isinstance(self._obj, Dataset):
            variables = self._obj._variables
        elif isinstance(self._obj, DataArray):
            variables = self._obj._coords

        vardict: dict[str, list[Hashable]] = {}
        for k, v in variables.items():
            if "cf_role" in v.attrs:
                role = v.attrs["cf_role"]
                vardict[role] = vardict.setdefault(role, []) + [k]

        return {role_: sort_maybe_hashable(v) for role_, v in vardict.items()}

    def get_associated_variable_names(
        self, name: Hashable, skip_bounds: bool = False, error: bool = True
    ) -> dict[str, list[Hashable]]:
        """
        Returns a dict mapping
            1. "ancillary_variables"
            2. "bounds"
            3. "cell_measures"
            4. "coordinates"
            5. "grid_mapping"
            6. "grid"
        to a list of variable names referred to in the appropriate attribute

        Parameters
        ----------
        name : Hashable
        skip_bounds : bool, optional
        error : bool, optional
            Raise or ignore errors.

        Returns
        -------
        names : dict
            Dictionary with keys "ancillary_variables", "cell_measures", "coordinates", "bounds",
            "grid_mapping", "grid".
        """
        keys = [
            "ancillary_variables",
            "cell_measures",
            "coordinates",
            "bounds",
            "grid_mapping",
            "grid",
        ]

        coords: dict[str, list[Hashable]] = {k: [] for k in keys}
        attrs_or_encoding = ChainMap(self._obj[name].attrs, self._obj[name].encoding)

        coordinates = attrs_or_encoding.get("coordinates", None)
        # Handles case where the coordinates attribute is None
        # This is used to tell xarray to not write a coordinates attribute
        if coordinates:
            coords["coordinates"] = coordinates.split(" ")

        if "cell_measures" in attrs_or_encoding:
            try:
                coords["cell_measures"] = list(
                    parse_cell_methods_attr(attrs_or_encoding["cell_measures"]).values()
                )
            except ValueError as e:
                if error:
                    msg = e.args[0] + " Ignore this error by passing 'error=False'"
                    raise ValueError(msg)
                else:
                    warnings.warn(
                        f"Ignoring bad cell_measures attribute: {attrs_or_encoding['cell_measures']}",
                        UserWarning,
                    )
                    coords["cell_measures"] = []

        if (
            isinstance(self._obj, Dataset)
            and "ancillary_variables" in attrs_or_encoding
        ):
            coords["ancillary_variables"] = attrs_or_encoding[
                "ancillary_variables"
            ].split(" ")

        if not skip_bounds:
            if "bounds" in attrs_or_encoding:
                coords["bounds"] = [attrs_or_encoding["bounds"]]
            for dim in self._obj[name].dims:
                dbounds = self._obj[dim].attrs.get("bounds", None)
                if dbounds:
                    coords["bounds"].append(dbounds)

        if "grid" in attrs_or_encoding:
            coords["grid"] = [attrs_or_encoding["grid"]]

        if "grid_mapping" in attrs_or_encoding:
            coords["grid_mapping"] = [attrs_or_encoding["grid_mapping"]]

        allvars = itertools.chain(*coords.values())
        missing = set(allvars) - set(self._maybe_to_dataset()._variables)
        if missing:
            if OPTIONS["warn_on_missing_variables"]:
                warnings.warn(
                    f"Variables {missing!r} not found in object but are referred to in the CF attributes.",
                    UserWarning,
                )
            for k, v in coords.items():
                for m in missing:
                    if m in v:
                        v.remove(m)
                        coords[k] = v

        return coords

    def _maybe_to_dataset(self, obj=None) -> Dataset:
        if obj is None:
            obj = self._obj
        if isinstance(self._obj, DataArray):
            return obj._to_temp_dataset()
        else:
            return obj

    def _maybe_to_dataarray(self, obj=None):
        if obj is None:
            obj = self._obj
        if isinstance(self._obj, DataArray):
            return self._obj._from_temp_dataset(obj)
        else:
            return obj

    def rename_like(
        self,
        other: DataArray | Dataset,
        skip: str | Iterable[str] | None = None,
    ) -> DataArray | Dataset:
        """
        Renames variables in object to match names of like-variables in ``other``.

        "Likeness" is determined by variables sharing similar attributes. If
        cf_xarray can identify a single "longitude" variable in both this object and
        ``other``, that variable will be renamed to match the "longitude" variable in
        ``other``.

        For now, this function only matches ``("latitude", "longitude", "vertical", "time")``

        Parameters
        ----------
        other : DataArray, Dataset
            Variables will be renamed to match variable names in this xarray object
        skip : str, Iterable[str], optional
            Limit the renaming excluding
            ("axes", "bounds", cell_measures", "coordinates", "standard_names")
            or a subset thereof.

        Returns
        -------
        DataArray or Dataset
            with renamed variables
        """
        if skip is None:
            skip_iter = []
        elif isinstance(skip, str):
            skip_iter = [skip]
        else:
            skip_iter = list(skip)

        ourkeys = self.keys()
        theirkeys = other.cf.keys()

        good_keys = ourkeys & theirkeys
        keydict = {}
        for key in good_keys:
            ours = set(apply_mapper(_get_all, self._obj, key))
            theirs = set(apply_mapper(_get_all, other, key))
            for attr in skip_iter:
                ours.difference_update(getattr(self, attr).get(key, []))
                theirs.difference_update(getattr(other.cf, attr).get(key, []))
            if ours and theirs:
                keydict[key] = dict(ours=list(ours), theirs=list(theirs))

        def get_renamer_and_conflicts(keydict):
            conflicts = {}
            for k0, v0 in keydict.items():
                if len(v0["ours"]) > 1 or len(v0["theirs"]) > 1:
                    conflicts[k0] = v0
                    continue
                for v1 in keydict.values():
                    # Conflicts have same ours but different theirs or vice versa
                    if (v0["ours"] == v1["ours"]) != (v0["theirs"] == v1["theirs"]):
                        conflicts[k0] = v0
                        break

            renamer = {
                v["ours"][0]: v["theirs"][0]
                for k, v in keydict.items()
                if k not in conflicts and v["ours"][0] != v["theirs"][0]
            }

            return renamer, conflicts

        # Run get_renamer_and_conflicts twice.
        # The second time add the bounds associated with variables to rename
        renamer, conflicts = get_renamer_and_conflicts(keydict)
        if "bounds" not in skip_iter:
            for k, v in renamer.items():
                ours = set(getattr(self, "bounds", {}).get(k, []))
                theirs = set(getattr(other.cf, "bounds", {}).get(v, []))
                if ours and theirs:
                    ours.update(keydict.get(k, {}).get("ours", []))
                    theirs.update(keydict.get(k, {}).get("theirs", []))
                    keydict[k] = dict(ours=list(ours), theirs=list(theirs))
            renamer, conflicts = get_renamer_and_conflicts(keydict)

        # Rename and warn
        if conflicts:
            warnings.warn(
                "Conflicting variables skipped:\n"
                + "\n".join(
                    [
                        f"{sorted(v['ours'])}: {sorted(v['theirs'])} ({k})"
                        for k, v in sorted(
                            conflicts.items(), key=lambda item: sorted(item[1]["ours"])
                        )
                    ]
                ),
                UserWarning,
            )
        newobj = self._obj.rename(renamer)

        # rename variable names in the attributes
        # if present
        ds = self._maybe_to_dataset(newobj)
        for _, variable in ds.variables.items():
            for attr in ("bounds", "coordinates", "cell_measures"):
                if attr == "cell_measures":
                    varlist = [
                        f"{k}: {renamer.get(v, v)}"
                        for k, v in parse_cell_methods_attr(
                            variable.attrs.get(attr, "")
                        ).items()
                    ]
                else:
                    varlist = [
                        renamer.get(var, var)
                        for var in variable.attrs.get(attr, "").split()
                    ]

                if varlist:
                    variable.attrs[attr] = " ".join(varlist)
        return self._maybe_to_dataarray(ds)

    def guess_coord_axis(self, verbose: bool = False) -> DataArray | Dataset:
        """
        Automagically guesses X, Y, Z, T, latitude, longitude, and adds
        appropriate attributes. Uses regexes from Metpy and inspired by Iris
        function of same name.

        Existing attributes will not be modified.

        Parameters
        ----------
        verbose : bool
            Print extra info to screen

        Returns
        -------
        DataArray or Dataset
            with appropriate attributes added
        """
        obj = self._obj.copy(deep=False)
        for var in obj.coords.variables:
            var_is_coord = any(var in val for val in obj.cf.coordinates.values())
            if not var_is_coord and obj[var].ndim == 1 and _is_datetime_like(obj[var]):
                if verbose:
                    print(
                        f"I think {var!r} is of type 'time'. It has a datetime-like type."
                    )
                obj[var].attrs = dict(ChainMap(obj[var].attrs, ATTRS["time"]))
                continue  # prevent second detection

            for name, pattern in regex.items():
                var_is_axis = any(var in val for val in obj.cf.axes.values())
                var_is_coord = any(var in val for val in obj.cf.coordinates.values())
                if (name in _AXIS_NAMES and var_is_axis) or (
                    name in _COORD_NAMES and var_is_coord
                ):
                    continue  # skip known axes/coordinates and prevent multiple guesses
                # match variable names
                if pattern.match(var.lower()):
                    if verbose:
                        print(
                            f"I think {var!r} is of type {name!r}. It matched {pattern!r}"
                        )
                    obj[var].attrs = dict(ChainMap(obj[var].attrs, ATTRS[name]))
        return obj

    def drop(self, *args, **kwargs):
        raise NotImplementedError(
            "cf-xarray does not support .drop."
            "Please use .cf.drop_vars or .cf.drop_sel as appropriate."
        )

    def stack(self, dimensions=None, **dimensions_kwargs):
        # stack needs to rewrite the _values_ of a dict
        # our other machinery rewrites the _keys_ of a dict
        # This seems somewhat rare, so do it explicitly for now

        if dimensions is None:
            dimensions = dimensions_kwargs
        for key, values in dimensions.items():
            updates = [
                apply_mapper(
                    (_single(_get_dims),), self._obj, v, error=True, default=[v]
                )
                for v in values
            ]
            dimensions.update({key: tuple(itertools.chain(*updates))})
        return self._obj.stack(dimensions)

    def differentiate(
        self, coord, *xr_args, positive_upward: bool = False, **xr_kwargs
    ):
        """
        Differentiate an xarray object.

        Parameters
        ----------
        positive_upward : optional, bool
            Change sign of the derivative based on the ``"positive"`` attribute of ``coord``
            so that positive values indicate increasing upward.
            If ``positive=="down"``, then multiplied by -1.

        Notes
        -----
        ``xr_args``, ``xr_kwargs`` are passed directly to the underlying xarray function.

        See Also
        --------
        DataArray.cf.differentiate
        Dataset.cf.differentiate
        xarray.DataArray.differentiate : underlying xarray function
        xarray.Dataset.differentiate : underlying xarray function
        """
        coord = apply_mapper(
            (_single(_get_coords),), self._obj, coord, error=False, default=[coord]
        )[0]
        result = self._obj.differentiate(coord, *xr_args, **xr_kwargs)
        if positive_upward:
            coord = self._obj[coord]
            attrs = coord.attrs
            if "positive" not in attrs:
                raise ValueError(
                    f"positive_upward=True and 'positive' attribute not present on {coord.name}"
                )
            if attrs["positive"] not in ["up", "down"]:
                raise ValueError(
                    f"positive_upward=True and received attrs['positive']={attrs['positive']}. Expected one of ['up', 'down'] "
                )
            if attrs["positive"] == "down":
                result *= -1
        return result

    def add_canonical_attributes(
        self,
        override: bool = False,
        skip: Hashable | Iterable[Hashable] | None = None,
        verbose: bool = False,
        source=None,
    ) -> Dataset | DataArray:
        """
        Add canonical CF attributes to variables with standard names.
        Attributes are parsed from the official CF standard name table [1]_.
        This function adds an entry to the "history" attribute.

        Parameters
        ----------
        override : bool
            Override existing attributes.
        skip : str, iterable, optional
            Attribute(s) to skip: ``{"units", "grib", "amip", "description"}``.
        verbose : bool
            Print added attributes to screen.
        source : optional
            Path of `cf-standard-name-table.xml` or file object containing XML data.
            If ``None``, use the default version associated with ``cf-xarray``.

        Returns
        -------
        DataArray or Dataset with attributes added.

        Notes
        -----
        The ``"units"`` attribute is never added to datetime-like variables.

        References
        ----------
        .. [1] https://cfconventions.org/standard-names.html
        """

        # Arguments to add to history
        args = ", ".join([f"{k!s}={v!r}" for k, v in locals().items() if k != "self"])

        # Defaults
        skip_ = [skip] if isinstance(skip, Hashable) else (skip or [])

        # Parse table
        info, table, aliases = parse_cf_standard_name_table(source)

        # Loop over standard names
        ds = self._maybe_to_dataset().copy(deep=False)
        attrs_to_print: dict = {}
        for std_name, var_names in ds.cf.standard_names.items():

            # Loop over variable names
            for var_name in var_names:
                old_attrs = ds[var_name].attrs
                std_name = aliases.get(std_name, std_name)
                new_attrs = table.get(std_name, {})

                # Loop over attributes
                for key, value in new_attrs.items():
                    if (
                        value
                        and key not in skip_
                        and (override or key not in old_attrs)
                    ):

                        # Don't add units to time variables (e.g., datetime64, ...)
                        if key == "units" and _is_datetime_like(ds[var_name]):
                            continue

                        # Add attribute
                        ds[var_name].attrs[key] = value

                        # Build verbose dictionary
                        if verbose:
                            attrs_to_print.setdefault(var_name, {})
                            attrs_to_print[var_name][key] = value

        if verbose:
            # Info
            strings = ["CF Standard Name Table info:"]
            for key, value in info.items():
                strings.append(f"- {key}: {value}")

            # Attributes added
            strings.append("\nAttributes added:")
            for varname, attrs in attrs_to_print.items():
                strings.append(f"- {varname}:")
                for key, value in attrs.items():
                    strings.append(f"    * {key}: {value}")
                strings.append("")

            print("\n".join(strings))

        # Prepend history
        now = datetime.now().ctime()
        method_name = inspect.stack()[0][3]
        version = _get_version()
        table_version = info["version_number"]
        history = (
            f"{now}:"
            f" cf.{method_name}({args})"
            f" [cf-xarray {version}, cf-standard-name-table {table_version}]\n"
        )
        obj = self._maybe_to_dataarray(ds)
        obj.attrs["history"] = history + obj.attrs.get("history", "")

        return obj


@xr.register_dataset_accessor("cf")
class CFDatasetAccessor(CFAccessor):
    def __getitem__(self, key: Hashable | Iterable[Hashable]) -> DataArray | Dataset:
        """
        Index into a Dataset making use of CF attributes.

        Parameters
        ----------
        key: str, Iterable[str], optional
            One of
              - axes names: "X", "Y", "Z", "T"
              - coordinate names: "longitude", "latitude", "vertical", "time"
              - cell measures: "area", "volume", or other names present in the \
                             ``cell_measures`` attribute
              - standard names: names present in ``standard_name`` attribute
              - cf roles: 'timeseries_id', 'profile_id', 'trajectory_id', 'mesh_topology', 'grid_topology'
              - grid mappings: 'grid_mapping' or a grid_mapping_name like 'rotated_latitude_longitude'

        Returns
        -------
        DataArray or Dataset
          ``Dataset.cf[str]`` will return a DataArray, \
          ``Dataset.cf[List[str]]``` will return a Dataset.

        Notes
        -----
        In all cases, associated CF variables will be attached as coordinate variables
        by parsing attributes such as ``bounds``, ``ancillary_variables``, etc.

        ``bounds`` variables will not be attached when a DataArray is returned. This
        is a limitation of the xarray data model.

        Add additional keys by specifying "custom criteria". See :ref:`custom_criteria` for more.
        """
        return _getitem(self, key)

    @property
    def formula_terms(self) -> dict[Hashable, dict[str, str]]:
        """
        Property that returns a dictionary mapping the parametric coordinate's name
        to a dictionary that maps "standard term names" to actual variable names.

        Returns
        -------
        dict
            Dictionary of the form ``{parametric_coord_name: {standard_term_name: variable_name}}``

        References
        ----------
        Please refer to the CF conventions document :
          1. http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-vertical-coordinate
          2. http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord.

        Examples
        --------
        >>> import cf_xarray
        >>> from cf_xarray.datasets import romsds

        The ``s_rho`` DataArray is an example of a parametric vertical coordinate.

        >>> romsds.s_rho
        <xarray.DataArray 's_rho' (s_rho: 30)>
        array([-0.983333, -0.95    , -0.916667, -0.883333, -0.85    , -0.816667,
               -0.783333, -0.75    , -0.716667, -0.683333, -0.65    , -0.616667,
               -0.583333, -0.55    , -0.516667, -0.483333, -0.45    , -0.416667,
               -0.383333, -0.35    , -0.316667, -0.283333, -0.25    , -0.216667,
               -0.183333, -0.15    , -0.116667, -0.083333, -0.05    , -0.016667])
        Coordinates:
          * s_rho       (s_rho) float64 -0.9833 -0.95 -0.9167 ... -0.05 -0.01667
            hc          float64 20.0
            h           float64 603.9
            Vtransform  float64 2.0
            Cs_r        (s_rho) float64 -0.933 -0.8092 -0.6988 ... -0.0005206 -5.758e-05
        Attributes:
            long_name:      S-coordinate at RHO-points
            valid_min:      -1.0
            valid_max:      0.0
            standard_name:  ocean_s_coordinate_g2
            formula_terms:  s: s_rho C: Cs_r eta: zeta depth: h depth_c: hc
            field:          s_rho, scalar

        Now access the formula terms

        >>> romsds.cf.formula_terms
        {'s_rho': {'s': 's_rho', 'C': 'Cs_r', 'eta': 'zeta', 'depth': 'h', 'depth_c': 'hc'}}
        """
        results = {}
        for dim in _get_dims(self._obj, "Z"):
            terms = self._obj[dim].cf.formula_terms
            variables = self._drop_missing_variables(list(terms.values()))
            terms_dict: dict[str, str] = {
                key: val for key, val in terms.items() if val in variables
            }
            if terms_dict:
                results[dim] = terms_dict

        return results

    @property
    def bounds(self) -> dict[Hashable, list[Hashable]]:
        """
        Property that returns a dictionary mapping keys
        to the variable names of their bounds.

        Returns
        -------
        dict
            Dictionary mapping keys to the variable names of their bounds.

        See Also
        --------
        Dataset.cf.get_bounds_dim_name

        Examples
        --------
        >>> from cf_xarray.datasets import mollwds
        >>> mollwds.cf.bounds
        {'lat': ['lat_bounds'], 'latitude': ['lat_bounds'], 'lon': ['lon_bounds'], 'longitude': ['lon_bounds']}
        """

        obj = self._obj
        keys = self.keys() | set(obj.variables)

        vardict = {
            key: self._drop_missing_variables(
                apply_mapper(_get_bounds, obj, key, error=False)
            )
            for key in keys
        }

        return {k: sort_maybe_hashable(v) for k, v in vardict.items() if v}

    def get_bounds(self, key: Hashable) -> DataArray | Dataset:
        """
        Get bounds variable corresponding to key.

        Parameters
        ----------
        key : str
            Name of variable whose bounds are desired

        Returns
        -------
        DataArray
        """

        results = self.bounds.get(key, [])
        if not results:
            raise KeyError(f"No results found for {key!r}.")

        return self._obj[results[0] if len(results) == 1 else results]

    def get_bounds_dim_name(self, key: Hashable) -> Hashable:
        """
        Get bounds dim name for variable corresponding to key.

        Parameters
        ----------
        key : str
            Name of variable whose bounds dimension name is desired.

        Returns
        -------
        str
        """
        crd = self[key]
        bounds = self.get_bounds(key)
        bounds_dims = set(bounds.dims) - set(crd.dims)
        assert len(bounds_dims) == 1
        bounds_dim = bounds_dims.pop()
        assert self._obj.sizes[bounds_dim] in [2, 4]
        return bounds_dim

    def add_bounds(
        self,
        keys: str | Iterable[str],
        *,
        dim: str | Iterable[str] | None = None,
        output_dim: str = "bounds",
    ):
        """
        Returns a new object with bounds variables. The bounds values are guessed assuming
        equal spacing on either side of a coordinate label. The linear estimation is only a
        coarse approximation, especially 2D bounds on curvilinear grids. It is always better to use
        bounds generated as part of the grid creation process. This method is purely for convenience.

        Parameters
        ----------
        keys : str or Iterable[str]
            Either a single variable name or a list of variable names.
        dim : str or Iterable[str], optional
            Core dimension(s) along which to estimate bounds. For 2D bounds, it can
            be a list of 2 dimension names.
        output_dim : str
            The name of the bounds dimension to add.

        Returns
        -------
        DataArray or Dataset
            with bounds variables added and appropriate "bounds" attribute set.

        Raises
        ------
        KeyError

        Notes
        -----
        The bounds variables are automatically named ``f"{var}_bounds"`` where ``var``
        is a variable name.

        Examples
        --------
        >>> from cf_xarray.datasets import airds
        >>> airds.cf.bounds
        {}
        >>> updated = airds.cf.add_bounds("time")
        >>> updated.cf.bounds
        {'T': ['time_bounds'], 'time': ['time_bounds']}
        """
        if isinstance(keys, str):
            keys = [keys]

        variables = set()
        for key in keys:
            variables.update(
                apply_mapper(_get_all, self._obj, key, error=False, default=[key])
            )

        obj = self._maybe_to_dataset(self._obj.copy(deep=False))

        bad_vars: set[str] = variables - set(obj.variables)
        if bad_vars:
            msg = f"{bad_vars!r} are not variables in the underlying object."
            dims_no_idx = bad_vars.intersection(obj.dims)
            if dims_no_idx:
                msg += f" {dims_no_idx!r} are dimensions with no index."
            raise ValueError(msg)

        for var in variables:
            bname = f"{var}_bounds"
            if bname in obj.variables:
                raise ValueError(f"Bounds variable name {bname!r} will conflict!")
            out = _guess_bounds(
                obj[var].reset_coords(drop=True), dim=dim, out_dim=output_dim
            )
            if output_dim in obj.dims and (new := out[output_dim].size) != (
                old := obj[output_dim].size
            ):
                raise ValueError(
                    f"The `{output_dim}` dimension already exists but has a different length than the new one "
                    f"({old} vs {new}). Please provide another bound dimension name with `output_dim`."
                )
            obj.coords[bname] = out
            obj[var].attrs["bounds"] = bname

        return self._maybe_to_dataarray(obj)

    def bounds_to_vertices(
        self,
        keys: Hashable | Iterable[Hashable] | None = None,
        order: str | None = "counterclockwise",
    ) -> Dataset:
        """
        Convert bounds variable to vertices.

        There 2 covered cases:
         - 1D coordinates, with bounds of shape (N, 2),
           converted to vertices of shape (N+1,)
         - 2D coordinates, with bounds of shape (N, M, 4).
           converted to vertices of shape (N+1, M+1).

        Parameters
        ----------
        keys : str or Iterable[str], optional
            The names of the variables whose bounds are to be converted to vertices.
            If not given, converts all available bounds within self.cf.keys().
        order : {'counterclockwise', 'clockwise', None}
            Valid for 2D coordinates only (bounds of shape (N, M, 4), ignored otherwise.
            Order the bounds are given in, assuming that ax0-ax1-upward is a right
            handed coordinate system, where ax0 and ax1 are the two first dimensions of
            the variable. If None, the counterclockwise version is computed and then
            verified. If the check fails the clockwise version is returned.

        Returns
        -------
        Dataset
            Copy of the dataset with added vertices variables.
            Either of shape (N+1,) or (N+1, M+1). New vertex dimensions are named
            from the initial dimension and suffix "_vertices". Variables with similar
            names are overwritten.

        Raises
        ------
        ValueError
            If any of the keys given doesn't corresponds to existing bounds.

        Notes
        -----
        Getting the correct axes "order" is tricky. There are no real standards for
        dimension names or even axes order, even though the CF conventions mentions the
        ax0-ax1-upward (counterclockwise bounds) as being the default. Moreover, xarray can
        tranpose data without raising any warning or error, which make attributes
        unreliable.

        References
        ----------
        Please refer to the CF conventions document : http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-boundaries.
        """
        if keys is None:
            coords = tuple(self.keys())
        elif isinstance(keys, Hashable):
            coords = (keys,)
        else:
            coords = tuple(keys)

        obj = self._maybe_to_dataset(self._obj.copy(deep=False))

        for coord in coords:
            try:
                bounds = self.get_bounds(coord)
            except KeyError as exc:
                if keys is not None:
                    raise ValueError(
                        f"vertices are computed from bounds but given key {coord} did not correspond to existing bounds."
                    ) from exc
            else:
                assert isinstance(bounds, DataArray)
                name = f"{self[coord].name}_vertices"
                obj = obj.assign(  # Overwrite any variable with the same name.
                    {
                        name: bounds_to_vertices(
                            bounds,
                            bounds_dim=list(set(bounds.dims) - set(self[coord].dims))[
                                0
                            ],
                            order=order,
                        )
                    }
                )
        return obj

    @property
    def grid_mapping_names(self) -> dict[str, list[str]]:
        """
        Property that returns a dictionary mapping the CF grid mapping name
        to the variable name containing the grid mapping attributes.

        Returns
        -------
        dict
            Dictionary mapping the CF grid mapping name to the grid mapping variable name.

        See Also
        --------
        DataArray.cf.grid_mapping

        References
        ----------
        Please refer to the CF conventions document : https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#grid-mappings-and-projections

        For a list of valid grid_mapping names, refer to: https://cfconventions.org/Data/cf-conventions/cf-conventions-1.10/cf-conventions.html#appendix-grid-mappings

        Examples
        --------
        >>> from cf_xarray.datasets import rotds
        >>> rotds.cf.grid_mapping_names
        {'rotated_latitude_longitude': ['rotated_pole']}
        """

        obj = self._obj
        keys = set(obj.variables)

        vardict = {
            key: obj.variables[key].attrs["grid_mapping_name"]
            for key in keys
            if "grid_mapping_name" in obj.variables[key].attrs
        }

        results = {}
        for k, v in vardict.items():
            if v not in results:
                results[v] = [k]
            else:
                results[v].append(k)
        return results

    def decode_vertical_coords(self, *, outnames=None, prefix=None):
        """
        Decode parameterized vertical coordinates in place.

        Parameters
        ----------
        outnames : dict, optional
            Keys of outnames are the input sigma/s coordinate variable name and
            the values are the name to use for the associated vertical coordinate.
        prefix : str, optional
            Prefix for newly created z variables.
            E.g. ``s_rho`` becomes ``z_rho``

        Returns
        -------
        None

        Notes
        -----
        Will only decode when the ``formula_terms`` and ``standard_name`` attributes
        are set on the parameter (e.g ``s_rho`` )

        Currently only supports ``ocean_s_coordinate_g1``, ``ocean_s_coordinate_g2``,
        and ``ocean_sigma_coordinate``.

        .. warning::
           Very lightly tested. Please double check the results.

        See Also
        --------
        Dataset.cf.formula_terms
        """
        ds = self._obj

        requirements = {
            "ocean_s_coordinate_g1": {"depth_c", "depth", "s", "C", "eta"},
            "ocean_s_coordinate_g2": {"depth_c", "depth", "s", "C", "eta"},
            "ocean_sigma_coordinate": {"sigma", "eta", "depth"},
        }

        allterms = self.formula_terms
        for dim in allterms:
            if prefix is None:
                assert (
                    outnames is not None
                ), "if prefix is None, outnames must be provided"
                # set outnames here
                try:
                    zname = outnames[dim]
                except KeyError:
                    raise KeyError("Your `outnames` need to include a key of `dim`.")

            else:
                warnings.warn(
                    "`prefix` is being deprecated; use `outnames` instead.",
                    DeprecationWarning,
                )
                suffix = dim.split("_")
                zname = f"{prefix}_" + "_".join(suffix[1:])

            if "standard_name" not in ds[dim].attrs:
                continue
            stdname = ds[dim].attrs["standard_name"]

            # map "standard" formula term names to actual variable names
            terms = {}
            for key, value in allterms[dim].items():
                if value not in ds:
                    raise KeyError(
                        f"Variable {value!r} is required to decode coordinate for {dim!r}"
                        " but it is absent in the Dataset."
                    )
                terms[key] = ds[value]

            absent_terms = requirements[stdname] - set(terms)
            if absent_terms:
                raise KeyError(f"Required terms {absent_terms} absent in dataset.")

            if stdname == "ocean_s_coordinate_g1":
                # S(k,j,i) = depth_c * s(k) + (depth(j,i) - depth_c) * C(k)
                S = (
                    terms["depth_c"] * terms["s"]
                    + (terms["depth"] - terms["depth_c"]) * terms["C"]
                )

                # z(n,k,j,i) = S(k,j,i) + eta(n,j,i) * (1 + S(k,j,i) / depth(j,i))
                ztemp = S + terms["eta"] * (1 + S / terms["depth"])

            elif stdname == "ocean_s_coordinate_g2":
                # make sure all necessary terms are present in terms
                # (depth_c * s(k) + depth(j,i) * C(k)) / (depth_c + depth(j,i))
                S = (terms["depth_c"] * terms["s"] + terms["depth"] * terms["C"]) / (
                    terms["depth_c"] + terms["depth"]
                )

                # z(n,k,j,i) = eta(n,j,i) + (eta(n,j,i) + depth(j,i)) * S(k,j,i)
                ztemp = terms["eta"] + (terms["eta"] + terms["depth"]) * S

            elif stdname == "ocean_sigma_coordinate":
                # z(n,k,j,i) = eta(n,j,i) + sigma(k)*(depth(j,i)+eta(n,j,i))
                ztemp = terms["eta"] + terms["sigma"] * (terms["depth"] + terms["eta"])

            else:
                raise NotImplementedError(
                    f"Coordinate function for {stdname!r} not implemented yet. Contributions welcome!"
                )

            ds.coords[zname] = ztemp


@xr.register_dataarray_accessor("cf")
class CFDataArrayAccessor(CFAccessor):
    @property
    def formula_terms(self) -> dict[str, str]:
        """
        Property that returns a dictionary mapping the parametric coordinate's name
        to a dictionary that maps "standard term names" to actual variable names.

        Returns
        -------
        dict
            Dictionary of the form ``{parametric_coord_name: {standard_term_name: variable_name}}``

        References
        ----------
        Please refer to the CF conventions document :
          1. http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-vertical-coordinate
          2. http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#parametric-v-coord.

        Examples
        --------
        >>> import cf_xarray
        >>> from cf_xarray.datasets import romsds

        The ``s_rho`` DataArray is an example of a parametric vertical coordinate.

        >>> romsds.s_rho
        <xarray.DataArray 's_rho' (s_rho: 30)>
        array([-0.983333, -0.95    , -0.916667, -0.883333, -0.85    , -0.816667,
               -0.783333, -0.75    , -0.716667, -0.683333, -0.65    , -0.616667,
               -0.583333, -0.55    , -0.516667, -0.483333, -0.45    , -0.416667,
               -0.383333, -0.35    , -0.316667, -0.283333, -0.25    , -0.216667,
               -0.183333, -0.15    , -0.116667, -0.083333, -0.05    , -0.016667])
        Coordinates:
          * s_rho       (s_rho) float64 -0.9833 -0.95 -0.9167 ... -0.05 -0.01667
            hc          float64 20.0
            h           float64 603.9
            Vtransform  float64 2.0
            Cs_r        (s_rho) float64 -0.933 -0.8092 -0.6988 ... -0.0005206 -5.758e-05
        Attributes:
            long_name:      S-coordinate at RHO-points
            valid_min:      -1.0
            valid_max:      0.0
            standard_name:  ocean_s_coordinate_g2
            formula_terms:  s: s_rho C: Cs_r eta: zeta depth: h depth_c: hc
            field:          s_rho, scalar

        Now access the formula terms

        >>> romsds.s_rho.cf.formula_terms
        {'s': 's_rho', 'C': 'Cs_r', 'eta': 'zeta', 'depth': 'h', 'depth_c': 'hc'}
        """
        da = self._obj
        if "formula_terms" not in ChainMap(da.attrs, da.encoding):
            var = da[_single(_get_dims)(da, "Z")[0]]
        else:
            var = da

        terms = {}
        formula_terms = ChainMap(var.attrs, var.encoding).get("formula_terms", "")
        for mapping in re.sub(r"\s*:\s*", ":", formula_terms).split():
            key, value = mapping.split(":")
            terms[key] = value
        return terms

    @property
    def grid_mapping_name(self) -> str:
        """
        Get CF grid mapping name associated with this variable.

        Parameters
        ----------
        key : str
            Name of variable whose grid_mapping name is desired.

        Returns
        -------
        str
            CF Name of the associated grid mapping.

        See Also
        --------
        Dataset.cf.grid_mapping_names

        Examples
        --------
        >>> from cf_xarray.datasets import rotds
        >>> rotds.cf["temp"].cf.grid_mapping_name
        'rotated_latitude_longitude'

        """

        da = self._obj

        attrs_or_encoding = ChainMap(da.attrs, da.encoding)
        grid_mapping = attrs_or_encoding.get("grid_mapping", None)
        if not grid_mapping:
            raise ValueError("No 'grid_mapping' attribute present.")

        grid_mapping_var = da[grid_mapping]
        return grid_mapping_var.attrs["grid_mapping_name"]

    def __getitem__(self, key: Hashable | Iterable[Hashable]) -> DataArray:
        """
        Index into a DataArray making use of CF attributes.

        Parameters
        ----------
        key: str, Iterable[str], optional
            One of
              - axes names: "X", "Y", "Z", "T"
              - coordinate names: "longitude", "latitude", "vertical", "time"
              - cell measures: "area", "volume", or other names present in the \
                             ``cell_measures`` attribute
              - standard names: names present in ``standard_name`` attribute of \
                coordinate variables
              - cf roles: 'timeseries_id', 'profile_id', 'trajectory_id', 'mesh_topology', 'grid_topology'
              - grid mappings: 'grid_mapping' or a grid_mapping_name like 'rotated_latitude_longitude'

        Returns
        -------
        DataArray

        Raises
        ------
        KeyError
          ``DataArray.cf[List[str]]`` will raise KeyError.

        Notes
        -----
        Associated CF variables will be attached as coordinate variables
        by parsing attributes such as ``cell_measures``, ``coordinates`` etc.

        Add additional keys by specifying "custom criteria". See :ref:`custom_criteria` for more.
        """

        if not isinstance(key, Hashable):
            raise KeyError(
                f"Cannot use an Iterable of keys with DataArrays. Expected a single string. Received {key!r} instead."
            )

        return _getitem(self, key)

    @property
    def is_flag_variable(self) -> bool:
        """
        Returns True if the DataArray satisfies CF conventions for flag variables.

        .. warning::
          Flag masks are not supported yet.

        Returns
        -------
        bool
        """
        if (
            isinstance(self._obj, DataArray)
            and "flag_meanings" in self._obj.attrs
            and "flag_values" in self._obj.attrs
        ):
            return True
        else:
            return False
