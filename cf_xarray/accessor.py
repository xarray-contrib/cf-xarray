import functools
import inspect
import itertools
import warnings
from collections import ChainMap
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
    cast,
)

import xarray as xr
from xarray import DataArray, Dataset
from xarray.core.arithmetic import SupportsArithmetic

from .helpers import bounds_to_vertices
from .utils import _is_datetime_like, invert_mappings, parse_cell_methods_attr

#: Classes wrapped by cf_xarray.
_WRAPPED_CLASSES = (
    xr.core.resample.Resample,
    xr.core.groupby.GroupBy,
    xr.core.rolling.Rolling,
    xr.core.rolling.Coarsen,
    xr.core.weighted.Weighted,
)

#:  `axis` names understood by cf_xarray
_AXIS_NAMES = ("X", "Y", "Z", "T")

#:  `coordinate` types understood by cf_xarray.
_COORD_NAMES = ("longitude", "latitude", "vertical", "time")

#:  Cell measures understood by cf_xarray.
_CELL_MEASURES = ("area", "volume")

# Define the criteria for coordinate matches
# Copied from metpy
# Internally we only use X, Y, Z, T
coordinate_criteria: MutableMapping[str, MutableMapping[str, Tuple]] = {
    "standard_name": {
        "X": ("projection_x_coordinate",),
        "Y": ("projection_y_coordinate",),
        "T": ("time",),
        "time": ("time",),
        "vertical": (
            "air_pressure",
            "height",
            "depth",
            "geopotential_height",
            # computed dimensional coordinate name
            "altitude",
            "height_above_geopotential_datum",
            "height_above_reference_ellipsoid",
            "height_above_mean_sea_level",
        ),
        "Z": (
            "model_level_number",
            "atmosphere_ln_pressure_coordinate",
            "atmosphere_sigma_coordinate",
            "atmosphere_hybrid_sigma_pressure_coordinate",
            "atmosphere_hybrid_height_coordinate",
            "atmosphere_sleve_coordinate",
            "ocean_sigma_coordinate",
            "ocean_s_coordinate",
            "ocean_s_coordinate_g1",
            "ocean_s_coordinate_g2",
            "ocean_sigma_z_coordinate",
            "ocean_double_sigma_coordinate",
        ),
        "latitude": ("latitude",),
        "longitude": ("longitude",),
    },
    "_CoordinateAxisType": {
        "T": ("Time",),
        "Z": ("GeoZ", "Height", "Pressure"),
        "Y": ("GeoY",),
        "latitude": ("Lat",),
        "X": ("GeoX",),
        "longitude": ("Lon",),
    },
    "axis": {"T": ("T",), "Z": ("Z",), "Y": ("Y",), "X": ("X",)},
    "cartesian_axis": {"T": ("T",), "Z": ("Z",), "Y": ("Y",), "X": ("X",)},
    "positive": {"vertical": ("up", "down")},
    "units": {
        "latitude": (
            "degree_north",
            "degree_N",
            "degreeN",
            "degrees_north",
            "degrees_N",
            "degreesN",
        ),
        "longitude": (
            "degree_east",
            "degree_E",
            "degreeE",
            "degrees_east",
            "degrees_E",
            "degreesE",
        ),
    },
}

# "long_name" and "standard_name" criteria are the same. For convenience.
coordinate_criteria["long_name"] = coordinate_criteria["standard_name"]

#: regular expressions for guess_coord_axis
regex = {
    "time": "\\bt\\b|(time|min|hour|day|week|month|year)[0-9]*",
    "vertical": (
        "(z|nav_lev|gdep|lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|"
        "isobaric|pres|isotherm)[a-z_]*[0-9]*"
    ),
    "Y": "y",
    "latitude": "y?(nav_lat|lat|gphi)[a-z0-9]*",
    "X": "x",
    "longitude": "x?(nav_lon|lon|glam)[a-z0-9]*",
}
regex["Z"] = regex["vertical"]
regex["T"] = regex["time"]


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
Mapper = Callable[[Union[DataArray, Dataset], str], List[str]]

# Type for decorators
F = TypeVar("F", bound=Callable[..., Any])


def apply_mapper(
    mappers: Union[Mapper, Tuple[Mapper, ...]],
    obj: Union[DataArray, Dataset],
    key: str,
    error: bool = True,
    default: Any = None,
) -> List[Any]:
    """
    Applies a mapping function; does error handling / returning defaults.

    Expects the mapper function to raise an error if passed a bad key.
    It should return a list in all other cases including when there are no
    results for a good key.
    """
    if default is None:
        default = []

    def _apply_single_mapper(mapper):

        try:
            results = mapper(obj, key)
        except KeyError as e:
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


def _get_groupby_time_accessor(var: Union[DataArray, Dataset], key: str) -> List[str]:
    """
    Time variable accessor e.g. 'T.month'
    """
    """
    Helper method for when our key name is of the nature "T.month" and we want to
    isolate the "T" for coordinate mapping

    Parameters
    ----------
    var : DataArray, Dataset
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

    if "." in key:
        key, ext = key.split(".", 1)

        results = apply_mapper((_get_all,), var, key, error=False)
        if len(results) > 1:
            raise KeyError(f"Multiple results received for {key}.")
        return [v + "." + ext for v in results]

    else:
        return []


def _get_axis_coord(var: Union[DataArray, Dataset], key: str) -> List[str]:
    """
    Translate from axis or coord name to variable name

    Parameters
    ----------
    var : DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key : str, ["X", "Y", "Z", "T", "longitude", "latitude", "vertical", "time"]
        key to check for.
    error : bool
        raise errors when key is not found or interpretable. Use False and provide default
        to replicate dict.get(k, None).
    default : Any
        default value to return when error is False.

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
    if "coordinates" in var.encoding:
        search_in.update(var.encoding["coordinates"].split(" "))
    if "coordinates" in var.attrs:
        search_in.update(var.attrs["coordinates"].split(" "))
    if not search_in:
        search_in = set(var.coords)

    # maybe only do this for key in _AXIS_NAMES?
    search_in.update(var.indexes)

    results: Set = set()
    for coord in search_in:
        for criterion, valid_values in coordinate_criteria.items():
            if key in valid_values:
                expected = valid_values[key]
                if (
                    coord in var.coords
                    and var.coords[coord].attrs.get(criterion, None) in expected
                ):
                    results.update((coord,))

    return list(results)


def _get_measure(obj: Union[DataArray, Dataset], key: str) -> List[str]:
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
        if "cell_measures" in da.attrs:
            attr = da.attrs["cell_measures"]
            measures = parse_cell_methods_attr(attr)
            if key in measures:
                results.update([measures[key]])

    if isinstance(results, str):
        return [results]
    return list(results)


def _get_with_standard_name(
    obj: Union[DataArray, Dataset], name: Union[str, List[str]]
) -> List[str]:
    """ returns a list of variable names with standard name == name. """
    if name is None:
        return []

    varnames = []
    if isinstance(obj, DataArray):
        obj = obj.coords.to_dataset()
    for vname, var in obj.variables.items():
        stdname = var.attrs.get("standard_name", None)
        if stdname == name:
            varnames.append(str(vname))

    return varnames


def _get_all(obj: Union[DataArray, Dataset], key: str) -> List[str]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names
    """
    all_mappers = (_get_axis_coord, _get_measure, _get_with_standard_name)
    results = apply_mapper(all_mappers, obj, key, error=False, default=None)
    return results


def _get_dims(obj: Union[DataArray, Dataset], key: str) -> List[str]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names present in .dims
    """
    return [k for k in _get_all(obj, key) if k in obj.dims]


def _get_indexes(obj: Union[DataArray, Dataset], key: str) -> List[str]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names present in .indexes
    """
    return [k for k in _get_all(obj, key) if k in obj.indexes]


def _get_coords(obj: Union[DataArray, Dataset], key: str) -> List[str]:
    """
    One or more of ('X', 'Y', 'Z', 'T', 'longitude', 'latitude', 'vertical', 'time',
    'area', 'volume'), or arbitrary measures, or standard names present in .coords
    """
    return [k for k in _get_all(obj, key) if k in obj.coords]


def _variables(func: F) -> F:
    @functools.wraps(func)
    def wrapper(obj: Union[DataArray, Dataset], key: str) -> List[DataArray]:
        return [obj[k] for k in func(obj, key)]

    return cast(F, wrapper)


def _single(func: F) -> F:
    @functools.wraps(func)
    def wrapper(obj: Union[DataArray, Dataset], key: str):
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
_DEFAULT_KEY_MAPPERS: Mapping[str, Tuple[Mapper, ...]] = {
    "dim": (_get_dims,),
    "dims": (_get_dims,),  # transpose
    "drop_dims": (_get_dims,),  # drop_dims
    "dimensions": (_get_dims,),  # stack
    "dims_dict": (_get_dims,),  # swap_dims, rename_dims
    "shifts": (_get_dims,),  # shift, roll
    "pad_width": (_get_dims,),  # shift, roll
    "names": (_get_all,),  # set_coords, reset_coords, drop_vars
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
    "weights": (_variables(_single(_get_all)),),  # type: ignore
    "chunks": (_get_dims,),  # chunk
}


def _guess_bounds_dim(da):
    """
    Guess bounds values given a 1D coordinate variable.
    Assumes equal spacing on either side of the coordinate label.
    """
    assert da.ndim == 1

    dim = da.dims[0]
    diff = da.diff(dim)
    lower = da - diff / 2
    upper = da + diff / 2
    bounds = xr.concat([lower, upper], dim="bounds")

    first = (bounds.isel({dim: 0}) - diff[0]).assign_coords({dim: da[dim][0]})
    result = xr.concat([first, bounds], dim=dim)

    return result


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
    obj: Union[DataArray, Dataset],
    attr: str,
    accessor: "CFAccessor",
    key_mappers: Mapping[str, Mapper],
    wrap_classes: bool = False,
    extra_decorator: Callable = None,
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
    try:
        attribute: Union[Mapping, Callable] = getattr(obj, attr)
    except AttributeError:
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

        skip = {"data_vars": ["coords"], "coords": None}
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

    wrapper.__doc__ = _build_docstring(func) + wrapper.__doc__

    return wrapper


def _getitem(
    accessor: "CFAccessor", key: Union[str, List[str]], skip: List[str] = None
) -> Union[DataArray, Dataset]:
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
    kind = str(type(obj).__name__)
    scalar_key = isinstance(key, str)

    if scalar_key:
        key = (key,)  # type: ignore

    if skip is None:
        skip = []

    def check_results(names, k):
        if scalar_key and len(names) > 1:
            raise KeyError(
                f"Receive multiple variables for key {k!r}: {names}. "
                f"Expected only one. Please pass a list [{k!r}] "
                f"instead to get all variables matching {k!r}."
            )

    varnames: List[Hashable] = []
    coords: List[Hashable] = []
    successful = dict.fromkeys(key, False)
    for k in key:
        if "coords" not in skip and k in _AXIS_NAMES + _COORD_NAMES:
            names = _get_all(obj, k)
            check_results(names, k)
            successful[k] = bool(names)
            coords.extend(names)
        elif "measures" not in skip and k in accessor._get_all_cell_measures():
            measure = _get_all(obj, k)
            check_results(measure, k)
            successful[k] = bool(measure)
            if measure:
                varnames.extend(measure)
        else:
            stdnames = set(_get_with_standard_name(obj, k))
            check_results(stdnames, k)
            successful[k] = bool(stdnames)
            objcoords = set(obj.coords)
            varnames.extend(stdnames - objcoords)
            coords.extend(stdnames & objcoords)

    # these are not special names but could be variable names in underlying object
    # we allow this so that we can return variables with appropriate CF auxiliary variables
    varnames.extend([k for k, v in successful.items() if not v])
    allnames = varnames + coords

    try:
        for name in allnames:
            extravars = accessor.get_associated_variable_names(
                name, skip_bounds=scalar_key
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
                    f"Received scalar key {key[0]!r} but multiple results: {allnames!r}. "
                    f"Please pass a list instead (['{key[0]}']) to get back a Dataset "
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


def _possible_x_y_plot(obj, key):
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
            value = getattr(accessor, attr).get(key)
            if not value or len(value) > 1:
                continue
            if not is_scalar(accessor._obj[value[0]]):
                return value[0]
        return None

    if key == "x":
        return _get_possible(obj.cf, x_criteria)
    elif key == "y":
        return _get_possible(obj.cf, y_criteria)


class _CFWrappedClass(SupportsArithmetic):
    """
    This class is used to wrap any class in _WRAPPED_CLASSES.
    """

    def __init__(self, towrap, accessor: "CFAccessor"):
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

    def _plot_decorator(self, func):
        """
        This decorator is used to set default kwargs on plotting functions.
        For now, this can
        1. set ``xincrease`` and ``yincrease``.
        2. automatically set ``x`` or ``y``.
        """
        valid_keys = self.accessor.keys()

        @functools.wraps(func)
        def _plot_wrapper(*args, **kwargs):
            def _process_x_or_y(kwargs, key):
                if key not in kwargs:
                    kwargs[key] = _possible_x_y_plot(self._obj, key)

                value = kwargs.get(key)
                if value:
                    if value in valid_keys:
                        var = self.accessor[value]
                    else:
                        var = self._obj[value]
                    if "positive" in var.attrs:
                        if var.attrs["positive"] == "down":
                            kwargs.setdefault(f"{key}increase", False)
                        else:
                            kwargs.setdefault(f"{key}increase", True)
                return kwargs

            is_line_plot = (func.__name__ == "line") or (
                func.__name__ == "wrapper"
                and (kwargs.get("hue") or self._obj.ndim == 1)
            )
            if is_line_plot:
                if not kwargs.get("hue"):
                    kwargs = _process_x_or_y(kwargs, "x")
                    if not kwargs.get("x"):
                        kwargs = _process_x_or_y(kwargs, "y")
            else:
                kwargs = _process_x_or_y(kwargs, "x")
                kwargs = _process_x_or_y(kwargs, "y")

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


class CFAccessor:
    """
    Common Dataset and DataArray accessor functionality.
    """

    def __init__(self, da):
        self._obj = da
        self._all_cell_measures = None

    def _get_all_cell_measures(self):
        """
        Get all cell measures defined in the object, adding CF pre-defined measures.
        """

        # get all_cell_measures only once
        if not self._all_cell_measures:
            self._all_cell_measures = set(_CELL_MEASURES + tuple(self.cell_measures))

        return self._all_cell_measures

    def _process_signature(
        self,
        func: Callable,
        args,
        kwargs,
        key_mappers: MutableMapping[str, Tuple[Mapper, ...]],
    ):
        """
        Processes a function's signature, args, kwargs:
        1. Binds *args so that everthing is a Mapping from kwarg name to values
        2. Calls _rewrite_values to rewrite any special CF names to normal xarray names.
           This uses key_mappers
        3. Unpacks arguments if necessary before returning them.
        """
        sig = inspect.signature(func, follow_wrapped=False)

        # Catch things like .isel(T=5).
        # This assigns indexers_kwargs=dict(T=5).
        # and indexers_kwargs is of kind VAR_KEYWORD
        var_kws: List = []
        # capture *args, e.g. transpose
        var_args: List = []
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
        key_mappers: Mapping[str, Tuple[Mapper, ...]],
        var_kws: Tuple[str, ...],
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
        all_mappers = ChainMap(
            key_mappers,
            dict.fromkeys(var_kws, (_get_all,)),
        )

        for key in set(all_mappers) & set(kwargs):
            value = kwargs[key]
            mappers = all_mappers[key]

            if isinstance(value, str):
                value = [value]

            if isinstance(value, dict):
                # this for things like isel where **kwargs captures things like T=5
                # .sel, .isel, .rolling
                # Account for multiple names matching the key.
                # e.g. .isel(X=5) → .isel(xi_rho=5, xi_u=5, xi_v=5, xi_psi=5)
                # where xi_* have attrs["axis"] = "X"
                updates[key] = ChainMap(
                    *[
                        dict.fromkeys(
                            apply_mapper(mappers, self._obj, k, False, [k]), v
                        )
                        for k, v in value.items()
                    ]
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

        coords = self._obj.coords
        dims = self._obj.dims

        def make_text_section(subtitle, vardict, valid_values, default_keys=None):

            star = " * "
            tab = len(star) * " "
            subtitle = f"- {subtitle}:"

            # Sort keys if there aren't extra keys,
            # preserve default keys order otherwise.
            default_keys = [] if not default_keys else list(default_keys)
            extra_keys = list(set(vardict) - set(default_keys))
            ordered_keys = sorted(vardict) if extra_keys else default_keys
            vardict = {key: vardict[key] for key in ordered_keys if key in vardict}

            # Keep only valid values (e.g., coords or data_vars)
            vardict = {
                key: set(value).intersection(valid_values)
                for key, value in vardict.items()
                if set(value).intersection(valid_values)
            }

            # Star for keys with dims only, tab otherwise
            rows = [
                f"{star if set(value) <= set(dims) else tab}{key}: {sorted(value)}"
                for key, value in vardict.items()
            ]

            # Append missing default keys followed by n/a
            if default_keys:
                missing_keys = [key for key in default_keys if key not in vardict]
                if missing_keys:
                    rows += [tab + ", ".join(missing_keys) + ": n/a"]
            elif not rows:
                rows = [tab + "n/a"]

            # Add subtitle to the first row, align other rows
            rows = [
                "\n" + subtitle + row if i == 0 else len(subtitle) * " " + row
                for i, row in enumerate(rows)
            ]

            return "\n".join(rows) + "\n"

        text = "Coordinates:"
        text += make_text_section("CF Axes", self.axes, coords, _AXIS_NAMES)
        text += make_text_section(
            "CF Coordinates", self.coordinates, coords, _COORD_NAMES
        )
        text += make_text_section(
            "Cell Measures", self.cell_measures, coords, _CELL_MEASURES
        )
        text += make_text_section("Standard Names", self.standard_names, coords)
        if isinstance(self._obj, Dataset):
            data_vars = self._obj.data_vars
            text += "\nData Variables:"
            text += make_text_section(
                "Cell Measures", self.cell_measures, data_vars, _CELL_MEASURES
            )
            text += make_text_section("Standard Names", self.standard_names, data_vars)

        return text

    def get_valid_keys(self) -> Set[str]:

        warnings.warn(
            "Now called `keys` and `get_valid_keys` will be removed in a future version.",
            DeprecationWarning,
        )

        return self.keys()

    def keys(self) -> Set[str]:
        """
        Utility function that returns valid keys for .cf[].

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.

        Returns
        -------
        Set of valid key names that can be used with __getitem__ or .cf[key].
        """

        varnames = list(self.axes) + list(self.coordinates)
        varnames.extend(list(self.cell_measures))
        varnames.extend(list(self.standard_names))

        return set(varnames)

    @property
    def axes(self) -> Dict[str, List[str]]:
        """
        Property that returns a dictionary mapping valid Axis standard names for ``.cf[]``
        to variable names.

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.
        However, it will only return the Axis names present in ``.coords``, not Coordinate names.

        Returns
        -------
        Dictionary of valid Axis names that can be used with ``__getitem__`` or ``.cf[key]``.
        Will be ("X", "Y", "Z", "T") or a subset thereof.
        """
        vardict = {key: _get_coords(self._obj, key) for key in _AXIS_NAMES}

        return {k: sorted(v) for k, v in vardict.items() if v}

    @property
    def coordinates(self) -> Dict[str, List[str]]:
        """
        Property that returns a dictionary mapping valid Coordinate standard names for ``.cf[]``
        to variable names.

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.
        However, it will only return the Coordinate names present in ``.coords``, not Axis names.

        Returns
        -------
        Dictionary of valid Coordinate names that can be used with ``__getitem__`` or ``.cf[key]``.
        Will be ("longitude", "latitude", "vertical", "time") or a subset thereof.
        """
        vardict = {key: _get_coords(self._obj, key) for key in _COORD_NAMES}

        return {k: sorted(v) for k, v in vardict.items() if v}

    @property
    def cell_measures(self) -> Dict[str, List[str]]:
        """
        Property that returns a dictionary mapping valid cell measure standard names for ``.cf[]``
        to variable names.

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.

        Returns
        -------
        Dictionary of valid cell measure names that can be used with __getitem__ or .cf[key].
        """

        obj = self._obj
        all_attrs = [da.attrs.get("cell_measures", "") for da in obj.coords.values()]
        if isinstance(obj, DataArray):
            all_attrs += [obj.attrs.get("cell_measures", "")]
        elif isinstance(obj, Dataset):
            all_attrs += [
                da.attrs.get("cell_measures", "") for da in obj.data_vars.values()
            ]

        keys = {}
        for attr in all_attrs:
            keys.update(parse_cell_methods_attr(attr))
        measures = {key: _get_all(self._obj, key) for key in keys}

        return {k: sorted(set(v)) for k, v in measures.items() if v}

    def get_standard_names(self) -> List[str]:

        warnings.warn(
            "`get_standard_names` will be removed in a future version in favor of `standard_names`.",
            DeprecationWarning,
        )

        return list(self.standard_names.keys())

    @property
    def standard_names(self) -> Dict[str, List[str]]:
        """
        Returns a sorted list of standard names in Dataset.

        Parameters
        ----------
        obj : DataArray, Dataset
            Xarray object to process

        Returns
        -------
        Dictionary of standard names in dataset
        """
        if isinstance(self._obj, Dataset):
            variables = self._obj.variables
        elif isinstance(self._obj, DataArray):
            variables = self._obj.coords

        vardict: Dict[str, List[str]] = {}
        for k, v in variables.items():
            if "standard_name" in v.attrs:
                std_name = v.attrs["standard_name"]
                vardict[std_name] = vardict.setdefault(std_name, []) + [k]

        return {k: sorted(v) for k, v in vardict.items()}

    def get_associated_variable_names(
        self, name: Hashable, skip_bounds: Optional[bool] = None
    ) -> Dict[str, List[str]]:
        """
        Returns a dict mapping
            1. "ancillary_variables"
            2. "bounds"
            3. "cell_measures"
            4. "coordinates"
        to a list of variable names referred to in the appropriate attribute

        Parameters
        ----------
        name : Hashable
        skip_bounds : bool, optional
        Returns
        ------
        Dict with keys "ancillary_variables", "cell_measures", "coordinates", "bounds"
        """
        keys = ["ancillary_variables", "cell_measures", "coordinates", "bounds"]
        coords: Dict[str, List[str]] = {k: [] for k in keys}
        attrs_or_encoding = ChainMap(self._obj[name].attrs, self._obj[name].encoding)

        if "coordinates" in attrs_or_encoding:
            coords["coordinates"] = attrs_or_encoding["coordinates"].split(" ")

        if "cell_measures" in attrs_or_encoding:
            coords["cell_measures"] = list(
                parse_cell_methods_attr(attrs_or_encoding["cell_measures"]).values()
            )

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

        allvars = itertools.chain(*coords.values())
        missing = set(allvars) - set(self._maybe_to_dataset().variables)
        if missing:
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
        self, other: Union[DataArray, Dataset]
    ) -> Union[DataArray, Dataset]:
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

        Returns
        -------
        DataArray or Dataset with renamed variables
        """
        ourkeys = self.keys()
        theirkeys = other.cf.keys()

        good_keys = ourkeys & theirkeys
        renamer = {}
        for key in good_keys:
            ours = _single(_get_all)(self._obj, key)[0]
            theirs = _single(_get_all)(other, key)[0]
            renamer[ours] = theirs

        newobj = self._obj.rename(renamer)

        # rename variable names in the coordinates attribute
        # if present
        ds = self._maybe_to_dataset(newobj)
        for _, variable in ds.variables.items():
            coordinates = variable.attrs.get("coordinates", None)
            if coordinates:
                for k, v in renamer.items():
                    coordinates = coordinates.replace(k, v)
                variable.attrs["coordinates"] = coordinates
        return self._maybe_to_dataarray(ds)

    def guess_coord_axis(self, verbose: bool = False) -> Union[DataArray, Dataset]:
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
        DataArray or Dataset with appropriate attributes added
        """
        import re

        obj = self._obj.copy(deep=True)
        for var in obj.coords.variables:
            if obj[var].ndim == 1 and _is_datetime_like(obj[var]):
                if verbose:
                    print(
                        f"I think {var!r} is of type 'time'. It has a datetime-like type."
                    )
                obj[var].attrs = dict(ChainMap(obj[var].attrs, ATTRS["time"]))
                continue  # prevent second detection

            for axis, pattern in regex.items():
                # match variable names
                if re.match(pattern, var.lower()):
                    if verbose:
                        print(
                            f"I think {var!r} is of type {axis!r}. It matched {pattern!r}"
                        )
                    obj[var].attrs = dict(ChainMap(obj[var].attrs, ATTRS[axis]))
        return obj

    def drop(self, *args, **kwargs):
        raise NotImplementedError(
            "cf-xarray does not support .drop."
            "Please use .cf.drop_vars or .cf.drop_sel as appropriate."
        )


@xr.register_dataset_accessor("cf")
class CFDatasetAccessor(CFAccessor):
    def __getitem__(self, key: Union[str, List[str]]) -> Union[DataArray, Dataset]:
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
        """
        return _getitem(self, key)

    def get_bounds(self, key: str) -> DataArray:
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
        name = apply_mapper(
            _single(_get_all), self._obj, key, error=False, default=[key]
        )[0]
        bounds = self._obj[name].attrs["bounds"]
        obj = self._maybe_to_dataset()
        return obj[bounds]

    def get_bounds_dim_name(self, key: str) -> str:
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

    def add_bounds(self, dims: Union[Hashable, Iterable[Hashable]]):
        """
        Returns a new object with bounds variables. The bounds values are guessed assuming
        equal spacing on either side of a coordinate label.

        Parameters
        ----------
        dims : Hashable or Iterable[Hashable]
            Either a single dimension name or a list of dimension names.

        Returns
        -------
        DataArray or Dataset with bounds variables added and appropriate "bounds" attribute set.

        Raises
        ------
        KeyError

        Notes
        -----
        The bounds variables are automatically named f"{dim}_bounds" where ``dim``
        is a dimension name.
        """
        if isinstance(dims, Hashable):
            dimensions = (dims,)
        else:
            dimensions = dims

        bad_dims: Set[Hashable] = set(dimensions) - set(self._obj.dims)
        if bad_dims:
            raise ValueError(
                f"{bad_dims!r} are not dimensions in the underlying object."
            )

        obj = self._maybe_to_dataset(self._obj.copy(deep=True))
        for dim in dimensions:
            bname = f"{dim}_bounds"
            if bname in obj.variables:
                raise ValueError(f"Bounds variable name {bname!r} will conflict!")
            obj.coords[bname] = _guess_bounds_dim(obj[dim].reset_coords(drop=True))
            obj[dim].attrs["bounds"] = bname

        return self._maybe_to_dataarray(obj)

    def bounds_to_vertices(
        self, keys: Union[str, Iterable[str]] = None, order: str = "counterclockwise"
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
            from the intial dimension and suffix "_vertices". Variables with similar
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

        Please refer to the CF conventions document : http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-boundaries.
        """
        if keys is None:
            coords: Iterable[str] = self.keys()
        elif isinstance(keys, str):
            coords = (keys,)
        else:
            coords = keys

        obj = self._maybe_to_dataset(self._obj.copy(deep=True))

        for coord in coords:
            try:
                bounds = self.get_bounds(coord)
            except KeyError as exc:
                if keys is not None:
                    raise ValueError(
                        f"vertices are computed from bounds but given key {coord} did not correspond to existing bounds."
                    ) from exc
            else:
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

    def decode_vertical_coords(self, prefix="z"):
        """
        Decode parameterized vertical coordinates in place.

        Parameters
        ----------
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

        Currently only supports ``ocean_s_coordinate_g1`` and ``ocean_s_coordinate_g2``.

        .. warning::
           Very lightly tested. Please double check the results.
        """
        import re

        ds = self._obj
        dims = _get_dims(ds, "Z")

        requirements = {
            "ocean_s_coordinate_g1": {"depth_c", "depth", "s", "C", "eta"},
            "ocean_s_coordinate_g2": {"depth_c", "depth", "s", "C", "eta"},
        }

        for dim in dims:
            suffix = dim.split("_")
            zname = f"{prefix}_" + "_".join(suffix[1:])

            if (
                "formula_terms" not in ds[dim].attrs
                or "standard_name" not in ds[dim].attrs
            ):
                continue

            formula_terms = ds[dim].attrs["formula_terms"]
            stdname = ds[dim].attrs["standard_name"]

            # map "standard" formula term names to actual variable names
            terms = {}
            for mapping in re.sub(": ", ":", formula_terms).split(" "):
                key, value = mapping.split(":")
                if value not in ds:
                    raise KeyError(
                        f"Variable {value!r} is required to decode coordinate for {dim} but it is absent in the Dataset."
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
                ds.coords[zname] = S + terms["eta"] * (1 + S / terms["depth"])

            elif stdname == "ocean_s_coordinate_g2":
                # make sure all necessary terms are present in terms
                # (depth_c * s(k) + depth(j,i) * C(k)) / (depth_c + depth(j,i))
                S = (terms["depth_c"] * terms["s"] + terms["depth"] * terms["C"]) / (
                    terms["depth_c"] + terms["depth"]
                )
                # z(n,k,j,i) = eta(n,j,i) + (eta(n,j,i) + depth(j,i)) * S(k,j,i)
                ds.coords[zname] = terms["eta"] + (terms["eta"] + terms["depth"]) * S

            else:
                raise NotImplementedError(
                    f"Coordinate function for {stdname} not implemented yet. Contributions welcome!"
                )


@xr.register_dataarray_accessor("cf")
class CFDataArrayAccessor(CFAccessor):
    def __getitem__(self, key: Union[str, List[str]]) -> DataArray:
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
        """

        if not isinstance(key, str):
            raise KeyError(
                f"Cannot use a list of keys with DataArrays. Expected a single string. Received {key!r} instead."
            )

        return _getitem(self, key)

    pass
