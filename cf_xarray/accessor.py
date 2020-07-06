import functools
import inspect
import itertools
import textwrap
from collections import ChainMap
from contextlib import suppress
from typing import (
    Callable,
    Hashable,
    List,
    Mapping,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Union,
)

import xarray as xr
from xarray import DataArray, Dataset

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
        "T": ("time",),
        "time": ("time",),
        "Z": (
            "air_pressure",
            "height",
            "geopotential_height",
            "altitude",
            "model_level_number",
            "atmosphere_ln_pressure_coordinate",
            "atmosphere_sigma_coordinate",
            "atmosphere_hybrid_sigma_pressure_coordinate",
            "atmosphere_hybrid_height_coordinate",
            "atmosphere_sleve_coordinate",
            "height_above_geopotential_datum",
            "height_above_reference_ellipsoid",
            "height_above_mean_sea_level",
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
    "positive": {"Z": ("up", "down"), "vertical": ("up", "down")},
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
    # "regular_expression": {
    #     "time": r"time[0-9]*",
    #     "vertical": (
    #         r"(lv_|bottom_top|sigma|h(ei)?ght|altitude|depth|isobaric|pres|"
    #         r"isotherm)[a-z_]*[0-9]*"
    #     ),
    #     "y": r"y",
    #     "latitude": r"x?lat[a-z0-9]*",
    #     "x": r"x",
    #     "longitude": r"x?lon[a-z0-9]*",
    # },
}

# "vertical" is just an alias for "Z"
coordinate_criteria["standard_name"]["vertical"] = coordinate_criteria["standard_name"][
    "Z"
]
# "long_name" and "standard_name" criteria are the same. For convenience.
coordinate_criteria["long_name"] = coordinate_criteria["standard_name"]

# Type for Mapper functions
Mapper = Callable[
    [Union[xr.DataArray, xr.Dataset], str, bool, str],
    Union[Optional[str], List[Optional[str]], DataArray],  # this sucks
]


def _strip_none_list(lst: List[Optional[str]]) -> List[str]:
    """ The mappers can return [None]. Strip that when necessary. Keeps mypy happy."""
    return [item for item in lst if item != [None]]  # type: ignore


def _get_axis_coord_single(
    var: Union[xr.DataArray, xr.Dataset],
    key: str,
    error: bool = True,
    default: str = None,
) -> Optional[str]:
    """ Helper method for when we really want only one result per key. """
    results = _get_axis_coord(var, key, error, default)
    if len(results) > 1:
        raise ValueError(
            f"Multiple results for {key!r} found: {results!r}. Is this valid CF? Please open an issue."
        )
    else:
        return results[0]


def _get_axis_coord(
    var: Union[xr.DataArray, xr.Dataset],
    key: str,
    error: bool = True,
    default: str = None,
) -> List[Optional[str]]:
    """
    Translate from axis or coord name to variable name

    Parameters
    ----------
    var: DataArray, Dataset
        DataArray belonging to the coordinate to be checked
    key: str, ["X", "Y", "Z", "T", "longitude", "latitude", "vertical", "time"]
        key to check for.
    error: bool
        raise errors when key is not found or interpretable. Use False and provide default
        to replicate dict.get(k, None).
    default: Any
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

    if key not in _COORD_NAMES and key not in _AXIS_NAMES:
        if error:
            raise KeyError(f"Did not understand key {key!r}")
        else:
            return [default]

    if "coordinates" in var.encoding:
        search_in = var.encoding["coordinates"].split(" ")
    elif "coordinates" in var.attrs:
        search_in = var.attrs["coordinates"].split(" ")
    else:
        search_in = list(var.coords)

    results: Set = set()
    for coord in search_in:
        for criterion, valid_values in coordinate_criteria.items():
            if key in valid_values:
                expected = valid_values[key]
                if var.coords[coord].attrs.get(criterion, None) in expected:
                    results.update((coord,))

    if not results:
        if error:
            raise KeyError(f"axis name {key!r} not found!")
        else:
            return [default]
    else:
        return list(results)


def _get_measure_variable(
    da: xr.DataArray, key: str, error: bool = True, default: str = None
) -> DataArray:
    """ tiny wrapper since xarray does not support providing str for weights."""
    return da[_get_measure(da, key, error, default)]


def _get_measure(
    da: xr.DataArray, key: str, error: bool = True, default: str = None
) -> Optional[str]:
    """
    Translate from cell measures ("area" or "volume") to appropriate variable name.
    This function interprets the ``cell_measures`` attribute on DataArrays.

    Parameters
    ----------
    da: DataArray
        DataArray belonging to the coordinate to be checked
    key: str, ["area", "volume"]
        key to check for.
    error: bool
        raise errors when key is not found or interpretable. Use False and provide default
        to replicate dict.get(k, None).
    default: Any
        default value to return when error is False.

    Returns
    -------
    List[str], Variable name(s) in parent xarray object that matches axis or coordinate `key`
    """
    if not isinstance(da, DataArray):
        raise NotImplementedError("Measures not implemented for Datasets yet.")
    if key not in _CELL_MEASURES:
        if error:
            raise ValueError(
                f"Cell measure must be one of {_CELL_MEASURES!r}. Received {key!r} instead."
            )
        else:
            return default

    if "cell_measures" not in da.attrs:
        if error:
            raise KeyError("'cell_measures' not present in 'attrs'.")
        else:
            return default

    attr = da.attrs["cell_measures"]
    strings = [s.strip() for s in attr.strip().split(":")]
    if len(strings) % 2 != 0:
        if error:
            raise ValueError(f"attrs['cell_measures'] = {attr!r} is malformed.")
        else:
            return default
    measures = dict(zip(strings[slice(0, None, 2)], strings[slice(1, None, 2)]))
    if key not in measures:
        if error:
            raise KeyError(
                f"Cell measure {key!r} not found. Please use .cf.describe() to see a list of key names that can be interpreted."
            )
        else:
            return default
    return measures[key]


#: Default mappers for common keys.
# TODO: Make the values of this a tuple,
#       so that multiple mappers can be used for a single key
#       We need this for groupby("T.month") and groupby("latitude") for example.
_DEFAULT_KEY_MAPPERS: Mapping[str, Mapper] = {
    "dim": _get_axis_coord,
    "dims_or_levels": _get_axis_coord,  # reset_index
    "coord": _get_axis_coord_single,
    "group": _get_axis_coord_single,
    "weights": _get_measure_variable,  # type: ignore
}


def _filter_by_standard_names(ds: xr.Dataset, name: Union[str, List[str]]) -> List[str]:
    """ returns a list of variable names with standard names matching name. """
    if isinstance(name, str):
        name = [name]

    varnames = []
    counts = dict.fromkeys(name, 0)
    for vname, var in ds.variables.items():
        stdname = var.attrs.get("standard_name", None)
        if stdname in name:
            varnames.append(str(vname))
            counts[stdname] += 1

    return varnames


def _get_list_standard_names(obj: xr.Dataset) -> List[str]:
    """
    Returns a sorted list of standard names in Dataset.

    Parameters
    ----------

    obj: DataArray, Dataset
        Xarray objec to process

    Returns
    -------
    list of standard names in dataset
    """
    names = []
    for k, v in obj.variables.items():
        if "standard_name" in v.attrs:
            names.append(v.attrs["standard_name"])
    return sorted(names)


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
    wrap_classes: bool
        Should we wrap the return value with _CFWrappedClass?
        Only True for the high level CFAccessor.
        Facilitates code reuse for _CFWrappedClass and _CFWrapppedPlotMethods
        For both of those, wrap_classes is False.
    extra_decorator: Callable (optional)
        An extra decorator, if necessary. This is used by _CFPlotMethods to set default
        kwargs based on CF attributes.
    """
    attribute: Union[Mapping, Callable] = getattr(obj, attr)

    if isinstance(attribute, Mapping):
        if not attribute:
            return dict(attribute)
        # attributes like chunks / sizes
        newmap = dict()
        unused_keys = set(attribute.keys())
        for key in _AXIS_NAMES + _COORD_NAMES:
            value = _get_axis_coord(obj, key, error=False, default=None)
            unused_keys -= set(value)
            if value != [None]:
                good_values = set(value) & set(obj.dims)
                if not good_values:
                    continue
                if len(good_values) > 1:
                    raise AttributeError(
                        f"cf_xarray can't wrap attribute {attr!r} because there are multiple values for {key!r} viz. {good_values!r}. "
                        f"There is no unique mapping from {key!r} to a value in {attr!r}."
                    )
                newmap.update({key: attribute[good_values.pop()]})
        newmap.update({key: attribute[key] for key in unused_keys})
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
        arguments = accessor._process_signature(func, args, kwargs, key_mappers)
        final_func = extra_decorator(func) if extra_decorator else func
        result = final_func(**arguments)
        if wrap_classes and isinstance(result, _WRAPPED_CLASSES):
            result = _CFWrappedClass(result, accessor)

        return result

    return wrapper


class _CFWrappedClass:
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

        For now, this is setting ``xincrease`` and ``yincrease``. It could set
        other arguments in the future.
        """
        valid_keys = self.accessor.get_valid_keys()

        @functools.wraps(func)
        def _plot_wrapper(*args, **kwargs):
            if "x" in kwargs:
                if kwargs["x"] in valid_keys:
                    xvar = self.accessor[kwargs["x"]]
                else:
                    xvar = self._obj[kwargs["x"]]
                if "positive" in xvar.attrs:
                    if xvar.attrs["positive"] == "down":
                        kwargs.setdefault("xincrease", False)
                    else:
                        kwargs.setdefault("xincrease", True)

            if "y" in kwargs:
                if kwargs["y"] in valid_keys:
                    yvar = self.accessor[kwargs["y"]]
                else:
                    yvar = self._obj[kwargs["y"]]
                if "positive" in yvar.attrs:
                    if yvar.attrs["positive"] == "down":
                        kwargs.setdefault("yincrease", False)
                    else:
                        kwargs.setdefault("yincrease", True)

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
            key_mappers=dict.fromkeys(self._keys, _get_axis_coord_single),
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
            key_mappers=dict.fromkeys(self._keys, _get_axis_coord_single),
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

    def _process_signature(self, func: Callable, args, kwargs, key_mappers):
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
        var_kws = []
        for param in sig.parameters:
            if sig.parameters[param].kind is inspect.Parameter.VAR_KEYWORD:
                var_kws.append(param)

        if args or kwargs:
            bound = sig.bind(*args, **kwargs)
            arguments = self._rewrite_values(
                bound.arguments, key_mappers, tuple(var_kws)
            )
        else:
            arguments = {}

        if arguments:
            # now unwrap the **indexers_kwargs type arguments
            # so that xarray can parse it :)
            for kw in var_kws:
                value = arguments.pop(kw, None)
                if value:
                    arguments.update(**value)

        return arguments

    def _rewrite_values(self, kwargs, key_mappers: dict, var_kws):
        """
        Rewrites the values in a Mapping from kwarg to value.

        Parameters
        ----------
        kwargs: Mapping
            Mapping from kwarg name to value
        key_mappers: Mapping
            Mapping from kwarg name to a Mapper function that will convert a
            given CF "special" name to an xarray name.
        var_kws: List[str]
            List of variable kwargs that need special treatment.
            e.g. **indexers_kwargs in isel

        Returns
        -------
        dict of kwargs with fully rewritten values.
        """
        updates: dict = {}

        # allow multiple return values here.
        # these are valid for .sel, .isel, .coarsen
        key_mappers.update(dict.fromkeys(var_kws, _get_axis_coord))

        for key, mapper in key_mappers.items():
            value = kwargs.get(key, None)

            if value is not None:
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
                            dict.fromkeys(mapper(self._obj, k, False, k), v)
                            for k, v in value.items()
                        ]
                    )

                elif value is Ellipsis:
                    pass

                else:
                    # things like sum which have dim
                    newvalue = [mapper(self._obj, v, False, v) for v in value]
                    if len(newvalue) == 1:
                        # works for groupby("time")
                        newvalue = newvalue[0]
                    else:
                        # Mappers return list by default
                        # for input dim=["lat", "X"], newvalue=[["lat"], ["lon"]],
                        # so we deal with that here.
                        newvalue = list(itertools.chain(*newvalue))
                    updates[key] = newvalue

        kwargs.update(updates)

        # TODO: is there a way to merge this with above?
        # maybe the keys we are looking for are in kwargs.
        # For example, this happens with DataArray.plot(),
        # where the signature is obscured and kwargs is
        #    kwargs = {"x": "X", "col": "T"}
        for vkw in var_kws:
            if vkw in kwargs:
                maybe_update = {
                    k: _get_axis_coord_single(self._obj, v, False, v)
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

    @property
    def plot(self):
        return _CFWrappedPlotMethods(self._obj, self)

    def _describe(self):
        text = "Axes:\n"
        for key in _AXIS_NAMES:
            text += f"\t{key}: {_get_axis_coord(self._obj, key, error=False, default=None)}\n"

        text += "\nCoordinates:\n"
        for key in _COORD_NAMES:
            text += f"\t{key}: {_get_axis_coord(self._obj, key, error=False, default=None)}\n"

        text += "\nCell Measures:\n"
        for measure in _CELL_MEASURES:
            if isinstance(self._obj, xr.Dataset):
                text += f"\t{measure}: unsupported\n"
            else:
                text += f"\t{measure}: {_get_measure(self._obj, measure, error=False, default=None)}\n"

        text += "\nStandard Names:\n"
        if isinstance(self._obj, xr.DataArray):
            text += "\tunsupported\n"
        else:
            stdnames = _get_list_standard_names(self._obj)
            text += "\t"
            text += "\n".join(
                textwrap.wrap(f"{stdnames!r}", 70, break_long_words=False)
            )
        return text

    def describe(self):
        """
        Print a string repr to screen.
        """
        print(self._describe())

    def get_valid_keys(self) -> Set[str]:
        """
        Utility function that returns valid keys for .cf[].

        This is useful for checking whether a key is valid for indexing, i.e.
        that the attributes necessary to allow indexing by that key exist.

        Returns
        -------
        Set of valid key names that can be used with __getitem__ or .cf[key].
        """
        varnames = [
            key
            for key in _AXIS_NAMES + _COORD_NAMES
            if _get_axis_coord(self._obj, key, error=False, default=None) != [None]
        ]
        with suppress(NotImplementedError):
            measures = [
                key
                for key in _CELL_MEASURES
                if _get_measure(self._obj, key, error=False) is not None
            ]
            if measures:
                varnames.append(*measures)

        if not isinstance(self._obj, xr.DataArray):
            varnames.extend(_get_list_standard_names(self._obj))
        return set(varnames)

    def __getitem__(self, key: Union[str, List[str]]):

        kind = str(type(self._obj).__name__)
        scalar_key = isinstance(key, str)

        if isinstance(self._obj, xr.DataArray) and not scalar_key:
            raise KeyError(
                f"Cannot use a list of keys with DataArrays. Expected a single string. Received {key!r} instead."
            )

        if scalar_key:
            key = (key,)  # type: ignore

        varnames: List[Hashable] = []
        coords: List[Hashable] = []
        successful = dict.fromkeys(key, False)
        for k in key:
            if k in _AXIS_NAMES + _COORD_NAMES:
                names = _get_axis_coord(self._obj, k)
                successful[k] = bool(names)
                coords.extend(_strip_none_list(names))
            elif k in _CELL_MEASURES:
                if isinstance(self._obj, xr.Dataset):
                    raise NotImplementedError(
                        "Invalid key {k!r}. Cell measures not implemented for Dataset yet."
                    )
                else:
                    measure = _get_measure(self._obj, k)
                    successful[k] = bool(measure)
                    if measure:
                        varnames.append(measure)
            elif not isinstance(self._obj, xr.DataArray):
                stdnames = _filter_by_standard_names(self._obj, k)
                successful[k] = bool(stdnames)
                varnames.extend(stdnames)
                coords.extend(list(set(stdnames) & set(self._obj.coords)))

        # these are not special names but could be variable names in underlying object
        # we allow this so that we can return variables with appropriate CF auxiliary variables
        varnames.extend([k for k, v in successful.items() if not v])

        try:
            # TODO: make this a get_auxiliary_variables function
            # make sure to set coordinate variables referred to in "coordinates" attribute
            for name in varnames:
                attrs = self._obj[name].attrs
                if "coordinates" in attrs:
                    coords.extend(attrs.get("coordinates").split(" "))

                if "cell_measures" in attrs:
                    measures = [
                        _get_measure(self._obj[name], measure)
                        for measure in _CELL_MEASURES
                        if measure in attrs["cell_measures"]
                    ]
                    coords.extend(_strip_none_list(measures))

                if isinstance(self._obj, xr.Dataset) and "ancillary_variables" in attrs:
                    anames = attrs["ancillary_variables"].split(" ")
                    coords.extend(anames)

            if isinstance(self._obj, xr.DataArray):
                ds = self._obj._to_temp_dataset()
            else:
                ds = self._obj

            if scalar_key and len(varnames) == 1:
                da = ds[varnames[0]]
                for k1 in coords:
                    da.coords[k1] = ds.variables[k1]
                return da

            ds = ds.reset_coords()[varnames + coords]
            if isinstance(self._obj, DataArray):
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
                f"Use {kind}.cf.describe() to see a list of key names that can be interpreted."
            )


@xr.register_dataset_accessor("cf")
class CFDatasetAccessor(CFAccessor):
    pass


@xr.register_dataarray_accessor("cf")
class CFDataArrayAccessor(CFAccessor):
    pass
