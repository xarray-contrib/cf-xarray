import functools
import inspect
from typing import Any, Union

import xarray as xr
from xarray import DataArray, Dataset

_WRAPPED_CLASSES = (
    xr.core.resample.Resample,
    xr.core.groupby.GroupBy,
    xr.core.rolling.Rolling,
    xr.core.rolling.Coarsen,
    xr.core.weighted.Weighted,
)


_AXIS_NAMES = ("X", "Y", "Z", "T")
_COORD_NAMES = ("longitude", "latitude", "vertical", "time")
_COORD_AXIS_MAPPING = dict(zip(_COORD_NAMES, _AXIS_NAMES))
_CELL_MEASURES = ("area", "volume")

# Define the criteria for coordinate matches
# Copied from metpy
# Internally we only use X, Y, Z, T
# TODO: Metpy adds latitude and longitude separately so we may revert to doing that too
coordinate_criteria = {
    "standard_name": {
        "T": ("time",),
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
        "Y": ("latitude",),
        "X": ("longitude",),
    },
    "_CoordinateAxisType": {
        "T": ("Time",),
        "Z": ("GeoZ", "Height", "Pressure"),
        "Y": ("GeoY", "Lat"),
        "X": ("GeoX", "Lon"),
    },
    "axis": {"T": ("T",), "Z": ("Z",), "Y": ("Y",), "X": ("X",)},
    "positive": {"Z": ("up", "down")},
    "units": {
        "Y": (
            "degree_north",
            "degree_N",
            "degreeN",
            "degrees_north",
            "degrees_N",
            "degreesN",
        ),
        "X": (
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


def _get_axis_coord(var: xr.DataArray, key, error: bool = True, default: Any = None):
    """
    Translate from axis or coord name to variable name

    Parameters
    ----------
    var : `xarray.DataArray`
        DataArray belonging to the coordinate to be checked
    key : str, ["X", "Y", "Z", "T", "longitude", "latitude", "vertical", "time"]
        key to check for.
    error : bool
        raise errors when key is not found or interpretable. Use False and provide default
        to replicate dict.get(k, None).
    default: Any
        default value to return when error is False.

    Returns
    -------
    str, Variable name in parent xarray object that matches axis or coordinate `key`

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

    axis = None
    if key in _COORD_NAMES:
        coord = key
        axis = _COORD_AXIS_MAPPING[key]
    elif key in _AXIS_NAMES:
        coord = ""
        axis = key
    else:
        if error:
            raise KeyError(f"Did not understand {key}")
        else:
            return default

    if axis is None:
        raise AssertionError("Should be unreachable")

    for coord in var.coords:
        for criterion, valid_values in coordinate_criteria.items():
            if axis in valid_values:  # type: ignore
                expected = valid_values[axis]  # type: ignore
                if var.coords[coord].attrs.get(criterion, None) in expected:
                    return coord

    if error:
        raise KeyError(f"axis name {key!r} not found!")
    else:
        return default


def _get_measure_variable(
    da: xr.DataArray, key: str, error: bool = True, default: Any = None
) -> DataArray:
    """ tiny wrapper since xarray does not support providing str for weights."""
    return da[_get_measure(da, key, error, default)]


def _get_measure(da: xr.DataArray, key: str, error: bool = True, default: Any = None):
    """
    Interprets 'cell_measures'.
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
    return measures[key]


_DEFAULT_KEY_MAPPERS: dict = dict.fromkeys(("dim", "coord", "group"), _get_axis_coord)
_DEFAULT_KEY_MAPPERS["weights"] = _get_measure_variable


def _getattr(
    obj: Union[DataArray, Dataset],
    attr: str,
    accessor: "CFAccessor",
    key_mappers: dict,
    wrap_classes=False,
):
    """
    Common getattr functionality.

    Parameters
    ----------

    obj : DataArray, Dataset
    attr : Name of attribute in obj that will be shadowed.
    accessor : High level accessor object: CFAccessor
    wrap_classes: bool
        Should we wrap the return value with _CFWrappedClass?
        Only True for the high level CFAccessor.
        Facilitates code reuse for _CFWrappedClass and _CFWrapppedPlotMethods
        For both of thos, wrap_classes is False.
    """
    func = getattr(obj, attr)

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        arguments = accessor._process_signature(func, args, kwargs, key_mappers)
        result = func(**arguments)
        if wrap_classes and isinstance(result, _WRAPPED_CLASSES):
            result = _CFWrappedClass(result, accessor)

        return result

    return wrapper


class _CFWrappedClass:
    def __init__(self, towrap, accessor: "CFAccessor"):
        """

        Parameters
        ----------
        obj : DataArray, Dataset
        towrap : Resample, GroupBy, Coarsen, Rolling, Weighted
            Instance of xarray class that is being wrapped.
        accessor : CFAccessor
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
    def __init__(self, obj, accessor):
        self._obj = obj
        self.accessor = accessor
        self._keys = ("x", "y", "hue", "col", "row")

    def __call__(self, *args, **kwargs):
        plot = _getattr(
            obj=self._obj,
            attr="plot",
            accessor=self.accessor,
            key_mappers=dict.fromkeys(self._keys, _get_axis_coord),
        )
        return plot(*args, **kwargs)

    def __getattr__(self, attr):
        return _getattr(
            obj=self._obj.plot,
            attr=attr,
            accessor=self.accessor,
            key_mappers=dict.fromkeys(self._keys, _get_axis_coord),
        )


class CFAccessor:
    def __init__(self, da):
        self._obj = da

    def _process_signature(self, func, args, kwargs, key_mappers):
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
            print(arguments)
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
        """ rewrites 'dim' for example using 'mapper' """
        updates: dict = {}
        key_mappers.update(dict.fromkeys(var_kws, _get_axis_coord))
        for key, mapper in key_mappers.items():
            value = kwargs.get(key, None)
            if value is not None:
                if isinstance(value, str):
                    value = [value]

                if isinstance(value, dict):
                    # this for things like isel where **kwargs captures things like T=5
                    updates[key] = {
                        mapper(self._obj, k, False, k): v for k, v in value.items()
                    }
                elif value is Ellipsis:
                    pass
                else:
                    # things like sum which have dim
                    updates[key] = [mapper(self._obj, v, False, v) for v in value]
                    if len(updates[key]) == 1:
                        updates[key] = updates[key][0]

        kwargs.update(updates)

        # TODO: is there a way to merge this with above?
        # maybe the keys we are looking for are in kwargs.
        # For example, this happens with DataArray.plot(),
        # where the signature is obscured and kwargs is
        #    kwargs = {"x": "X", "col": "T"}
        for vkw in var_kws:
            if vkw in kwargs:
                maybe_update = {
                    k: _get_axis_coord(self._obj, v, False, v)
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


@xr.register_dataset_accessor("cf")
class CFDatasetAccessor(CFAccessor):
    def __getitem__(self, key):
        if key in _AXIS_NAMES + _COORD_NAMES:
            return self._obj[_get_axis_coord(self._obj, key)]
        elif key in _CELL_MEASURES:
            raise NotImplementedError("measures not implemented for Dataset yet.")
            # return self._obj[_get_measure(self._obj)[key]]
        else:
            raise KeyError(f"DataArray.cf does not understand the key {key}")

    # def __getitem__(self, key):
    #     raise AttributeError("Dataset.cf does not support [] indexing or __getitem__")


@xr.register_dataarray_accessor("cf")
class CFDataArrayAccessor(CFAccessor):
    def __getitem__(self, key):
        if key in _AXIS_NAMES + _COORD_NAMES:
            return self._obj[_get_axis_coord(self._obj, key)]
        elif key in _CELL_MEASURES:
            return self._obj[_get_measure(self._obj, key)]
        else:
            raise KeyError(f"DataArray.cf does not understand the key {key}")
