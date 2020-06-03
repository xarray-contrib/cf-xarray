import functools
import inspect
from typing import Union

import xarray as xr
from xarray import DataArray, Dataset

_WRAPPED_CLASSES = (
    xr.core.resample.Resample,
    xr.core.groupby.GroupBy,
    xr.core.rolling.Rolling,
    xr.core.rolling.Coarsen,
    xr.core.weighted.Weighted,
)


_DEFAULT_KEYS_TO_REWRITE = ("dim", "coord", "group")


def _get_axis_name_mapping(da: xr.DataArray):
    return {"X": "lon", "Y": "lat", "T": "time"}


def _getattr(
    obj: Union[DataArray, Dataset],
    attr: str,
    accessor: "CFAccessor",
    wrap_classes=False,
    keys=_DEFAULT_KEYS_TO_REWRITE,
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
        arguments = accessor._process_signature(func, args, kwargs, keys=keys)

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
        return _getattr(obj=self.wrapped, attr=attr, accessor=self.accessor)


class _CFWrappedPlotMethods:
    def __init__(self, obj, accessor):
        self._obj = obj
        self.accessor = accessor
        self._keys = ("x", "y", "hue", "col", "row")

    def __call__(self, *args, **kwargs):
        plot = _getattr(
            obj=self._obj, attr="plot", accessor=self.accessor, keys=self._keys
        )
        return plot(*args, **kwargs)

    def __getattr__(self, attr):
        return _getattr(
            obj=self._obj.plot, attr=attr, accessor=self.accessor, keys=self._keys
        )


@xr.register_dataarray_accessor("cf")
@xr.register_dataset_accessor("cf")
class CFAccessor:
    def __init__(self, da):
        self._obj = da
        self._coords = _get_axis_name_mapping(da)

    def _process_signature(self, func, args, kwargs, keys):
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
            arguments = self._rewrite_values_with_axis_names(
                bound.arguments, keys, tuple(var_kws)
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

    def _rewrite_values_with_axis_names(self, kwargs, keys, var_kws):
        """ rewrites 'dim' for example. """
        updates = {}
        for key in tuple(keys) + tuple(var_kws):
            value = kwargs.get(key, None)
            if value:
                if isinstance(value, str):
                    value = [value]

                if isinstance(value, dict):
                    # this for things like isel where **kwargs captures things like T=5
                    updates[key] = {self._coords.get(k, k): v for k, v in value.items()}
                elif value is Ellipsis:
                    pass
                else:
                    # things like sum which have dim
                    updates[key] = [self._coords.get(v, v) for v in value]
                    if len(updates[key]) == 1:
                        updates[key] = updates[key][0]

        kwargs.update(updates)

        # maybe the keys we are looking for are in kwargs.
        # For example, this happens with DataArray.plot(),
        # where the signature is obscured and kwargs is
        #    kwargs = {"x": "X", "col": "T"}
        for vkw in var_kws:
            if vkw in kwargs:
                maybe_update = {
                    k: self._coords.get(v, v)
                    for k, v in kwargs[vkw].items()
                    if k in keys
                }
                kwargs[vkw].update(maybe_update)

        return kwargs

    def __getattr__(self, attr):
        return _getattr(obj=self._obj, attr=attr, accessor=self, wrap_classes=True)

    @property
    def plot(self):
        return _CFWrappedPlotMethods(self._obj, self)
