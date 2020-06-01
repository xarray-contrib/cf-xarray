import copy
import inspect
import functools

import xarray as xr


_WRAPPED_CLASSES = (
    xr.core.resample.Resample,
    xr.core.groupby.GroupBy,
    xr.core.rolling.Rolling,
    xr.core.rolling.Coarsen,
    xr.core.weighted.Weighted,
)


def _get_axis_name_mapping(da: xr.DataArray):
    return {"X": "lon", "Y": "lat", "T": "time"}


class _CFWrapped:
    def __init__(self, towrap, accessor):
        self.wrapped = towrap
        self.accessor = accessor
        self._can_wrap_classes = False

    def __repr__(self):
        return "--- CF-xarray wrapped \n" + repr(self.wrapped)

    def __getattr__(self, attr):
        func = getattr(self.wrapped, attr)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arguments = self.accessor._process_signature(func, args, kwargs)
            rv = func(**arguments)
            return rv

        return wrapper


class _CFWrappedPlotMethods:
    def __init__(self, obj, accessor):
        self._obj = obj
        self.accessor = accessor
        self._can_wrap_classes = False

    def __call__(self, *args, **kwargs):
        func = self._obj.plot  # (*args, **kwargs)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arguments = self.accessor._process_signature(
                func, args, kwargs, keys=("x", "y", "hue", "col", "row")
            )
            print(arguments)
            rv = func(**arguments)
            return rv

        return wrapper(*args, **kwargs)

    def __getattr__(self, attr):
        func = getattr(self._obj.plot, attr)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arguments = self.accessor._process_signature(
                func, args, kwargs, keys=("x", "y", "hue", "col", "row")
            )
            rv = func(**arguments)
            return rv

        return wrapper


@xr.register_dataarray_accessor("cf")
class CFAccessor:
    def __init__(self, da):
        self._obj = da
        self._coords = _get_axis_name_mapping(da)
        self._can_wrap_classes = True

    def _process_signature(self, func, args, kwargs, keys=("dim",)):
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
                else:
                    # things like sum which have dim
                    updates[key] = [self._coords.get(v, v) for v in value]
                    if len(updates[key]) == 1:
                        updates[key] = updates[key][0]

        kwargs.update(updates)

        # maybe the keys I'm looking for are in kwargs.
        # This happens with DataArray.plot() for example, where the signature is obscured.
        for vkw in var_kws:
            if vkw in kwargs:
                maybe_update = {
                    k: self._coords.get(v, v)
                    for k, v in kwargs[vkw].items()
                    if k in keys
                }
                kwargs[vkw].update(maybe_update)

        return kwargs

    def __getattr__(self, name):
        func = getattr(self._obj, name)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            arguments = self._process_signature(func, args, kwargs)
            rv = func(**arguments)
            if isinstance(rv, _WRAPPED_CLASSES):
                return _CFWrapped(rv, self)
            else:
                return rv

        return wrapper

    @property
    def plot(self):
        return _CFWrappedPlotMethods(self._obj, self)
