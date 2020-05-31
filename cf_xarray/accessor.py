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


@xr.register_dataarray_accessor("cf")
class CFAccessor:
    def __init__(self, da):
        self._obj = da
        self._coords = _get_axis_name_mapping(da)

    def _process_signature(self, func, args, kwargs):
        sig = inspect.signature(func)

        # Catch things like .isel(T=5).
        # This assigns indexers_kwargs=dict(T=5).
        # and indexers_kwargs is of kind VAR_KEYWORD
        var_kws = []
        for param in sig.parameters:
            if sig.parameters[param].kind is inspect.Parameter.VAR_KEYWORD:
                var_kws.append(param)

        bound = sig.bind(*args, **kwargs)
        arguments = self._rewrite_values_with_axis_names(
            bound.arguments, ["dim",] + var_kws
        )

        if arguments:
            # now unwrap the **indexers_kwargs type arguments
            for kw in var_kws:
                value = arguments.pop(kw, None)
                if value:
                    arguments.update(**value)

        return arguments

    def _rewrite_values_with_axis_names(self, kwargs, keys):
        """ rewrites 'dim' for example. """
        updates = {}
        for key in keys:
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

    def plot(self, *args, **kwargs):
        if args:
            raise ValueError("cf.plot can only be called with keyword arguments.")

        kwargs = self._rewrite_values_with_axis_names(
            kwargs, ("x", "y", "hue", "col", "row")
        )
        return self._obj.plot(*args, **kwargs)
