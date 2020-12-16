from typing import Any, Dict, Hashable, Mapping, Optional, TypeVar, cast

K = TypeVar("K")
V = TypeVar("V")
T = TypeVar("T")


def either_dict_or_kwargs(
    pos_kwargs: Optional[Mapping[Hashable, T]],
    kw_kwargs: Mapping[str, T],
    func_name: str,
) -> Mapping[Hashable, T]:
    if pos_kwargs is not None:
        if not is_dict_like(pos_kwargs):
            raise ValueError(
                "the first argument to .%s must be a dictionary" % func_name
            )
        if kw_kwargs:
            raise ValueError(
                "cannot specify both keyword and positional "
                "arguments to .%s" % func_name
            )
        return pos_kwargs
    else:
        # Need an explicit cast to appease mypy due to invariance; see
        # https://github.com/python/mypy/issues/6228
        return cast(Mapping[Hashable, T], kw_kwargs)


def is_dict_like(value: Any) -> bool:
    return hasattr(value, "keys") and hasattr(value, "__getitem__")


# copied from xarray
class UncachedAccessor:
    """Acts like a property, but on both classes and class instances
    This class is necessary because some tools (e.g. pydoc and sphinx)
    inspect classes for which property returns itself and not the
    accessor.
    """

    def __init__(self, accessor):
        self._accessor = accessor

    def __get__(self, obj, cls):
        if obj is None:
            return self._accessor

        return self._accessor(obj)


def parse_cell_methods_attr(attr: str) -> Dict[str, str]:
    """
    Parse cell_methods attributes (format is 'measure: name').

    Parameters
    ----------
    attr: str
        String to parse

    Returns
    -------
    Dictionary mapping measure to name
    """
    strings = [s for scolons in attr.split(":") for s in scolons.split()]
    if len(strings) % 2 != 0:
        raise ValueError(f"attrs['cell_measures'] = {attr!r} is malformed.")

    return dict(zip(strings[slice(0, None, 2)], strings[slice(1, None, 2)]))
