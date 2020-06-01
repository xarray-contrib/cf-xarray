from typing import Any, Hashable, Mapping, Optional, TypeVar, cast

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
