import warnings
from functools import partial
from typing import Dict, Hashable, Iterable, List

STAR = " * "
TAB = len(STAR) * " "


def _format_missing_row(row: str, rich: bool) -> str:
    if rich:
        return f"[grey62]{row}[/grey62]"
    else:
        return row


def _format_varname(name, rich: bool):
    return name


def _format_subtitle(name: str, rich: bool) -> str:
    if rich:
        return f"[bold]{name}[/bold]"
    else:
        return name


def _format_cf_name(name: str, rich: bool) -> str:
    if rich:
        return f"[color(33)]{name}[/color(33)]"
    else:
        return name


def make_text_section(
    accessor,
    subtitle: str,
    attr: str,
    dims=None,
    valid_keys=None,
    valid_values=None,
    default_keys=None,
    rich: bool = False,
):

    from .accessor import sort_maybe_hashable

    if dims is None:
        dims = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if isinstance(attr, str):
            try:
                vardict: Dict[str, Iterable[Hashable]] = getattr(accessor, attr, {})
            except ValueError:
                vardict = {}
        else:
            assert isinstance(attr, dict)
            vardict = attr
    if valid_keys:
        vardict = {k: v for k, v in vardict.items() if k in valid_keys}

    # Sort keys if there aren't extra keys,
    # preserve default keys order otherwise.
    default_keys = [] if not default_keys else list(default_keys)
    extra_keys = list(set(vardict) - set(default_keys))
    ordered_keys = sorted(vardict) if extra_keys else default_keys
    vardict = {key: vardict[key] for key in ordered_keys if key in vardict}

    # Keep only valid values (e.g., coords or data_vars)
    if valid_values is not None:
        vardict = {
            key: set(value).intersection(valid_values)
            for key, value in vardict.items()
            if set(value).intersection(valid_values)
        }

    # Star for keys with dims only, tab otherwise
    rows = [
        (
            f"{STAR if dims and set(value) <= set(dims) else TAB}"
            f"{_format_cf_name(key, rich)}: "
            f"{_format_varname(sort_maybe_hashable(value), rich)}"
        )
        for key, value in vardict.items()
    ]

    # Append missing default keys followed by n/a
    if default_keys:
        missing_keys = [key for key in default_keys if key not in vardict]
        if missing_keys:
            rows.append(
                _format_missing_row(TAB + ", ".join(missing_keys) + ": n/a", rich)
            )
    elif not rows:
        rows.append(_format_missing_row(TAB + "n/a", rich))

    return _print_rows(subtitle, rows, rich)


def _print_rows(subtitle: str, rows: List[str], rich: bool):
    subtitle = f"{subtitle.rjust(20)}:"

    # Add subtitle to the first row, align other rows
    rows = [
        _format_subtitle(subtitle, rich=rich) + row
        if i == 0
        else len(subtitle) * " " + row
        for i, row in enumerate(rows)
    ]

    return "\n".join(rows) + "\n\n"


def _format_conventions(string: str, rich: bool):
    row = _print_rows(
        subtitle="Conventions",
        rows=[_format_cf_name(TAB + string, rich=rich)],
        rich=rich,
    )
    if rich:
        row = row.rstrip()
    return row


def _maybe_panel(textgen, title: str, rich: bool):
    text = "".join(textgen)
    if rich:
        from rich.panel import Panel

        return Panel(
            f"[color(241)]{text.rstrip()}[/color(241)]",
            expand=True,
            title_align="left",
            title=f"[bold][color(244)]{title}[/bold][/color(244)]",
            highlight=True,
            width=100,
        )
    else:
        return title + ":\n" + text


def _format_flags(accessor, rich):
    from .accessor import create_flag_dict

    flag_dict = create_flag_dict(accessor._obj)
    rows = [
        f"{TAB}{_format_varname(v, rich)}: {_format_cf_name(k, rich)}"
        for k, v in flag_dict.items()
    ]
    return _print_rows("Flag Meanings", rows, rich)


def _format_dsg_roles(accessor, dims, rich):
    from .criteria import _DSG_ROLES

    yield make_text_section(
        accessor,
        "CF Roles",
        "cf_roles",
        dims=dims,
        valid_keys=_DSG_ROLES,
        rich=rich,
    )


def _format_coordinates(accessor, dims, coords, rich):
    from .accessor import _AXIS_NAMES, _CELL_MEASURES, _COORD_NAMES

    section = partial(
        make_text_section, accessor=accessor, dims=dims, valid_values=coords, rich=rich
    )

    yield section(subtitle="CF Axes", attr="axes", default_keys=_AXIS_NAMES)
    yield section(
        subtitle="CF Coordinates", attr="coordinates", default_keys=_COORD_NAMES
    )
    yield section(
        subtitle="Cell Measures", attr="cell_measures", default_keys=_CELL_MEASURES
    )
    yield section(subtitle="Standard Names", attr="standard_names")
    yield section(subtitle="Bounds", attr="bounds")
    yield section(subtitle="Grid Mappings", attr="grid_mapping_names")


def _format_data_vars(accessor, data_vars, rich):
    from .accessor import _CELL_MEASURES

    section = partial(
        make_text_section,
        accessor=accessor,
        dims=None,
        valid_values=data_vars,
        rich=rich,
    )

    yield section(
        subtitle="Cell Measures", attr="cell_measures", default_keys=_CELL_MEASURES
    )
    yield section(subtitle="Standard Names", attr="standard_names")
    yield section(subtitle="Bounds", attr="bounds")
    yield section(subtitle="Grid Mappings", attr="grid_mapping_names")


def _format_sgrid(accessor, axes, rich):
    yield make_text_section(
        accessor,
        "CF role",
        "cf_roles",
        valid_keys=["grid_topology"],
        rich=rich,
    )

    yield make_text_section(
        accessor,
        "Axes",
        axes,
        accessor._obj.dims,
        valid_values=accessor._obj.dims,
        default_keys=axes.keys(),
        rich=rich,
    )
