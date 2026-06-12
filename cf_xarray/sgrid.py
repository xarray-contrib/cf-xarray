SGRID_DIM_ATTRS = [
    "face_dimensions",
    "volume_dimensions",
    # the following are optional and should be redundant with the above
    # at least for dimension names
    # "face1_dimensions",
    # "face2_dimensions",
    # "face3_dimensions",
    "edge1_dimensions",
    "edge2_dimensions",
    # "edge3_dimensions",
]

SGRID_COORD_ATTRS = [
    "node_coordinates",
    "face_coordinates",
    "edge1_coordinates",
    "edge2_coordinates",
    "volume_coordinates",
]


def get_topology_coords(ds, grid_var_name):
    """Return coordinate variable names referenced by an SGRID topology variable.

    Reads ``node_coordinates``, ``face_coordinates``, ``edge{1,2}_coordinates``,
    and ``volume_coordinates`` from the topology variable's attrs and filters
    to names that are actually present in ``ds``.
    """
    if grid_var_name not in ds.variables:
        return []
    grid_attrs = ds[grid_var_name].attrs
    names: list[str] = []
    for attr_name in SGRID_COORD_ATTRS:
        if coord_str := grid_attrs.get(attr_name):
            names.extend(n for n in coord_str.split() if n in ds.variables)
    return names


def parse_axes(ds):
    import re

    (gridvar,) = ds.cf.cf_roles["grid_topology"]
    grid = ds[gridvar]
    pattern = re.compile("\\s?(.*?):\\s*(.*?)\\s+(?:\\(padding:(.+?)\\))?")
    ndim = grid.attrs["topology_dimension"]
    axes_names = ["X", "Y", "Z"][:ndim]
    axes = dict(
        zip(
            axes_names,
            ({k} for k in grid.attrs["node_dimensions"].split(" ")),
            strict=False,
        )
    )
    for attr in SGRID_DIM_ATTRS:
        if attr in grid.attrs:
            matches = re.findall(pattern, grid.attrs[attr] + "\n")
            assert len(matches) == ndim, matches
            for ax, match in zip(axes_names, matches, strict=False):
                axes[ax].update(set(match[:2]))

    if ndim == 2 and "vertical_dimensions" in grid.attrs:
        matches = re.findall(pattern, grid.attrs["vertical_dimensions"] + "\n")
        assert len(matches) == 1
        axes["Z"] = set(matches[0][:2])

    return axes
