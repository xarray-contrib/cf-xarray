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
        )
    )
    for attr in ["face_dimensions", "edge1_dimensions", "edge2_dimensions", "foo"]:
        if attr in grid.attrs:
            matches = re.findall(pattern, grid.attrs[attr] + "\n")
            assert len(matches) == ndim, matches
            for ax, match in zip(axes_names, matches):
                axes[ax].update(set(match[:2]))

    if ndim == 2 and "vertical_dimensions" in grid.attrs:
        matches = re.findall(pattern, grid.attrs["vertical_dimensions"] + "\n")
        assert len(matches) == 1
        axes["Z"] = set(matches[0][:2])

    return axes
