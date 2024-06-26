from __future__ import annotations

import copy
from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr

GEOMETRY_CONTAINER_NAME = "geometry_container"

__all__ = [
    "decode_geometries",
    "encode_geometries",
    "cf_to_shapely",
    "shapely_to_cf",
]


# Useful convention language:
# 1. Whether linked to normal CF space-time coordinates with a nodes attribute or not, inclusion of such coordinates is
#    recommended to maintain backward compatibility with software that has not implemented geometry capabilities.
# 2. The geometry node coordinate variables must each have an axis attribute whose allowable values are X, Y, and Z.
# 3. If a coordinates attribute is carried by the geometry container variable or its parent data variable, then those coordinate variables
#    that have a meaningful correspondence with node coordinates are indicated as such by a nodes attribute that names the corresponding node
#    coordinates, but only if the grid_mapping associated the geometry node variables is the same as that of the coordinate variables.
#    If a different grid mapping is used, then the provided coordinates must not have the nodes attribute.
#
# Interpretation:
# 1. node coordinates are exact; the 'normal' coordinates are a reasonable value to use, if you do not know how to interpret the nodes.


def decode_geometries(encoded: xr.Dataset) -> xr.Dataset:
    """
    Decode CF encoded geometries to a numpy object array containing shapely geometries.

    Parameters
    ----------
    encoded : Dataset
        A Xarray Dataset containing encoded geometries.

    Returns
    -------
    Dataset
        A Xarray Dataset containing decoded geometries.

    See Also
    --------
    shapely_to_cf
    cf_to_shapely
    encode_geometries
    """
    if GEOMETRY_CONTAINER_NAME not in encoded._variables:
        raise NotImplementedError(
            f"Currently only a single geometry variable named {GEOMETRY_CONTAINER_NAME!r} is supported."
            "A variable by this name is not present in the provided dataset."
        )

    enc_geom_var = encoded[GEOMETRY_CONTAINER_NAME]
    geom_attrs = enc_geom_var.attrs
    # Grab the coordinates attribute
    geom_attrs.update(enc_geom_var.encoding)

    geom_var = cf_to_shapely(encoded).variable

    todrop = (GEOMETRY_CONTAINER_NAME,) + tuple(
        s
        for s in " ".join(
            geom_attrs.get(attr, "")
            for attr in [
                "interior_ring",
                "node_coordinates",
                "node_count",
                "part_node_count",
                "coordinates",
            ]
        ).split(" ")
        if s
    )
    decoded = encoded.drop_vars(todrop)

    name = geom_attrs.get("variable_name", None)
    if name in decoded.dims:
        decoded = decoded.assign_coords(
            xr.Coordinates(coords={name: geom_var}, indexes={})
        )
    else:
        decoded[name] = geom_var

    # Is this a good idea? We are deleting information.
    for var in decoded._variables.values():
        if var.attrs.get("geometry") == GEOMETRY_CONTAINER_NAME:
            var.attrs.pop("geometry")
    return decoded


def encode_geometries(ds: xr.Dataset):
    """
    Encode any discovered geometry variables using the CF conventions.

    Practically speaking, geometry variables are numpy object arrays where the first
    element is a shapely geometry.

    .. warning::

       Only a single geometry variable is supported at present. Contributions to fix this
       are welcome.

    Parameters
    ----------
    ds : Dataset
       Dataset containing at least one geometry variable.

    Returns
    -------
    Dataset
       Where all geometry variables are encoded.

    See Also
    --------
    shapely_to_cf
    cf_to_shapely
    """
    from shapely import (
        LineString,
        MultiLineString,
        MultiPoint,
        MultiPolygon,
        Point,
        Polygon,
    )

    SHAPELY_TYPES = (
        Point,
        LineString,
        Polygon,
        MultiPoint,
        MultiLineString,
        MultiPolygon,
    )

    geom_var_names = [
        name
        for name, var in ds._variables.items()
        if var.dtype == "O" and isinstance(var.data.flat[0], SHAPELY_TYPES)
    ]
    if not geom_var_names:
        return ds

    if to_drop := set(geom_var_names) & set(ds._indexes):
        # e.g. xvec GeometryIndex
        ds = ds.drop_indexes(to_drop)

    if len(geom_var_names) > 1:
        raise NotImplementedError(
            "Multiple geometry variables are not supported at this time. "
            "Contributions to fix this are welcome. "
            f"Detected geometry variables are {geom_var_names!r}"
        )

    (name,) = geom_var_names
    variables = {}
    # If `name` is a dimension name, then we need to drop it. Otherwise we don't
    # So set errors="ignore"
    variables.update(
        shapely_to_cf(ds[name]).drop_vars(name, errors="ignore")._variables
    )

    geom_var = ds[name]

    more_updates = {}
    for varname, var in ds._variables.items():
        if varname == name:
            continue
        # TODO: this is incomplete. It works for vector data cubes where one of the geometry vars
        # is a dimension coordinate.
        if name in var.dims:
            var = var.copy()
            var._attrs = copy.deepcopy(var._attrs)
            var.attrs["geometry"] = GEOMETRY_CONTAINER_NAME
            # The grid_mapping and coordinates attributes can be carried by the geometry container
            # variable provided they are also carried by the data variables associated with the container.
            if to_add := geom_var.attrs.get("coordinates", ""):
                var.attrs["coordinates"] = var.attrs.get("coordinates", "") + to_add
        more_updates[varname] = var
    variables.update(more_updates)

    # WARNING: cf-xarray specific convention.
    # For vector data cubes, `name` is a dimension name.
    # By encoding to CF, we have
    # encoded the information in that variable across many different
    # variables (e.g. node_count) with `name` as a dimension.
    # We have to record `name` somewhere so that we reconstruct
    # a geometry variable of the right name at decode-time.
    variables[GEOMETRY_CONTAINER_NAME].attrs["variable_name"] = name

    encoded = xr.Dataset(variables)

    return encoded


def reshape_unique_geometries(
    ds: xr.Dataset,
    geom_var: str = "geometry",
    new_dim: str = "features",
) -> xr.Dataset:
    """Reshape a dataset containing a geometry variable so that all unique features are
    identified along a new dimension.

    This function only makes sense if the dimension of the geometry variable has no coordinate,
    or if that coordinate has repeated values for each geometry.

    Parameters
    ----------
    ds : xr.Dataset
        A Dataset.
    geom_var : string
        Name of the variable in `ds` that contains the geometry objects of type shapely.geometry.
        The variable must be 1D.
    new_dim : string
        Name of the new dimension in the returned object.

    Returns
    -------
    Dataset
        All variables sharing the dimension of `ds[geom_var]` are reshaped so that `new_dim`
        as a length equal to the number of unique geometries.
    """
    if ds[geom_var].ndim > 1:
        raise ValueError(
            f"The geometry variable must be 1D. Got ds[{geom_var}] with dims {ds[geom_var].dims}."
        )

    # Shapely objects are not hashable, thus np.unique cannot be used directly.
    # This trick is stolen from geopandas.
    _, unique_indexes, inv_indexes = np.unique(
        [g.wkb for g in ds[geom_var].values], return_index=True, return_inverse=True
    )
    old_name = ds[geom_var].dims[0]

    if old_name in ds.coords:
        old_values = ds[old_name].values
    else:
        # A dummy coord, a kind of counter, independent for each unique geometries
        old_values = np.array(
            [(inv_indexes[:i] == ind).sum() for i, ind in enumerate(inv_indexes)]
        )

    multi_index = pd.MultiIndex.from_arrays(
        (inv_indexes, old_values), names=(new_dim, old_name)
    )
    temp_name = "__temp_multi_index__"
    out = ds.rename({old_name: temp_name})
    out[temp_name] = multi_index
    out = out.unstack(temp_name)

    # geom_var was reshaped also, reconstruct it from the unique values.
    unique_indexes = xr.DataArray(unique_indexes, dims=(new_dim,))
    out[geom_var] = ds[geom_var].isel({old_name: unique_indexes})
    if old_name not in ds.coords:
        # If there was no coord before, drop the dummy one we made.
        out = out.drop_vars(old_name)  # type: ignore[arg-type,unused-ignore]  # Hashable/str stuff
    return out


def shapely_to_cf(geometries: xr.DataArray | Sequence, grid_mapping: str | None = None):
    """
    Convert a DataArray with shapely geometry objects into a CF-compliant dataset.

    Parameters
    ----------
    geometries : sequence of shapely geometries or xarray.DataArray
        A sequence of geometry objects or a Dataset with a "geometry" variable storing such geometries.
        All geometries must be of the same base type : Point, Line or Polygon, but multipart geometries are accepted.

    grid_mapping : str, optional
        A CF grid mapping name. When given, coordinates and attributes are named and set accordingly.
        Defaults to None, in which case the coordinates are simply names "crd_x" and "crd_y".

        .. warning::
            Only the `longitude_latitude` grid mapping is currently implemented.

    Returns
    -------
    xr.Dataset
        A dataset with shapely geometry objects translated into CF-compliant variables :
         - 'x', 'y' : the node coordinates
         - 'crd_x', 'crd_y' : the feature coordinates (might have different names if `grid_mapping` is available).
         - 'node_count' : The number of nodes per feature. Always present for Lines and Polygons. For Points: only present if there are multipart geometries.
         - 'part_node_count' : The number of nodes per individual geometry. Only for Lines with multipart geometries and for Polygons with multipart geometries or holes.
         - 'interior_ring' : Integer boolean indicating whether rings are interior or exterior. Only for Polygons with holes.
         - 'geometry_container' : Empty variable with attributes describing the geometry type.

    References
    ----------
    Please refer to the CF conventions document: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#geometries
    """

    if isinstance(geometries, xr.DataArray) and grid_mapping is not None:
        raise DeprecationWarning(
            "Explicitly passing `grid_mapping` with DataArray of geometries is deprecated. "
            "Please set a `grid_mapping` attribute on `geometries`, ",
            "and set the grid mapping variable as a coordinate",
        )

    # Get all types to call the appropriate translation function.
    types = {
        geom.item().geom_type if isinstance(geom, xr.DataArray) else geom.geom_type
        for geom in geometries
    }
    if types.issubset({"Point", "MultiPoint"}):
        ds = points_to_cf(geometries)
    elif types.issubset({"LineString", "MultiLineString"}):
        ds = lines_to_cf(geometries)
    elif types.issubset({"Polygon", "MultiPolygon"}):
        ds = polygons_to_cf(geometries)
    else:
        raise ValueError(
            f"Mixed geometry types are not supported in CF-compliant datasets. Got {types}"
        )

    ds[GEOMETRY_CONTAINER_NAME].attrs.update(coordinates="crd_x crd_y")

    if (
        grid_mapping is None
        and isinstance(geometries, xr.DataArray)
        and (grid_mapping_varname := geometries.attrs.get("grid_mapping"))
    ):
        if grid_mapping_varname in geometries.coords:
            grid_mapping = geometries.coords[grid_mapping_varname].attrs[
                "grid_mapping_name"
            ]
            for name_ in ["x", "y", "crd_x", "crd_y"]:
                ds[name_].attrs["grid_mapping"] = grid_mapping_varname

    # Special treatment of selected grid mappings
    if grid_mapping in ["latitude_longitude", "rotated_latitude_longitude"]:
        # Special case for longitude_latitude type grid mappings
        ds = ds.rename(crd_x="lon", crd_y="lat")
        if grid_mapping == "latitude_longitude":
            ds.lon.attrs.update(units="degrees_east", standard_name="longitude")
            ds.x.attrs.update(units="degrees_east", standard_name="longitude")
            ds.lat.attrs.update(units="degrees_north", standard_name="latitude")
            ds.y.attrs.update(units="degrees_north", standard_name="latitude")
        elif grid_mapping == "rotated_latitude_longitude":
            ds.lon.attrs.update(units="degrees", standard_name="grid_longitude")
            ds.x.attrs.update(units="degrees", standard_name="grid_longitude")
            ds.lat.attrs.update(units="degrees", standard_name="grid_latitude")
            ds.y.attrs.update(units="degrees", standard_name="grid_latitude")
        ds[GEOMETRY_CONTAINER_NAME].attrs.update(coordinates="lon lat")
    elif grid_mapping is not None:
        ds.crd_x.attrs.update(standard_name="projection_x_coordinate")
        ds.x.attrs.update(standard_name="projection_x_coordinate")
        ds.crd_y.attrs.update(standard_name="projection_y_coordinate")
        ds.y.attrs.update(standard_name="projection_y_coordinate")

    return ds


def cf_to_shapely(ds: xr.Dataset):
    """
    Convert geometries stored in a CF-compliant way to shapely objects stored in a single variable.

    Parameters
    ----------
    ds : xr.Dataset
        Must contain a ``geometry_container`` variable with attributes giving the geometry specifications.
        Must contain all variables needed to reconstruct the geometries listed in these specifications.

    Returns
    -------
    da: xr.DataArray
        A 1D DataArray of shapely objects.
        It has the same dimension as the ``node_count`` or the coordinates variables, or
        ``features`` if those were not present in ``ds``.

    References
    ----------
    Please refer to the CF conventions document: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#geometries
    """
    geom_type = ds[GEOMETRY_CONTAINER_NAME].attrs["geometry_type"]
    if geom_type == "point":
        geometries = cf_to_points(ds)
    elif geom_type == "line":
        geometries = cf_to_lines(ds)
    elif geom_type == "polygon":
        geometries = cf_to_polygons(ds)
    else:
        raise ValueError(
            f"Valid CF geometry types are 'point', 'line' and 'polygon'. Got {geom_type}"
        )

    return geometries.rename("geometry")


def points_to_cf(pts: xr.DataArray | Sequence):
    """Get a list of points (shapely.geometry.[Multi]Point) and return a CF-compliant geometry dataset.

    Parameters
    ----------
    pts : sequence of shapely.geometry.Point or MultiPoint
        The sequence of [multi]points to translate to a CF dataset.

    Returns
    -------
    xr.Dataset
        A Dataset with variables 'x', 'y', 'crd_x', 'crd_y', 'node_count' and 'geometry_container'.
        The coordinates of MultiPoint instances are their first point.
    """
    from shapely.geometry import MultiPoint

    if isinstance(pts, xr.DataArray):
        dim = pts.dims[0]
        coord = pts[dim] if dim in pts.coords else None
        pts_ = pts.values.tolist()
    else:
        dim = "features"
        coord = None
        pts_ = pts

    x, y, node_count, crdX, crdY = [], [], [], [], []
    for pt in pts_:
        if isinstance(pt, MultiPoint):
            xy = np.concatenate([p.coords for p in pt.geoms])
        else:
            xy = np.atleast_2d(pt.coords)
        x.extend(xy[:, 0])
        y.extend(xy[:, 1])
        node_count.append(xy.shape[0])
        crdX.append(xy[0, 0])
        crdY.append(xy[0, 1])

    ds = xr.Dataset(
        data_vars={
            "node_count": xr.DataArray(node_count, dims=(dim,)),
            "geometry_container": xr.DataArray(
                attrs={
                    "geometry_type": "point",
                    "node_count": "node_count",
                    "node_coordinates": "x y",
                    "coordinates": "crd_x crd_y",
                }
            ),
        },
        coords={
            "x": xr.DataArray(x, dims=("node",), attrs={"axis": "X"}),
            "y": xr.DataArray(y, dims=("node",), attrs={"axis": "Y"}),
            "crd_x": xr.DataArray(crdX, dims=(dim,), attrs={"nodes": "x"}),
            "crd_y": xr.DataArray(crdY, dims=(dim,), attrs={"nodes": "y"}),
        },
    )

    if coord is not None:
        ds = ds.assign_coords({dim: coord})

    # Special case when we have no MultiPoints
    if (ds.node_count == 1).all():
        ds = ds.drop_vars("node_count")
        del ds[GEOMETRY_CONTAINER_NAME].attrs["node_count"]
    return ds


def cf_to_points(ds: xr.Dataset):
    """Convert point geometries stored in a CF-compliant way to shapely points stored in a single variable.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with CF-compliant point geometries.
        Must have a "geometry_container" variable with at least a 'node_coordinates' attribute.
        Must also have the two 1D variables listed by this attribute.

    Returns
    -------
    geometry : xr.DataArray
        A 1D array of shapely.geometry.[Multi]Point objects.
        It has the same dimension as the ``node_count`` or the coordinates variables, or
        ``'features'`` if those were not present in ``ds``.
    """
    from shapely.geometry import MultiPoint, Point

    # Shorthand for convenience
    geo = ds[GEOMETRY_CONTAINER_NAME].attrs

    # The features dimension name, defaults to the one of 'node_count' or the dimension of the coordinates, if present.
    feat_dim = None
    if "coordinates" in geo and feat_dim is None:
        xcoord_name, _ = geo["coordinates"].split(" ")
        (feat_dim,) = ds[xcoord_name].dims

    x_name, y_name = ds[GEOMETRY_CONTAINER_NAME].attrs["node_coordinates"].split(" ")
    xy = np.stack([ds[x_name].values, ds[y_name].values], axis=-1)

    node_count_name = ds[GEOMETRY_CONTAINER_NAME].attrs.get("node_count")
    if node_count_name is None:
        # No node_count means all geometries are single points (node_count = 1)
        # And if we had no coordinates, then the dimension defaults to "features"
        feat_dim = feat_dim or "features"
        node_count = xr.DataArray([1] * xy.shape[0], dims=(feat_dim,))
        if feat_dim in ds.coords:
            node_count = node_count.assign_coords({feat_dim: ds[feat_dim]})
    else:
        node_count = ds[node_count_name]

    j = 0  # The index of the first node.
    geoms = np.empty(node_count.shape, dtype=object)
    # i is the feature index, n its number of nodes
    for i, n in enumerate(node_count.values):
        if n == 1:
            geoms[i] = Point(xy[j, :])
        else:
            geoms[i] = MultiPoint(xy[j : j + n, :])
        j += n

    return xr.DataArray(geoms, dims=node_count.dims, coords=node_count.coords)


def lines_to_cf(lines: xr.DataArray | Sequence):
    """Convert an iterable of lines (shapely.geometry.[Multi]Line) into a CF-compliant geometry dataset.

    Parameters
    ----------
    lines : sequence of shapely.geometry.Line or MultiLine
        The sequence of [multi]lines to translate to a CF dataset.

    Returns
    -------
    xr.Dataset
        A Dataset with variables 'x', 'y', 'crd_x', 'crd_y', 'node_count' and 'geometry_container'
        and optionally 'part_node_count'.
    """
    from shapely import to_ragged_array

    if isinstance(lines, xr.DataArray):
        dim = lines.dims[0]
        coord = lines[dim] if dim in lines.coords else None
        lines_ = lines.values
    else:
        dim = "index"
        coord = None
        lines_ = np.array(lines)

    _, arr, offsets = to_ragged_array(lines_)
    x = arr[:, 0]
    y = arr[:, 1]

    part_node_count = np.diff(offsets[0])
    if len(offsets) == 1:
        indices = offsets[0]
        node_count = part_node_count
    else:
        indices = np.take(offsets[0], offsets[1])
        node_count = np.diff(indices)

    geom_coords = arr.take(indices[:-1], 0)
    crdX = geom_coords[:, 0]
    crdY = geom_coords[:, 1]

    ds = xr.Dataset(
        data_vars={
            "node_count": xr.DataArray(node_count, dims=(dim,)),
            "part_node_count": xr.DataArray(part_node_count, dims=("part",)),
            "geometry_container": xr.DataArray(
                attrs={
                    "geometry_type": "line",
                    "node_count": "node_count",
                    "part_node_count": "part_node_count",
                    "node_coordinates": "x y",
                    "coordinates": "crd_x crd_y",
                }
            ),
        },
        coords={
            "x": xr.DataArray(x, dims=("node",), attrs={"axis": "X"}),
            "y": xr.DataArray(y, dims=("node",), attrs={"axis": "Y"}),
            "crd_x": xr.DataArray(crdX, dims=(dim,), attrs={"nodes": "x"}),
            "crd_y": xr.DataArray(crdY, dims=(dim,), attrs={"nodes": "y"}),
        },
    )

    if coord is not None:
        ds = ds.assign_coords({dim: coord})

    # Special case when we have no MultiLines
    if len(ds.part_node_count) == len(ds.node_count):
        ds = ds.drop_vars("part_node_count")
        del ds[GEOMETRY_CONTAINER_NAME].attrs["part_node_count"]
    return ds


def cf_to_lines(ds: xr.Dataset):
    """Convert line geometries stored in a CF-compliant way to shapely lines stored in a single variable.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with CF-compliant line geometries.
        Must have a "geometry_container" variable with at least a 'node_coordinates' attribute.
        Must also have the two 1D variables listed by this attribute.

    Returns
    -------
    geometry : xr.DataArray
        A 1D array of shapely.geometry.[Multi]Line objects.
        It has the same dimension as the ``part_node_count`` or the coordinates variables, or
        ``'features'`` if those were not present in ``ds``.
    """
    from shapely import GeometryType, from_ragged_array

    # Shorthand for convenience
    geo = ds[GEOMETRY_CONTAINER_NAME].attrs

    # The features dimension name, defaults to the one of 'node_count'
    # or the dimension of the coordinates, if present.
    feat_dim = None
    if "coordinates" in geo:
        xcoord_name, _ = geo["coordinates"].split(" ")
        (feat_dim,) = ds[xcoord_name].dims

    x_name, y_name = geo["node_coordinates"].split(" ")
    xy = np.stack([ds[x_name].values, ds[y_name].values], axis=-1)

    node_count_name = geo.get("node_count")
    part_node_count_name = geo.get("part_node_count", node_count_name)
    if node_count_name is None:
        raise ValueError("'node_count' must be provided for line geometries")
    else:
        node_count = ds[node_count_name]
        feat_dim = feat_dim or "index"
        if feat_dim in ds.coords:
            node_count = node_count.assign_coords({feat_dim: ds[feat_dim]})

    # first get geometries for all the parts
    part_node_count = ds[part_node_count_name]
    offset1 = np.insert(np.cumsum(part_node_count.values), 0, 0)
    lines = from_ragged_array(GeometryType.LINESTRING, xy, offsets=(offset1,))

    # get index of offset2 values that are edges for part_node_count
    offset2 = np.nonzero(np.isin(offset1, np.insert(np.cumsum(node_count), 0, 0)))[0]

    multilines = from_ragged_array(
        GeometryType.MULTILINESTRING, xy, offsets=(offset1, offset2)
    )

    # get items from lines or multilines depending on number of parts
    geoms = np.where(np.diff(offset2) == 1, lines[offset2[:-1]], multilines)

    return xr.DataArray(geoms, dims=node_count.dims, coords=node_count.coords)


def polygons_to_cf(polygons: xr.DataArray | Sequence):
    """Convert an iterable of polygons (shapely.geometry.[Multi]Polygon) into a CF-compliant geometry dataset.

    Parameters
    ----------
    polygons : sequence of shapely.geometry.Polygon or MultiPolygon
        The sequence of [multi]polygons to translate to a CF dataset.

    Returns
    -------
    xr.Dataset
        A Dataset with variables 'x', 'y', 'crd_x', 'crd_y', 'node_count' and 'geometry_container'
        and optionally 'part_node_count'.
    """
    from shapely import to_ragged_array

    if isinstance(polygons, xr.DataArray):
        dim = polygons.dims[0]
        coord = polygons[dim] if dim in polygons.coords else None
        polygons_ = polygons.values
    else:
        dim = "index"
        coord = None
        polygons_ = np.array(polygons)

    _, arr, offsets = to_ragged_array(polygons_)
    x = arr[:, 0]
    y = arr[:, 1]

    part_node_count = np.diff(offsets[0])
    if len(offsets) == 1:
        indices = offsets[0]
        node_count = part_node_count
    elif len(offsets) >= 2:
        indices = np.take(offsets[0], offsets[1])
        interior_ring = np.isin(offsets[0], indices, invert=True)[:-1].astype(int)

        if len(offsets) == 3:
            indices = np.take(indices, offsets[2])

        node_count = np.diff(indices)

    geom_coords = arr.take(indices[:-1], 0)
    crdX = geom_coords[:, 0]
    crdY = geom_coords[:, 1]

    ds = xr.Dataset(
        data_vars={
            "node_count": xr.DataArray(node_count, dims=(dim,)),
            "interior_ring": xr.DataArray(interior_ring, dims=("part",)),
            "part_node_count": xr.DataArray(part_node_count, dims=("part",)),
            "geometry_container": xr.DataArray(
                attrs={
                    "geometry_type": "polygon",
                    "node_count": "node_count",
                    "part_node_count": "part_node_count",
                    "interior_ring": "interior_ring",
                    "node_coordinates": "x y",
                    "coordinates": "crd_x crd_y",
                }
            ),
        },
        coords={
            "x": xr.DataArray(x, dims=("node",), attrs={"axis": "X"}),
            "y": xr.DataArray(y, dims=("node",), attrs={"axis": "Y"}),
            "crd_x": xr.DataArray(crdX, dims=(dim,), attrs={"nodes": "x"}),
            "crd_y": xr.DataArray(crdY, dims=(dim,), attrs={"nodes": "y"}),
        },
    )

    if coord is not None:
        ds = ds.assign_coords({dim: coord})

    # Special case when we have no MultiPolygons and no holes
    if len(ds.part_node_count) == len(ds.node_count):
        ds = ds.drop_vars("part_node_count")
        del ds[GEOMETRY_CONTAINER_NAME].attrs["part_node_count"]

    # Special case when we have no holes
    if (ds.interior_ring == 0).all():
        ds = ds.drop_vars("interior_ring")
        del ds[GEOMETRY_CONTAINER_NAME].attrs["interior_ring"]
    return ds


def cf_to_polygons(ds: xr.Dataset):
    """Convert polygon geometries stored in a CF-compliant way to shapely polygons stored in a single variable.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with CF-compliant polygon geometries.
        Must have a "geometry_container" variable with at least a 'node_coordinates' attribute.
        Must also have the two 1D variables listed by this attribute.

    Returns
    -------
    geometry : xr.DataArray
        A 1D array of shapely.geometry.[Multi]Polygon objects.
        It has the same dimension as the ``part_node_count`` or the coordinates variables, or
        ``'features'`` if those were not present in ``ds``.
    """
    from shapely import GeometryType, from_ragged_array

    # Shorthand for convenience
    geo = ds[GEOMETRY_CONTAINER_NAME].attrs

    # The features dimension name, defaults to the one of 'part_node_count'
    # or the dimension of the coordinates, if present.
    feat_dim = None
    if "coordinates" in geo:
        xcoord_name, _ = geo["coordinates"].split(" ")
        (feat_dim,) = ds[xcoord_name].dims

    x_name, y_name = geo["node_coordinates"].split(" ")
    xy = np.stack([ds[x_name].values, ds[y_name].values], axis=-1)

    node_count_name = geo.get("node_count")
    part_node_count_name = geo.get("part_node_count", node_count_name)
    interior_ring_name = geo.get("interior_ring")

    if node_count_name is None:
        raise ValueError("'node_count' must be provided for polygon geometries")
    else:
        node_count = ds[node_count_name]
        feat_dim = feat_dim or "index"
        if feat_dim in ds.coords:
            node_count = node_count.assign_coords({feat_dim: ds[feat_dim]})

    # first get geometries for all the rings
    part_node_count = ds[part_node_count_name]
    offset1 = np.insert(np.cumsum(part_node_count.values), 0, 0)

    if interior_ring_name is None:
        offset2 = np.array(list(range(len(offset1))))
    else:
        interior_ring = ds[interior_ring_name]
        if not interior_ring[0] == 0:
            raise ValueError("coordinate array must start with an exterior ring")
        offset2 = np.append(np.where(interior_ring == 0)[0], [len(part_node_count)])

    polygons = from_ragged_array(GeometryType.POLYGON, xy, offsets=(offset1, offset2))

    # get index of offset2 values that are edges for node_count
    offset3 = np.nonzero(
        np.isin(
            offset2,
            np.nonzero(np.isin(offset1, np.insert(np.cumsum(node_count), 0, 0)))[0],
        )
    )[0]
    multipolygons = from_ragged_array(
        GeometryType.MULTIPOLYGON, xy, offsets=(offset1, offset2, offset3)
    )

    # get items from polygons or multipolygons depending on number of parts
    geoms = np.where(np.diff(offset3) == 1, polygons[offset3[:-1]], multipolygons)

    return xr.DataArray(geoms, dims=node_count.dims, coords=node_count.coords)
