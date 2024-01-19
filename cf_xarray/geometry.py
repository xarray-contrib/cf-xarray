from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import xarray as xr


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
        out = out.drop_vars(old_name)
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
         - 'node_count' : The number of nodes per feature. Absent if all instances are Points.
         - 'geometry_container' : Empty variable with attributes describing the geometry type.
         - Other variables are not implemented as only Points are currently understood.

    References
    ----------
    Please refer to the CF conventions document: http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#geometries
    """
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

    # Special treatment of selected grid mappings
    if grid_mapping == "longitude_latitude":
        # Special case for longitude_latitude grid mapping
        ds = ds.rename(crd_x="lon", crd_y="lat")
        ds.lon.attrs.update(units="degrees_east", standard_name="longitude")
        ds.lat.attrs.update(units="degrees_north", standard_name="latitude")
        ds.geometry_container.attrs.update(coordinates="lon lat")
        ds.x.attrs.update(units="degrees_east", standard_name="longitude")
        ds.y.attrs.update(units="degrees_north", standard_name="latitude")
    elif grid_mapping is not None:
        raise NotImplementedError(
            f"Only grid mapping longitude_latitude is implemented. Got {grid_mapping}."
        )

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
    geom_type = ds.geometry_container.attrs["geometry_type"]
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
        del ds.geometry_container.attrs["node_count"]
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
    geo = ds.geometry_container.attrs

    # The features dimension name, defaults to the one of 'node_count' or the dimension of the coordinates, if present.
    feat_dim = None
    if "coordinates" in geo and feat_dim is None:
        xcoord_name, _ = geo["coordinates"].split(" ")
        (feat_dim,) = ds[xcoord_name].dims

    x_name, y_name = ds.geometry_container.attrs["node_coordinates"].split(" ")
    xy = np.stack([ds[x_name].values, ds[y_name].values], axis=-1)

    node_count_name = ds.geometry_container.attrs.get("node_count")
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
        del ds.geometry_container.attrs["part_node_count"]
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
    geo = ds.geometry_container.attrs

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
        del ds.geometry_container.attrs["part_node_count"]

    # Special case when we have no holes
    if (ds.interior_ring == 0).all():
        ds = ds.drop_vars("interior_ring")
        del ds.geometry_container.attrs["interior_ring"]
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
    geo = ds.geometry_container.attrs

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
