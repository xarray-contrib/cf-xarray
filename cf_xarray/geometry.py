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
    """Convert a DataArray with shapely geometry objects into a CF-compliant dataset.

    .. warning::
        Only point geometries are currently implemented.

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
    elif types.issubset({"Polygon", "MultiPolygon"}) or types.issubset(
        {"LineString", "MultiLineString"}
    ):
        raise NotImplementedError("Only point geometries conversion is implemented.")
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
    """Convert geometries stored in a CF-compliant way to shapely objects stored in a single variable.

    .. warning::
        Only point geometries are currently implemented.

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
    elif geom_type in ["line", "polygon"]:
        raise NotImplementedError("Only point geometries conversion is implemented.")
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


def grid_to_polygons(ds: xr.Dataset) -> xr.DataArray:
    """
    Converts a regular 2D lat/lon grid to a 2D array of shapely polygons.

    Modified from https://notebooksharing.space/view/c6c1f3a7d0c260724115eaa2bf78f3738b275f7f633c1558639e7bbd75b31456.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with "latitude" and "longitude" variables as well as their bounds variables.
        1D "latitude" and "longitude" variables are supported. This function will automatically
        broadcast them against each other.

    Returns
    -------
    DataArray
        DataArray with shapely polygon per grid cell.
    """
    import shapely

    grid = ds.cf[["latitude", "longitude"]].load()
    bounds = grid.cf.bounds
    dims = grid.cf.dims

    if "latitude" in dims or "longitude" in dims:
        # for 1D lat, lon, this allows them to be
        # broadcast against each other
        grid = grid.reset_coords()

    assert "latitude" in bounds
    assert "longitude" in bounds
    (lon_bounds,) = bounds["longitude"]
    (lat_bounds,) = bounds["latitude"]

    with xr.set_options(keep_attrs=True):
        (points,) = xr.broadcast(grid)

    bounds_dim = grid.cf.get_bounds_dim_name("latitude")
    points = points.transpose(..., bounds_dim)
    lonbnd = points[lon_bounds].data
    latbnd = points[lat_bounds].data

    if points.sizes[bounds_dim] == 2:
        lonbnd = lonbnd[..., [0, 0, 1, 1]]
        latbnd = latbnd[..., [0, 1, 1, 0]]

    elif points.sizes[bounds_dim] != 4:
        raise ValueError(
            f"The size of the detected bounds or vertex dimension {bounds_dim} is not 2 or 4."
        )

    # geopandas needs this
    mask = lonbnd[..., 0] >= 180
    lonbnd[mask, :] = lonbnd[mask, :] - 360

    polyarray = shapely.polygons(shapely.linearrings(lonbnd, latbnd))

    # 'geometry' is a blessed name in geopandas.
    boxes = points[lon_bounds][..., 0].copy(data=polyarray).rename("geometry")

    return boxes
