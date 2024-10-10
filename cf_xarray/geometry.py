from __future__ import annotations

import copy
from collections import ChainMap
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
import xarray as xr
from numpy.typing import ArrayLike

GEOMETRY_CONTAINER_NAME = "geometry_container"
FEATURES_DIM_NAME = "features"

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


@dataclass
class GeometryNames:
    """Helper class to ease handling of all the variable names needed for CF geometries."""

    def __init__(
        self,
        suffix: str = "",
        grid_mapping_name: str | None = None,
        grid_mapping: str | None = None,
    ):
        self.container_name: str = GEOMETRY_CONTAINER_NAME + suffix
        self.node_dim: str = "node" + suffix
        self.node_count: str = "node_count" + suffix
        self.node_coordinates_x: str = "x" + suffix
        self.node_coordinates_y: str = "y" + suffix
        self.coordinates_x: str = "crd_x" + suffix
        self.coordinates_y: str = "crd_y" + suffix
        self.part_node_count: str = "part_node_count" + suffix
        self.part_dim: str = "part" + suffix
        self.interior_ring: str = "interior_ring" + suffix
        self.attrs_x: dict[str, str] = {}
        self.attrs_y: dict[str, str] = {}
        self.grid_mapping_attr = {"grid_mapping": grid_mapping} if grid_mapping else {}

        # Special treatment of selected grid mappings
        if grid_mapping_name in ["latitude_longitude", "rotated_latitude_longitude"]:
            # Special case for longitude_latitude type grid mappings
            self.coordinates_x = "lon"
            self.coordinates_y = "lat"
            if grid_mapping_name == "latitude_longitude":
                self.attrs_x = dict(units="degrees_east", standard_name="longitude")
                self.attrs_y = dict(units="degrees_north", standard_name="latitude")
            elif grid_mapping_name == "rotated_latitude_longitude":
                self.attrs_x = dict(
                    units="degrees_east", standard_name="grid_longitude"
                )
                self.attrs_y = dict(
                    units="degrees_north", standard_name="grid_latitude"
                )
        elif grid_mapping_name is not None:
            self.attrs_x = dict(standard_name="projection_x_coordinate")
            self.attrs_y = dict(standard_name="projection_y_coordinate")
        self.attrs_x.update(self.grid_mapping_attr)
        self.attrs_y.update(self.grid_mapping_attr)

    @property
    def geometry_container_attrs(self) -> dict[str, str]:
        return {
            "node_count": self.node_count,
            "node_coordinates": f"{self.node_coordinates_x} {self.node_coordinates_y}",
            "coordinates": f"{self.coordinates_x} {self.coordinates_y}",
            **self.grid_mapping_attr,
        }

    def coords(
        self,
        *,
        dim: Hashable,
        x: ArrayLike,
        y: ArrayLike,
        crdX: ArrayLike | None = None,
        crdY: ArrayLike | None = None,
    ) -> dict[str, xr.DataArray]:
        """
        Construct coordinate DataArrays for the numpy data (x, y, crdX, crdY)

        Parameters
        ----------
        x: array
            Node coordinates for X coordinate
        y: array
            Node coordinates for Y coordinate
        crdX: array, optional
            Nominal X coordinate
        crdY: array, optional
            Nominal X coordinate
        """
        mapping = {
            self.node_coordinates_x: xr.DataArray(
                x, dims=self.node_dim, attrs={"axis": "X", **self.attrs_x}
            ),
            self.node_coordinates_y: xr.DataArray(
                y, dims=self.node_dim, attrs={"axis": "Y", **self.attrs_y}
            ),
        }
        if crdX is not None:
            mapping[self.coordinates_x] = xr.DataArray(
                crdX,
                dims=(dim,),
                attrs={"nodes": self.node_coordinates_x, **self.attrs_x},
            )
        if crdY is not None:
            mapping[self.coordinates_y] = xr.DataArray(
                crdY,
                dims=(dim,),
                attrs={"nodes": self.node_coordinates_y, **self.attrs_y},
            )
        return mapping


def _assert_single_geometry_container(ds: xr.Dataset) -> Hashable:
    container_names = _get_geometry_containers(ds)
    if len(container_names) > 1:
        raise ValueError(
            "Only one geometry container is supported by cf_to_points. "
            "To handle multiple geometries use `decode_geometries` instead."
        )
    (container_name,) = container_names
    return container_name


def _get_geometry_containers(obj: xr.DataArray | xr.Dataset) -> list[Hashable]:
    """
    Translate from key (either CF key or variable name) to its bounds' variable names.

    This function interprets the ``geometry`` attribute on DataArrays.

    Parameters
    ----------
    obj : DataArray, Dataset
        DataArray belonging to the coordinate to be checked

    Returns
    -------
    List[str]
        Variable name(s) in parent xarray object that are bounds of `key`
    """

    if isinstance(obj, xr.DataArray):
        obj = obj._to_temp_dataset()
    variables = obj._variables

    results = set()
    for name, var in variables.items():
        attrs_or_encoding = ChainMap(var.attrs, var.encoding)
        if "geometry_type" in attrs_or_encoding:
            results.update([name])
    return list(results)


def decode_geometries(encoded: xr.Dataset) -> xr.Dataset:
    """
    Decode CF encoded geometries to numpy object arrays containing shapely geometries.

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

    containers = _get_geometry_containers(encoded)
    if not containers:
        raise NotImplementedError(
            "No geometry container variables detected, none of the provided variables "
            "have a `geometry_type` attribute."
        )

    todrop: list[Hashable] = []
    decoded = xr.Dataset()
    for container_name in containers:
        enc_geom_var = encoded[container_name]
        geom_attrs = enc_geom_var.attrs

        # Grab the coordinates attribute
        geom_attrs.update(enc_geom_var.encoding)

        geom_var = cf_to_shapely(encoded, container=container_name).variable

        todrop.extend(
            (container_name,)
            + tuple(
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
        )

        name = geom_attrs.get("variable_name", None)
        if name in encoded.dims:
            decoded = decoded.assign_coords(
                xr.Coordinates(coords={name: geom_var}, indexes={})
            )
        else:
            decoded[name] = geom_var

    decoded.update(encoded.drop_vars(todrop))

    # Is this a good idea? We are deleting information.
    # OTOH we have decoded it to a useful in-memory representation
    for var in decoded._variables.values():
        if var.attrs.get("geometry") in containers:
            var.attrs.pop("geometry")
    return decoded


def encode_geometries(ds: xr.Dataset):
    """
    Encode any discovered geometry variables using the CF conventions.

    Practically speaking, geometry variables are numpy object arrays where the first
    element is a shapely geometry.

    Parameters
    ----------
    ds : Dataset
       Dataset containing at least one geometry variable.

    Returns
    -------
    Dataset
       Where all geometry variables are encoded. The information in a single geometry
       variable in the input is split across multiple variables in the returned Dataset
       following the CF conventions.

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

    variables = {}
    for name in geom_var_names:
        # TODO: do we prefer this choice be invariant to number of geometry variables
        suffix = "_" + str(name) if len(geom_var_names) > 1 else ""
        container_name = GEOMETRY_CONTAINER_NAME + suffix
        # If `name` is a dimension name, then we need to drop it. Otherwise we don't
        # So set errors="ignore"
        variables.update(
            shapely_to_cf(ds[name], suffix=suffix)
            .drop_vars(name, errors="ignore")
            ._variables
        )

        geom_var = ds[name]
        more_updates = {}
        for varname, var in ds._variables.items():
            if varname == name:
                continue
            # TODO: this is incomplete. It works for vector data cubes where one of the geometry vars
            # is a dimension coordinate.
            if name in var.dims:
                var = var.copy(deep=False)
                var._attrs = copy.deepcopy(var._attrs)
                var.attrs["geometry"] = container_name
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
        variables[container_name].attrs["variable_name"] = name

    encoded = xr.Dataset(variables).set_coords(
        set(ds._coord_names) - set(geom_var_names)
    )

    return encoded


def reshape_unique_geometries(
    ds: xr.Dataset,
    geom_var: str = "geometry",
    new_dim: str = FEATURES_DIM_NAME,
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


def shapely_to_cf(
    geometries: xr.DataArray | Sequence,
    grid_mapping: str | None = None,
    *,
    suffix: str = "",
):
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

    container_name: str, optional
        Name for the "geometry container" scalar variable in the encoded Dataset

    Returns
    -------
    xr.Dataset
        A dataset with shapely geometry objects translated into CF-compliant variables :
         - 'x', 'y' : the node coordinates
         - 'crd_x', 'crd_y' : the feature coordinates (might have different names if `grid_mapping` is available).
         - 'node_count' : The number of nodes per feature. Always present for Lines and Polygons. For Points: only present if there are multipart geometries.
         - 'part_node_count' : The number of nodes per individual geometry. Only for Lines with multipart geometries and for Polygons with multipart geometries or holes.
         - 'interior_ring' : Integer boolean indicating whether rings are interior or exterior. Only for Polygons with holes.
         - container_name : Empty variable with attributes describing the geometry type.

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

    as_data = geometries.data if isinstance(geometries, xr.DataArray) else geometries
    type_ = as_data[0].geom_type

    grid_mapping_varname = None
    if (
        grid_mapping is None
        and isinstance(geometries, xr.DataArray)
        and (grid_mapping_varname := geometries.attrs.get("grid_mapping"))
    ):
        if grid_mapping_varname in geometries.coords:
            # Not all CRS can be encoded in CF
            grid_mapping = geometries.coords[grid_mapping_varname].attrs.get(
                "grid_mapping_name", None
            )

    # TODO: consider accepting a GeometryNames instance from the user instead
    names = GeometryNames(
        suffix=suffix, grid_mapping_name=grid_mapping, grid_mapping=grid_mapping_varname
    )

    try:
        if type_ in ["Point", "MultiPoint"]:
            ds = points_to_cf(geometries, names=names)
        elif type_ in ["LineString", "MultiLineString"]:
            ds = lines_to_cf(geometries, names=names)
        elif type_ in ["Polygon", "MultiPolygon"]:
            ds = polygons_to_cf(geometries, names=names)
        else:
            raise ValueError(
                f"This geometry type is not supported in CF-compliant datasets. Got {type_}"
            )
    except NotImplementedError as e:
        raise ValueError(
            "Error converting geometries. Possibly you have provided mixed geometry types."
        ) from e

    return ds


def cf_to_shapely(ds: xr.Dataset, *, container: Hashable = GEOMETRY_CONTAINER_NAME):
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
    if container not in ds._variables:
        raise ValueError(
            f"{container!r} is not the name of a variable in the provided Dataset."
        )
    if not (geom_type := ds[container].attrs.get("geometry_type", None)):
        raise ValueError(
            f"{container!r} is not the name of a valid geometry variable. "
            "It does not have a `geometry_type` attribute."
        )

    # Extract all necessary geometry variables
    subds = ds.cf[[container]]
    if geom_type == "point":
        geometries = cf_to_points(subds)
    elif geom_type == "line":
        geometries = cf_to_lines(subds)
    elif geom_type == "polygon":
        geometries = cf_to_polygons(subds)
    else:
        raise ValueError(
            f"Valid CF geometry types are 'point', 'line' and 'polygon'. Got {geom_type}"
        )
    if gm := ds[container].attrs.get("grid_mapping"):
        geometries.attrs["grid_mapping"] = gm

    return geometries.rename("geometry")


def points_to_cf(pts: xr.DataArray | Sequence, *, names: GeometryNames | None = None):
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
        # TODO: Fix this hardcoding
        if pts.ndim != 1:
            raise ValueError("Only 1D DataArrays are supported.")
        dim = pts.dims[0]
        coord = pts[dim] if dim in pts.coords else None
        pts_ = pts.values.tolist()
    else:
        dim = FEATURES_DIM_NAME
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

    if names is None:
        names = GeometryNames()

    ds = xr.Dataset(
        data_vars={
            names.node_count: xr.DataArray(node_count, dims=(dim,)),
            names.container_name: xr.DataArray(
                data=np.nan,
                attrs={"geometry_type": "point", **names.geometry_container_attrs},
            ),
        },
        coords=names.coords(x=x, y=y, crdX=crdX, crdY=crdY, dim=dim),
    )

    if coord is not None:
        ds = ds.assign_coords({dim: coord})

    # Special case when we have no MultiPoints
    if (ds[names.node_count] == 1).data.all():
        ds = ds.drop_vars(names.node_count)
        del ds[names.container_name].attrs["node_count"]
    return ds


def cf_to_points(ds: xr.Dataset):
    """Convert point geometries stored in a CF-compliant way to shapely points stored in a single variable.

    Parameters
    ----------
    ds : xr.Dataset
        A dataset with CF-compliant point geometries.
        Must have a *single* "geometry container" variable with at least a 'node_coordinates' attribute.
        Must also have the two 1D variables listed by this attribute.

    Returns
    -------
    geometry : xr.DataArray
        A 1D array of shapely.geometry.[Multi]Point objects.
        It has the same dimension as the ``node_count`` or the coordinates variables, or
        ``'features'`` if those were not present in ``ds``.
    """
    from shapely.geometry import MultiPoint, Point

    container_name = _assert_single_geometry_container(ds)
    # Shorthand for convenience
    geo = ds[container_name].attrs

    # The features dimension name, defaults to the one of 'node_count' or the dimension of the coordinates, if present.
    feat_dim = None
    if "coordinates" in geo and feat_dim is None:
        xcoord_name, _ = geo["coordinates"].split(" ")
        (feat_dim,) = ds[xcoord_name].dims

    x_name, y_name = ds[container_name].attrs["node_coordinates"].split(" ")
    xy = np.stack([ds[x_name].values, ds[y_name].values], axis=-1)

    node_count_name = ds[container_name].attrs.get("node_count")
    if node_count_name is None:
        # No node_count means all geometries are single points (node_count = 1)
        # And if we had no coordinates, then the dimension defaults to FEATURES_DIM_NAME
        feat_dim = feat_dim or FEATURES_DIM_NAME
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

    da = xr.DataArray(geoms, dims=node_count.dims, coords=node_count.coords)
    if node_count_name:
        del da[node_count_name]
    return da


def lines_to_cf(lines: xr.DataArray | Sequence, *, names: GeometryNames | None = None):
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

    if names is None:
        names = GeometryNames()

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
            names.node_count: xr.DataArray(node_count, dims=(dim,)),
            names.container_name: xr.DataArray(
                data=np.nan,
                attrs={"geometry_type": "line", **names.geometry_container_attrs},
            ),
        },
        coords=names.coords(x=x, y=y, crdX=crdX, crdY=crdY, dim=dim),
    )

    if coord is not None:
        ds = ds.assign_coords({dim: coord})

    # Special case when we have no MultiLines
    if len(part_node_count) != len(node_count):
        ds[names.part_node_count] = xr.DataArray(part_node_count, dims=names.part_dim)
        ds[names.container_name].attrs["part_node_count"] = names.part_node_count

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

    container_name = _assert_single_geometry_container(ds)

    # Shorthand for convenience
    geo = ds[container_name].attrs

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

    return xr.DataArray(
        geoms, dims=node_count.dims, coords=node_count.coords
    ).drop_vars(node_count_name)


def polygons_to_cf(
    polygons: xr.DataArray | Sequence, *, names: GeometryNames | None = None
):
    """Convert an iterable of polygons (shapely.geometry.[Multi]Polygon) into a CF-compliant geometry dataset.

    Parameters
    ----------
    polygons : sequence of shapely.geometry.Polygon or MultiPolygon
        The sequence of [multi]polygons to translate to a CF dataset.

    names: GeometryNames, optional
       Structure that helps manipulate geometry attrs.

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

    if names is None:
        names = GeometryNames()

    _, arr, offsets = to_ragged_array(polygons_)
    x = arr[:, 0]
    y = arr[:, 1]

    part_node_count = np.diff(offsets[0])
    if len(offsets) == 1:
        indices = offsets[0]
        node_count = part_node_count
    elif len(offsets) >= 2:
        indices = np.take(offsets[0], offsets[1])
        interior_ring = np.isin(offsets[0], indices, invert=True)[:-1]

        if len(offsets) == 3:
            indices = np.take(indices, offsets[2])

        node_count = np.diff(indices)

    geom_coords = arr.take(indices[:-1], 0)
    crdX = geom_coords[:, 0]
    crdY = geom_coords[:, 1]

    data_vars = {names.node_count: (dim, node_count)}
    geometry_attrs = names.geometry_container_attrs

    # Special case when we have no MultiPolygons and no holes
    if len(part_node_count) != len(node_count):
        data_vars[names.part_node_count] = (names.part_dim, part_node_count)
        geometry_attrs["part_node_count"] = names.part_node_count

    # Special case when we have no holes
    if interior_ring.any():
        data_vars[names.interior_ring] = (names.part_dim, interior_ring)
        geometry_attrs["interior_ring"] = names.interior_ring

    data_vars[names.container_name] = (  # type: ignore[assignment]
        (),
        np.nan,
        {"geometry_type": "polygon", **geometry_attrs},
    )
    ds = xr.Dataset(
        data_vars=data_vars,
        coords=names.coords(x=x, y=y, crdX=crdX, crdY=crdY, dim=dim),
    )

    if coord is not None:
        ds = ds.assign_coords({dim: coord})

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

    container_name = _assert_single_geometry_container(ds)

    # Shorthand for convenience
    geo = ds[container_name].attrs

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

    return xr.DataArray(
        geoms, dims=node_count.dims, coords=node_count.coords
    ).drop_vars(node_count_name)
