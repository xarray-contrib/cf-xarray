from __future__ import annotations

from collections.abc import Hashable, Sequence

import numpy as np
import xarray as xr
from xarray import DataArray


def _guess_bounds_1d(da, dim):
    """
    Guess bounds values given a 1D coordinate variable.
    Assumes equal spacing on either side of the coordinate label.
    This is an approximation only.
    Output has an added "bounds" dimension at the end.
    """
    if dim not in da.dims:
        (dim,) = da.cf.axes[dim]
    ADDED_INDEX = False
    if dim not in da.coords:
        # For proper alignment in the lines below, we need an index on dim.
        da = da.assign_coords({dim: da[dim]})
        ADDED_INDEX = True

    diff = da.diff(dim)
    lower = da - diff / 2
    upper = da + diff / 2
    bounds = xr.concat([lower, upper], dim="bounds")

    first = (bounds.isel({dim: 0}) - diff.isel({dim: 0})).assign_coords(
        {dim: da[dim][0]}
    )
    result = xr.concat([first, bounds], dim=dim).transpose(..., "bounds")
    if ADDED_INDEX:
        result = result.drop_vars(dim)
    return result


def _guess_bounds_2d(da, dims):
    """
    Guess bounds values given a 2D coordinate variable.
    Assumes equal spacing on either side of the coordinate label.
    This is a coarse approximation, especially for curvilinear grids.
    Output has an added "bounds" dimension at the end.
    """
    daX = _guess_bounds_1d(da, dims[0]).rename(bounds="Xbnds")
    daXY = _guess_bounds_1d(daX, dims[1]).rename(bounds="Ybnds")
    # At this point, we might have different corners for adjacent cells, we average them together to have a nice grid
    # To make this vectorized and keep the edges, we'll pad with NaNs and ignore them in the averages
    daXYp = (
        daXY.pad({d: (1, 1) for d in dims}, mode="constant", constant_values=np.nan)
        .transpose(*dims, "Xbnds", "Ybnds")
        .values
    )  # Tranpose for an easier notation
    # Mean of the corners that should be the same point.
    daXYm = np.stack(
        (
            # Lower left corner (mean of : upper right of the lower left cell, lower right of the upper left cell, and so on, ccw)
            np.nanmean(
                np.stack(
                    (
                        daXYp[:-2, :-2, 1, 1],
                        daXYp[:-2, 1:-1, 1, 0],
                        daXYp[1:-1, 1:-1, 0, 0],
                        daXYp[1:-1, :-2, 0, 1],
                    )
                ),
                axis=0,
            ),
            # Upper left corner
            np.nanmean(
                np.stack(
                    (
                        daXYp[:-2, 1:-1, 1, 1],
                        daXYp[:-2, 2:, 1, 0],
                        daXYp[1:-1, 2:, 0, 0],
                        daXYp[1:-1, 1:-1, 0, 1],
                    )
                ),
                axis=0,
            ),
            # Upper right
            np.nanmean(
                np.stack(
                    (
                        daXYp[1:-1, 1:-1, 1, 1],
                        daXYp[1:-1, 2:, 1, 0],
                        daXYp[2:, 2:, 0, 0],
                        daXYp[2:, 1:-1, 0, 1],
                    )
                ),
                axis=0,
            ),
            # Lower right
            np.nanmean(
                np.stack(
                    (
                        daXYp[1:-1, :-2, 1, 1],
                        daXYp[1:-1, 1:-1, 1, 0],
                        daXYp[2:, 1:-1, 0, 0],
                        daXYp[2:, :-2, 0, 1],
                    )
                ),
                axis=0,
            ),
        ),
        axis=-1,
    )
    return xr.DataArray(daXYm, dims=(*dims, "bounds"), coords=da.coords)


def bounds_to_vertices(
    bounds: DataArray,
    bounds_dim: Hashable,
    core_dims=None,
    order: str | None = "counterclockwise",
) -> DataArray:
    """
    Convert bounds variable to vertices.

    There are 2 covered cases:
     - 1D coordinates, with bounds of shape (N, 2),
       converted to vertices of shape (N+1,)
     - 2D coordinates, with bounds of shape (N, M, 4).
       converted to vertices of shape (N+1, M+1).

    Parameters
    ----------
    bounds : DataArray
        The bounds to convert.
    bounds_dim : str
        The name of the bounds dimension of `bounds` (the one of length 2 or 4).
    core_dims : list, optional
        List of core dimensions for apply_ufunc. This must not include bounds_dims.
        The shape of ``(*core_dims, bounds_dim)`` must be (N, 2) or (N, M, 4).
    order : {'counterclockwise', 'clockwise', None}
        Valid for 2D coordinates only (i.e. bounds of shape (..., N, M, 4), ignored otherwise.
        Order the bounds are given in, assuming that ax0-ax1-upward is a right handed
        coordinate system, where ax0 and ax1 are the two first dimensions of `bounds`.
        If None, the counterclockwise version is computed and then verified. If the
        check fails the clockwise version is returned. See Notes for more details.

    Returns
    -------
    DataArray
        Either of shape (N+1,) or (N+1, M+1). New vertex dimensions are named
        from the initial dimension and suffix "_vertices".

    Notes
    -----
    Getting the correct axes "order" is tricky. There are no real standards for
    dimension names or even axes order, even though the CF conventions mentions the
    ax0-ax1-upward (counterclockwise bounds) as being the default. Moreover, xarray can
    tranpose data without raising any warning or error, which make attributes
    unreliable.

    References
    ----------
    Please refer to the CF conventions document : http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-boundaries.
    """

    if core_dims is None:
        core_dims = [dim for dim in bounds.dims if dim != bounds_dim]

    output_sizes = {f"{dim}_vertices": bounds.sizes[dim] + 1 for dim in core_dims}
    output_core_dims = list(output_sizes.keys())

    n_core_dims = len(core_dims)
    nbounds = bounds[bounds_dim].size

    if not (n_core_dims == 2 and nbounds == 4) and not (
        n_core_dims == 1 and nbounds == 2
    ):
        raise ValueError(
            f"Bounds format not understood. Got {bounds.dims} with shape {bounds.shape}."
        )

    core_dim_coords = {
        dim: bounds.coords[dim].values for dim in core_dims if dim in bounds.coords
    }
    core_dim_orders = _get_core_dim_orders(core_dim_coords)

    return xr.apply_ufunc(
        _bounds_helper,
        bounds,
        input_core_dims=[core_dims + [bounds_dim]],
        dask="parallelized",
        kwargs={
            "n_core_dims": n_core_dims,
            "nbounds": nbounds,
            "order": order,
            "core_dim_orders": core_dim_orders,
        },
        output_core_dims=[output_core_dims],
        dask_gufunc_kwargs=dict(output_sizes=output_sizes),
        output_dtypes=[bounds.dtype],
    )


def _get_core_dim_orders(core_dim_coords: dict[str, np.ndarray]) -> dict[str, str]:
    """
    Determine the order (ascending, descending, or mixed) of each core dimension
    based on its coordinates.

    Repeated (equal) coordinates are ignored when determining the order. If all
    coordinates are equal, the order is treated as "ascending".

    Parameters
    ----------
    core_dim_coords : dict of str to np.ndarray
        A dictionary mapping dimension names to their coordinate arrays.

    Returns
    -------
    core_dim_orders : dict of str to str
        A dictionary mapping each dimension name to a string indicating the order:
        - "ascending": strictly increasing (ignoring repeated values)
        - "descending": strictly decreasing (ignoring repeated values)
        - "mixed": neither strictly increasing nor decreasing (ignoring repeated values)
    """
    core_dim_orders = {}

    for dim, coords in core_dim_coords.items():
        diffs = np.diff(coords)

        # Handle datetime64 and timedelta64 safely for both numpy 1.26.4 and numpy 2
        if np.issubdtype(coords.dtype, np.datetime64) or np.issubdtype(
            coords.dtype, np.timedelta64
        ):
            # Cast to float64 for safe comparison
            diffs_float = diffs.astype("float64")
            nonzero_diffs = diffs_float[diffs_float != 0]
        else:
            zero = 0
            nonzero_diffs = diffs[diffs != zero]

        if nonzero_diffs.size == 0:
            # All values are equal, treat as ascending
            core_dim_orders[dim] = "ascending"
        elif np.all(nonzero_diffs > 0):
            core_dim_orders[dim] = "ascending"
        elif np.all(nonzero_diffs < 0):
            core_dim_orders[dim] = "descending"
        else:
            core_dim_orders[dim] = "mixed"

    return core_dim_orders


def _bounds_helper(values, n_core_dims, nbounds, order, core_dim_orders):
    if n_core_dims == 2 and nbounds == 4:
        # Vertices case (2D lat/lon)
        if order in ["counterclockwise", None]:
            # Names assume we are drawing axis 1 upward et axis 2 rightward.
            bot_left = values[..., :, :, 0]
            bot_right = values[..., :, -1:, 1]
            top_right = values[..., -1:, -1:, 2]
            top_left = values[..., -1:, :, 3]
            vertex_vals = np.block([[bot_left, bot_right], [top_left, top_right]])
        if order is None:  # We verify if the ccw version works.
            calc_bnds = vertices_to_bounds(vertex_vals).values
            order = (
                "counterclockwise" if np.allclose(calc_bnds, values) else "clockwise"
            )
        if order == "clockwise":
            bot_left = values[..., :, :, 0]
            top_left = values[..., -1:, :, 1]
            top_right = values[..., -1:, -1:, 2]
            bot_right = values[..., :, -1:, 3]
            # Our assumption was wrong, axis 1 is rightward and axis 2 is upward
            vertex_vals = np.block([[bot_left, bot_right], [top_left, top_right]])
    elif n_core_dims == 1 and nbounds == 2:
        # Middle points case (1D lat/lon)
        vertex_vals = _get_ordered_vertices(values, core_dim_orders)

    return vertex_vals


def _get_ordered_vertices(
    bounds: np.ndarray, core_dim_orders: dict[str, str]
) -> np.ndarray:
    """
    Convert a bounds array of shape (..., N, 2) or (N, 2) into a 1D array of vertices.

    This function reconstructs the vertices from a bounds array, handling both
    monotonic and non-monotonic cases.

    Monotonic bounds (all values strictly increase or decrease when flattened):
        - Concatenate the left endpoints (bounds[..., :, 0]) with the last right
          endpoint (bounds[..., -1, 1]) to form the vertices.

    Non-monotonic bounds:
        - Determine the order of the core dimension(s) ('ascending' or 'descending').
        - For ascending order:
            - Use the minimum of each interval as the vertex.
            - Use the maximum of the last interval as the final vertex.
        - For descending order:
            - Use the maximum of each interval as the vertex.
            - Use the minimum of the last interval as the final vertex.
        - Vertices are then sorted to match the coordinate direction.

    Features:
        - Handles both ascending and descending bounds.
        - Preserves repeated coordinates if present.
        - Output shape is (..., N+1) or (N+1,).

    Parameters
    ----------
    bounds : np.ndarray
        Array of bounds, typically with shape (N, 2) or (..., N, 2).
    core_dim_orders : dict[str, str]
        Dictionary mapping core dimension names to their order ('ascending' or
        'descending'). Used for sorting the vertices.

    Returns
    -------
    np.ndarray
        Array of vertices with shape (..., N+1) or (N+1,).
    """
    order = _get_order_of_core_dims(core_dim_orders)

    if _is_bounds_monotonic(bounds):
        vertices = np.concatenate((bounds[..., :, 0], bounds[..., -1:, 1]), axis=-1)
    else:
        if order == "ascending":
            endpoints = np.minimum(bounds[..., :, 0], bounds[..., :, 1])
            last_endpoint = np.maximum(bounds[..., -1, 0], bounds[..., -1, 1])
        elif order == "descending":
            endpoints = np.maximum(bounds[..., :, 0], bounds[..., :, 1])
            last_endpoint = np.minimum(bounds[..., -1, 0], bounds[..., -1, 1])

        vertices = np.concatenate(
            [endpoints, np.expand_dims(last_endpoint, axis=-1)], axis=-1
        )

    vertices = _sort_vertices(vertices, order)

    return vertices


def _is_bounds_monotonic(bounds: np.ndarray) -> bool:
    """Check if the bounds are monotonic.

    Arrays are monotonic if all values are increasing or decreasing. This
    functions ignores  an intervals where consecutive values are equal, which
    represent repeated coordinates.

    Parameters
    ----------
    arr : np.ndarray
        Numpy array to check, typically with shape (..., N, 2).

    Returns
    -------
    bool
        True if the flattened array is increasing or decreasing, False otherwise.
    """
    # NOTE: Python 3.10 uses numpy 1.26.4. If the input is a datetime64 array,
    # numpy 1.26.4 may raise: numpy.core._exceptions._UFuncInputCastingError:
    # Cannot cast ufunc 'greater' input 0 from dtype('<m8[ns]') to dtype('<m8')
    # with casting rule 'same_kind' To avoid this, always cast to float64 before
    # np.diff.
    arr_numeric = bounds.astype("float64").flatten()
    diffs = np.diff(arr_numeric)
    nonzero_diffs = diffs[diffs != 0]

    # All values are equal, treat as monotonic
    if nonzero_diffs.size == 0:
        return True

    return bool(np.all(nonzero_diffs > 0) or np.all(nonzero_diffs < 0))


def _get_order_of_core_dims(core_dim_orders: dict[str, str]) -> str:
    """
    Determines the common order of core dimensions from a dictionary of
    dimension orders.

    Parameters
    ----------
    core_dim_orders : dict of str
        A dictionary mapping dimension names to their respective order strings.

    Returns
    -------
    order : str
        The common order string shared by all core dimensions.

    Raises
    ------
    ValueError
        If the core dimension orders are not all aligned (i.e., not all values
        are the same).
    """
    orders = set(core_dim_orders.values())

    if len(orders) != 1:
        raise ValueError(
            f"All core dimension orders must be aligned. Got orders: {core_dim_orders}"
        )

    order = next(iter(orders))

    return order


def _sort_vertices(vertices: np.ndarray, order: str) -> np.ndarray:
    """
    Sorts the vertices array along the last axis in ascending or descending order.

    Parameters
    ----------
    vertices : np.ndarray
        An array of vertices to be sorted. Sorting is performed along the last
        axis.
    order : str
        The order in which to sort the vertices. Must be either "ascending" or
        any other value for descending order.

    Returns
    -------
    np.ndarray
        The sorted array of vertices, with the same shape as the input.

    Examples
    --------
    >>> import numpy as np
    >>> vertices = np.array([[3, 1, 2], [6, 5, 4]])
    >>> _sort_vertices(vertices, "ascending")
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> _sort_vertices(vertices, "descending")
    array([[3, 2, 1],
           [6, 5, 4]])
    """
    if order == "ascending":
        new_vertices = np.sort(vertices, axis=-1)
    else:
        new_vertices = np.sort(vertices, axis=-1)[..., ::-1]

    return new_vertices


def vertices_to_bounds(
    vertices: DataArray, out_dims: Sequence[str] = ("bounds", "x", "y")
) -> DataArray:
    """
    Convert vertices to CF-compliant bounds.

    There are 2 covered cases:
     - 1D coordinates, with vertices of shape (N+1,),
       converted to bounds of shape (N, 2)
     - 2D coordinates, with vertices of shape (N+1, M+1).
       converted to bounds of shape (N, M, 4).

    Parameters
    ----------
    vertices : DataArray
        The vertices to convert. Must be of shape (N + 1) or (N + 1, M + 1).
    out_dims : Sequence[str],
        The name of the dimension in the output. The first is the 'bounds'
        dimension and the following are the coordinate dimensions.

    Returns
    -------
    DataArray
        Either of shape (2, N) or (4, N, M).

    References
    ----------
    Please refer to the CF conventions document : http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-boundaries.
    """
    if vertices.ndim == 1:
        bnd_vals = np.stack((vertices[:-1], vertices[1:]), axis=0)
    elif vertices.ndim == 2:
        bnd_vals = np.stack(
            (
                vertices[:-1, :-1],
                vertices[:-1, 1:],
                vertices[1:, 1:],
                vertices[1:, :-1],
            ),
            axis=0,
        )
    else:
        raise ValueError(
            f"vertices format not understood. Got {vertices.dims} with shape {vertices.shape}."
        )
    return xr.DataArray(bnd_vals, dims=out_dims[: vertices.ndim + 1]).transpose(
        ..., out_dims[0]
    )
