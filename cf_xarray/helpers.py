from __future__ import annotations

from typing import Hashable, Sequence

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
        daXY.pad({d: (1, 1) for d in dims}, mode="constant", constant_values=np.NaN)
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
    Convert bounds variable to vertices. There are 2 covered cases:
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
    order : {'counterclockwise', 'clockwise', None}
        Valid for 2D coordinates only (i.e. bounds of shape (..., N, M, 4), ignored otherwise.
        Order the bounds are given in, assuming that ax0-ax1-upward is a right handed
        coordinate system, where ax0 and ax1 are the two first dimensions of `bounds`.
        If None, the counterclockwise version is computed and then verified. If the
        check fails the clockwise version is returned. See Notes for more details.
    core_dims : list, optional
        List of core dimensions for apply_ufunc. This must not include bounds_dims.
        The shape of (*core_dims, bounds_dim) must be (N, 2) or (N, M, 4).

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

    return xr.apply_ufunc(
        _bounds_helper,
        bounds,
        input_core_dims=[core_dims + [bounds_dim]],
        dask="parallelized",
        kwargs={"n_core_dims": n_core_dims, "nbounds": nbounds, "order": order},
        output_core_dims=[output_core_dims],
        dask_gufunc_kwargs=dict(output_sizes=output_sizes),
        output_dtypes=[bounds.dtype],
    )


def _bounds_helper(values, n_core_dims, nbounds, order):
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
        vertex_vals = np.concatenate((values[..., :, 0], values[..., -1:, 1]), axis=-1)

    return vertex_vals


def vertices_to_bounds(
    vertices: DataArray, out_dims: Sequence[str] = ("bounds", "x", "y")
) -> DataArray:
    """
    Convert vertices to CF-compliant bounds. There are 2 covered cases:
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
