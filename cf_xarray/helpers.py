from typing import Optional, Sequence

import numpy as np
import xarray as xr
from xarray import DataArray


def bounds_to_corners(
    bounds: DataArray, bounds_dim: str, order: Optional[str] = "counterclockwise"
) -> DataArray:
    """
    Convert bounds variable to corners. There 2 covered cases:
     - 1D coordinates, with bounds of shape (N, 2),
       converted to corners of shape (N+1,)
     - 2D coordinates, with bounds of shape (N, M, 4).
       converted to corners of shape (N+1, M+1).

    Parameters
    ----------
    bounds: DataArray
        The bounds to convert. Must be of shape (N, 2) or (N, M, 4).
    bounds_dim : str
        The name of the bounds dimension of `bounds` (the one of length 2 or 4).
    order : {'counterclockwise', 'ccw', 'clockwise', 'cw', None}
        Valid for 2D coordinates only (bounds of shape (N, M, 4), ignored otherwise.
        Order the bounds are given in, assuming that axis0-axis1-upward
        is a right handed coordinate system.
        If None, the counterclockwise version is computed and then
        verified. If the check fails the clockwise version is returned.
    Returns
    -------
    DataArray
        Either of shape (N+1,) or (N+1, M+1). New corner dimensions are named
        from the intial dimension and suffix "_corners".
    """
    # Get old and new dimension names and retranspose array to have bounds dim at axis 0.
    bnd_dim = (
        bounds_dim if isinstance(bounds_dim, str) else bounds.get_axis_num(bounds_dim)
    )
    old_dims = [dim for dim in bounds.dims if dim != bnd_dim]
    new_dims = [f"{dim}_corners" for dim in old_dims]
    values = bounds.transpose(bnd_dim, *old_dims).values
    if len(old_dims) == 2 and bounds.ndim == 3 and bounds[bnd_dim].size == 4:
        # Vertices case (2D lat/lon)
        if order in ["counterclockwise", "ccw", None]:
            # Names assume we are drawing axis 1 upward et axis 2 rightward.
            bot_left = values[0, :, :]
            bot_right = values[1, :, -1:]
            top_right = values[2, -1:, -1:]
            top_left = values[3, -1:, :]
            corner_vals = np.block([[bot_left, bot_right], [top_left, top_right]])
        if order is None:  # We verify if the ccw version works.
            calc_bnds = corners_to_bounds(corner_vals).values
            order = "ccw" if np.all(calc_bnds == values) else "cw"
        if order in ["cw", "clockwise"]:
            bot_left = values[0, :, :]
            top_left = values[1, -1:, :]
            top_right = values[2, -1:, -1:]
            bot_right = values[3, :, -1:]
            # Our asumption was wrong, axis 1 is rightward and axis 2 is upward
            corner_vals = np.block([[bot_left, bot_right], [top_left, top_right]])
    elif len(old_dims) == 1 and bounds.ndim == 2 and bounds[bnd_dim].size == 2:
        # Middle points case (1D lat/lon)
        corner_vals = np.concatenate((values[0, :], values[1, -1:]))
    else:
        raise ValueError(
            f"Bounds format not understood. Got {bounds.dims} with shape {bounds.shape}."
        )

    return xr.DataArray(corner_vals, dims=new_dims)


def corners_to_bounds(
    corners: DataArray, out_dims: Sequence[str] = ("bounds", "x", "y")
) -> DataArray:
    """
    Convert corners to CF-compliant bounds. There 2 covered cases:
     - 1D coordinates, with corners of shape (N+1,),
       converted to bounds of shape (N, 2)
     - 2D coordinates, with corners of shape (N+1, M+1).
       converted to bounds of shape (N, M, 4).

    Parameters
    ----------
    bounds: DataArray
        The bounds to convert. Must be of shape (N, 2) or (N, M, 4).
    out_dims : Sequence[str],
        The name of the dimension in the output. The first is the 'bounds'
        dimension and the following are the coordinate dimensions.
    Returns
    -------
    DataArray
    """
    if corners.ndim == 1:
        bnd_vals = np.stack((corners[:-1], corners[1:]), axis=0)
    elif corners.ndim == 2:
        bnd_vals = np.stack(
            (corners[:-1, :-1], corners[:-1, 1:], corners[1:, 1:], corners[1:, :-1]),
            axis=0,
        )
    else:
        raise ValueError(
            f"Corners format not understood. Got {corners.dims} with shape {corners.shape}."
        )
    return xr.DataArray(bnd_vals, dims=out_dims[: corners.ndim + 1])
