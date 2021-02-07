from typing import Optional, Sequence

import numpy as np
import xarray as xr
from xarray import DataArray


def bounds_to_vertices(
    bounds: DataArray, bounds_dim: str, order: Optional[str] = "counterclockwise"
) -> DataArray:
    """
    Convert bounds variable to vertices. There 2 covered cases:
     - 1D coordinates, with bounds of shape (N, 2),
       converted to vertices of shape (N+1,)
     - 2D coordinates, with bounds of shape (N, M, 4).
       converted to vertices of shape (N+1, M+1).

    Parameters
    ----------
    bounds : DataArray
        The bounds to convert. Must be of shape (N, 2) or (N, M, 4).
    bounds_dim : str
        The name of the bounds dimension of `bounds` (the one of length 2 or 4).
    order : {'counterclockwise', 'clockwise', None}
        Valid for 2D coordinates only (bounds of shape (N, M, 4), ignored otherwise.
        Order the bounds are given in, assuming that ax0-ax1-upward is a right handed
        coordinate system, where ax0 and ax1 are the two first dimensions of `bounds`.
        If None, the counterclockwise version is computed and then verified. If the
        check fails the clockwise version is returned. See Notes for more details.

    Returns
    -------
    DataArray
        Either of shape (N+1,) or (N+1, M+1). New vertex dimensions are named
        from the intial dimension and suffix "_vertices".

    Notes
    -----
    Getting the correct axes "order" is tricky. There are no real standards for
    dimension names or even axes order, even though the CF conventions mentions the
    ax0-ax1-upward (counterclockwise bounds) as being the default. Moreover, xarray can
    tranpose data without raising any warning or error, which make attributes
    unreliable.

    Please refer to the CF conventions document : http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#cell-boundaries.
    """
    # Get old and new dimension names and retranspose array to have bounds dim at axis 0.
    bnd_dim = (
        bounds_dim if isinstance(bounds_dim, str) else bounds.get_axis_num(bounds_dim)
    )
    old_dims = [dim for dim in bounds.dims if dim != bnd_dim]
    new_dims = [f"{dim}_vertices" for dim in old_dims]
    values = bounds.transpose(bnd_dim, *old_dims).data
    if len(old_dims) == 2 and bounds.ndim == 3 and bounds[bnd_dim].size == 4:
        # Vertices case (2D lat/lon)
        if order in ["counterclockwise", None]:
            # Names assume we are drawing axis 1 upward et axis 2 rightward.
            bot_left = values[0, :, :]
            bot_right = values[1, :, -1:]
            top_right = values[2, -1:, -1:]
            top_left = values[3, -1:, :]
            vertex_vals = np.block([[bot_left, bot_right], [top_left, top_right]])
        if order is None:  # We verify if the ccw version works.
            calc_bnds = vertices_to_bounds(vertex_vals).values
            order = "counterclockwise" if np.all(calc_bnds == values) else "clockwise"
        if order == "clockwise":
            bot_left = values[0, :, :]
            top_left = values[1, -1:, :]
            top_right = values[2, -1:, -1:]
            bot_right = values[3, :, -1:]
            # Our asumption was wrong, axis 1 is rightward and axis 2 is upward
            vertex_vals = np.block([[bot_left, bot_right], [top_left, top_right]])
    elif len(old_dims) == 1 and bounds.ndim == 2 and bounds[bnd_dim].size == 2:
        # Middle points case (1D lat/lon)
        vertex_vals = np.concatenate((values[0, :], values[1, -1:]))
    else:
        raise ValueError(
            f"Bounds format not understood. Got {bounds.dims} with shape {bounds.shape}."
        )

    return xr.DataArray(vertex_vals, dims=new_dims)


def vertices_to_bounds(
    vertices: DataArray, out_dims: Sequence[str] = ("bounds", "x", "y")
) -> DataArray:
    """
    Convert vertices to CF-compliant bounds. There 2 covered cases:
     - 1D coordinates, with vertices of shape (N+1,),
       converted to bounds of shape (N, 2)
     - 2D coordinates, with vertices of shape (N+1, M+1).
       converted to bounds of shape (N, M, 4).

    Parameters
    ----------
    bounds : DataArray
        The bounds to convert. Must be of shape (N, 2) or (N, M, 4).
    out_dims : Sequence[str],
        The name of the dimension in the output. The first is the 'bounds'
        dimension and the following are the coordinate dimensions.
    Returns
    -------
    DataArray
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
    return xr.DataArray(bnd_vals, dims=out_dims[: vertices.ndim + 1])
