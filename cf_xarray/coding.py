"""
Encoders and decoders for CF conventions not implemented by Xarray.
"""
import numpy as np
import pandas as pd
import xarray as xr


def encode_compress(ds, idxnames=None):
    """
    Encode a MultiIndexed dimension using the "compression by gathering" CF convention.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with at least one MultiIndexed dimension
    idxnames : hashable or iterable of hashable, optional
        Dimensions that are MultiIndex-ed. If None, will detect all MultiIndex-ed dimensions.

    Returns
    -------
    xarray.Dataset
        Encoded Dataset with ``name`` as a integer coordinate with a ``"compress"`` attribute.

    See Also
    --------
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering
    """
    if idxnames is None:
        idxnames = tuple(
            name
            for name, idx in ds.indexes.items()
            if isinstance(idx, pd.MultiIndex) and idx.name == name
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    if not idxnames:
        raise ValueError("No MultiIndex-ed dimensions found in Dataset.")

    encoded = ds.reset_index(idxnames)
    for idxname in idxnames:
        mindex = ds.indexes[idxname]
        coords = dict(zip(mindex.names, mindex.levels))
        for coord in coords:
            encoded[coord] = coords[coord].values
        shape = [encoded.sizes[coord] for coord in coords]
        encoded[idxname] = np.ravel_multi_index(mindex.codes, shape)
        encoded[idxname].attrs["compress"] = " ".join(mindex.names)
    return encoded


def decode_compress(encoded, idxnames=None):
    """
    Decode a compressed variable to a pandas MultiIndex.

    Parameters
    ----------
    encoded : xarray.Dataset
        Encoded Dataset with variables that use "compression by gathering".capitalize
    idxnames : hashable or iterable of hashable, optional
        Variable names that represents a compressed dimension. These variables must have
        the attribute ``"compress"``. If None, will detect all indexes with a ``"compress"``
        attribute and decode those.

    Returns
    -------
    xarray.Dataset
        Decoded Dataset with ``name`` as a MultiIndexed dimension.

    See Also
    --------
    http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering
    """
    decoded = xr.Dataset()
    if idxnames is None:
        idxnames = tuple(
            name for name in encoded.indexes if "compress" in encoded[name].attrs
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    for idxname in idxnames:
        if "compress" not in encoded[idxname].attrs:
            raise ValueError("Attribute 'compress' not found in provided Dataset.")

        if not isinstance(encoded, xr.Dataset):
            raise ValueError(
                f"Must provide a Dataset. Received {type(encoded)} instead."
            )

        names = encoded[idxname].attrs["compress"].split(" ")
        shape = [encoded.sizes[dim] for dim in names]
        indices = np.unravel_index(encoded.landpoint.values, shape)
        arrays = [encoded[dim].values[index] for dim, index in zip(names, indices)]
        mindex = pd.MultiIndex.from_arrays(arrays, names=names)

        decoded.coords[idxname] = mindex
        for varname in encoded.data_vars:
            if idxname in encoded[varname].dims:
                decoded[varname] = (idxname, encoded[varname].values)
    return decoded
