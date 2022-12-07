"""
Encoders and decoders for CF conventions not implemented by Xarray.
"""
import numpy as np
import pandas as pd
import xarray as xr


def encode_multi_index_as_compress(ds, idxnames=None):
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

    References
    ----------
    CF conventions on `compression by gathering <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering>`_
    """
    if idxnames is None:
        idxnames = tuple(
            name
            for name, idx in ds.indexes.items()
            if isinstance(idx, pd.MultiIndex)
            # After the flexible indexes refactor, all MultiIndex Levels
            # have a MultiIndex but the name won't match.
            # Prior to that refactor, there is only a single MultiIndex with name=None
            and (idx.name == name if idx.name is not None else True)
        )
    elif isinstance(idxnames, str):
        idxnames = (idxnames,)

    if not idxnames:
        raise ValueError("No MultiIndex-ed dimensions found in Dataset.")

    encoded = ds.reset_index(idxnames)
    for idxname in idxnames:
        mindex = ds.indexes[idxname]
        coords = dict(zip(mindex.names, mindex.levels))
        encoded.update(coords)
        for c in coords:
            encoded[c].attrs = ds[c].attrs
            encoded[c].encoding = ds[c].encoding
        encoded[idxname] = np.ravel_multi_index(mindex.codes, mindex.levshape)
        encoded[idxname].attrs = ds[idxname].attrs.copy()
        if (
            "compress" in encoded[idxname].encoding
            or "compress" in encoded[idxname].attrs
        ):
            raise ValueError(
                f"Does not support the 'compress' attribute in {idxname}.encoding or {idxname}.attrs. "
                "This is generated automatically."
            )
        encoded[idxname].attrs["compress"] = " ".join(mindex.names)
    return encoded


def decode_compress_to_multi_index(encoded, idxnames=None):
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

    References
    ----------
    CF conventions on `compression by gathering <http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html#compression-by-gathering>`_
    """
    decoded = xr.Dataset(data_vars=encoded.data_vars, attrs=encoded.attrs.copy())
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
        indices = np.unravel_index(encoded[idxname].data, shape)
        try:
            from xarray.indexes import PandasMultiIndex

            variables = {
                dim: encoded[dim].isel({dim: xr.Variable(data=index, dims=idxname)})
                for dim, index in zip(names, indices)
            }
            decoded = decoded.assign_coords(variables).set_xindex(
                names, PandasMultiIndex
            )
        except ImportError:
            arrays = [encoded[dim].data[index] for dim, index in zip(names, indices)]
            mindex = pd.MultiIndex.from_arrays(arrays, names=names)
            decoded.coords[idxname] = mindex

        decoded[idxname].attrs = encoded[idxname].attrs.copy()
        for coord in names:
            variable = encoded._variables[coord]
            decoded[coord].attrs = variable.attrs.copy()
            decoded[coord].encoding = variable.encoding.copy()
        del decoded[idxname].attrs["compress"]

    return decoded
