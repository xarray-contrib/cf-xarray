import inspect
import sys

import numpy as np
import xarray as xr

ocean_stdname_map = {
    "altitude": {
        "zlev": "altitude",
        "eta": "sea_surface_height_above_geoid",
        "depth": "sea_floor_depth_below_geoid",
    },
    "height_above_geopotential_datum": {
        "zlev": "height_above_geopotential_datum",
        "eta": "sea_surface_height_above_ geopotential_datum",
        "depth": "sea_floor_depth_below_ geopotential_datum",
    },
    "height_above_reference_ellipsoid": {
        "zlev": "height_above_reference_ellipsoid",
        "eta": "sea_surface_height_above_ reference_ellipsoid",
        "depth": "sea_floor_depth_below_ reference_ellipsoid",
    },
    "height_above_mean_sea_level": {
        "zlev": "height_above_mean_sea_level",
        "eta": "sea_surface_height_above_mean_ sea_level",
        "depth": "sea_floor_depth_below_mean_ sea_level",
    },
}


def _derive_ocean_stdname(**kwargs):
    """Derive standard name for computer ocean coordinates.

    Uses the concatentation of formula terms `zlev`, `eta`, and `depth`
    standard names to compare against formula term and standard names
    from a table. This can occur with any combination e.g. `zlev`, or
    `zlev` + `depth`. If a match is found the standard name for the
    computed value is returned.

    Parameters
    ----------
    zlev : dict
        Attributes for `zlev` variable.
    eta : dict
        Attributes for `eta` variable.
    depth : dict
        Attributes for `depth` variable.

    Returns
    -------
    str
        Standard name for the computer value.

    Raises
    ------
    ValueError
        If `kwargs` is empty, missing values for `kwargs` keys, or could not derive the standard name.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#table-computed-standard-names
    """
    found_stdname = None

    allowed_names = {"zlev", "eta", "depth"}

    if len(kwargs) == 0 or not (set(kwargs) <= allowed_names):
        raise ValueError(
            f"Must provide atleast one of {', '.join(sorted(allowed_names))}."
        )

    search_term = ""

    for x, y in sorted(kwargs.items(), key=lambda x: x[0]):
        try:
            search_term = f"{search_term}{y['standard_name']}"
        except TypeError:
            raise ValueError(
                f"The values for {', '.join(sorted(kwargs.keys()))} cannot be `None`."
            ) from None
        except KeyError:
            raise ValueError(
                f"The standard name for the {x!r} variable is not available."
            ) from None

    for x, y in ocean_stdname_map.items():
        check_term = "".join(
            [
                y[i]
                for i, j in sorted(kwargs.items(), key=lambda x: x[0])
                if j is not None
            ]
        )

        if search_term == check_term:
            found_stdname = x

            break

    if found_stdname is None:
        stdnames = ", ".join(
            [y["standard_name"] for _, y in sorted(kwargs.items(), key=lambda x: x[0])]
        )

        raise ValueError(
            f"Could not derive standard name from combination of {stdnames}."
        )

    return found_stdname


def check_requirements(func, terms):
    """Checks terms against function requirements.

    Uses `func` argument specification as requirements and checks terms against this.
    Postitional arguments without a default are required but when a default value is
    provided the arguement is considered optional. Atleast one optional argument must
    be present (special case for atmosphere_hybrid_sigma_pressure_coordinate).

    Parameters
    ----------
    func : function
        Function to check requirements.
    terms : list
        List of terms to check `func` requirements against.

    Raises
    ------
    KeyError
        If `terms` is empty or missing required/optional terms.
    """
    if not isinstance(terms, set):
        terms = set(terms)

    spec = inspect.getfullargspec(func)

    args = spec.args or []

    if len(terms) == 0:
        raise KeyError(f"Required terms {', '.join(sorted(args))} absent in dataset.")

    n = len(spec.defaults or [])

    # last `n` arguments are optional
    opt = set(args[len(args) - n :])

    req = set(args) - opt

    # req must all be present in terms
    if (req & terms) != req:
        req_diff = sorted(req - terms)

        opt_err = ""

        # optional is not present in terms
        if len(opt) > 0 and not (opt <= terms):
            opt_err = f"and atleast one optional term {', '.join(sorted(opt))} "

        raise KeyError(
            f"Required terms {', '.join(req_diff)} {opt_err}absent in dataset."
        )

    # atleast one optional is in diff, only required for atmoshphere hybrid sigma pressure coordinate
    if len(opt) > 0 and not (opt <= terms):
        raise KeyError(
            f"Atleast one of the optional terms {', '.join(sorted(opt))} is absent in dataset."
        )


def func_from_stdname(stdname):
    """Get function from module.

    Uses `stdname` to return function from module.

    Parameters
    ----------
    stdname : str
        Name of the function.

    Raises
    ------
    AttributeError
        If a function name `stdname` is not in the module.
    """
    m = sys.modules[__name__]

    return getattr(m, stdname)


def derive_dimension_order(output_order, **dim_map):
    """Derive dimension ordering from input map.

    This will derive a dimensinal ordering from a map of dimension
    identifiers and variables containing the dimensions.

    This is useful when dimension names are not know.

    For example if the desired output ordering was "nkji" where
    variable "A" contains "nji" (time, lat, lon) and "B" contains
    "k" (height) then the output would be (time, height, lat, lon).

    This also works when dimensions are missing.

    For example if the desired output ordering was "nkji" where
    variable "A" contains "n" (time) and "B" contains
    "k" (height) then the output would be (time, height).

    Parameters
    ----------
    output_order : str
        Dimension identifiers in desired order, e.g. "nkji".
    **dim_map : dict
        Dimension identifiers and variable containing them, e.g. "nji": eta, "k": s.

    Returns
    -------
    list
        Output dimensions in desired order.
    """
    dims = {}

    for x, y in dim_map.items():
        for i, z in enumerate(x):
            try:
                dims[z] = y.dims[i]
            except IndexError:
                dims[z] = None

    return tuple(dims[x] for x in list(output_order) if dims[x] is not None)


def atmosphere_ln_pressure_coordinate(p0, lev):
    """Atmosphere natural log pressure coordinate.

    Standard name: atmosphere_ln_pressure_coordinate

    Parameters
    ----------
    p0 : xr.DataArray
        Reference pressure.
    lev : xr.DataArray
        Vertical dimensionless coordinate.

    Returns
    -------
    xr.DataArray
        A DataArray with new pressure coordinate.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#atmosphere-natural-log-pressure-coordinate
    """
    p = p0 * np.exp(-lev)

    p = p.squeeze().rename("p").assign_attrs(standard_name="air_pressure")

    return p


def atmosphere_sigma_coordinate(sigma, ps, ptop):
    """Atmosphere sigma coordinate.

    Standard name: atmosphere_sigma_coordinate

    Parameters
    ----------
    sigma : xr.DataArray
        Vertical dimensionless coordinate.
    ps : xr.DataArray
        Horizontal surface pressure.

    Returns
    -------
    xr.DataArray
        A DataArray with new pressure coordinate.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_atmosphere_sigma_coordinate
    """
    p = ptop + sigma * (ps - ptop)

    p = p.squeeze().rename("p").assign_attrs(standard_name="air_pressure")

    output_order = derive_dimension_order("nkji", nji=ps, k=sigma)

    return p.transpose(*output_order)


def atmosphere_hybrid_sigma_pressure_coordinate(b, ps, p0, a=None, ap=None):
    """Atmosphere hybrid sigma pressure coordinate.

    Standard name: atmosphere_hybrid_sigma_pressure_coordinate

    Parameters
    ----------
    b : xr.DataArray
        Component of hybrid coordinate.
    ps : xr.DataArray
        Horizontal surface pressure.
    p0 : xr.DataArray
        Reference pressure.
    a : xr.DataArray
        Component of hybrid coordinate.
    ap : xr.DataArray
        Component of hybrid coordinate.

    Returns
    -------
    xr.DataArray
        A DataArray with new pressure coordinate.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_atmosphere_hybrid_sigma_pressure_coordinate
    """
    if a is None:
        p = ap + b * ps
    else:
        p = a * p0 + b * ps

    p = p.squeeze().rename("p").assign_attrs(standard_name="air_pressure")

    output_order = derive_dimension_order("nkji", nji=ps, k=b)

    return p.transpose(*output_order)


def atmosphere_hybrid_height_coordinate(a, b, orog):
    """Atmosphere hybrid height coordinate.

    Standard name: atmosphere_hybrid_height_coordinate

    Parameters
    ----------
    a : xr.DataArray
        Height.
    b : xr.DataArray
        Dimensionless.
    orog : xr.DataArray
        Height of the surface above the datum.

    Returns
    -------
    xr.DataArray
        A DataArray with the height above the datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#atmosphere-hybrid-height-coordinate
    """
    z = a + b * orog

    orog_stdname = orog.attrs["standard_name"]

    if orog_stdname == "surface_altitude":
        out_stdname = "altitude"
    elif orog_stdname == "surface_height_above_geopotential_datum":
        out_stdname = "height_above_geopotential_datum"

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=orog, k=b)

    return z.transpose(*output_order)


def atmosphere_sleve_coordinate(a, b1, b2, ztop, zsurf1, zsurf2):
    """Atmosphere smooth level vertical (SLEVE) coordinate.

    Standard name: atmosphere_sleve_coordinate

    Parameters
    ----------
    a : xr.DataArray
        Dimensionless coordinate whcih defines hybrid level.
    b1 : xr.DataArray
        Dimensionless coordinate whcih defines hybrid level.
    b2 : xr.DataArray
        Dimensionless coordinate whcih defines hybrid level.
    ztop : xr.DataArray
        Height above the top of the model above datum.
    zsurf1 : xr.DataArray
        Large-scale component of the topography.
    zsurf2 : xr.DataArray
        Small-scale component of the topography.

    Returns
    -------
    xr.DataArray
        A DataArray with the height above the datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_atmosphere_smooth_level_vertical_sleve_coordinate
    """
    z = a * ztop + b1 * zsurf1 + b2 * zsurf2

    ztop_stdname = ztop.attrs["standard_name"]

    if ztop_stdname == "altitude_at_top_of_atmosphere_model":
        out_stdname = "altitude"
    elif ztop_stdname == "height_above_geopotential_datum_at_top_of_atmosphere_model":
        out_stdname = "height_above_geopotential_datum"

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=zsurf1, k=a)

    return z.transpose(*output_order)


def ocean_sigma_coordinate(sigma, eta, depth):
    """Ocean sigma coordinate.

    Standard name: ocean_sigma_coordinate

    Parameters
    ----------
    sigma : xr.DataArray
        Vertical dimensionless coordinate.
    eta : xr.DataArray
        Height of the sea surface (positive upwards) relative to the datum.
    depth : xr.DataArray
        Distance (positive value) from the datum to the sea floor.

    Returns
    -------
    xr.DataArray
        A DataArray with the height (positive upwards) relative to the datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_sigma_coordinate
    """
    z = eta + sigma * (depth + eta)

    out_stdname = _derive_ocean_stdname(eta=eta.attrs, depth=depth.attrs)

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=eta, k=sigma)

    return z.transpose(*output_order)


def ocean_s_coordinate(s, eta, depth, a, b, depth_c):
    """Ocean s-coordinate.

    Standard name: ocean_s_coordinate

    Parameters
    ----------
    s : xr.DataArray
        Dimensionless coordinate.
    eta : xr.DataArray
        Height of the sea surface (positive upwards) relative to the datum.
    depth : xr.DataArray
        Distance (positive value) from the datum to the sea floor.
    a : xr.DataArray
        Constant controlling stretch.
    b : xr.DataArray
        Constant controlling stretch.
    depth_c : xr.DataArray
        Constant controlling stretch.

    Returns
    -------
    xr.DataArray
        A DataArray with the height (positive upwards) relative to the datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate
    """
    C = (1 - b) * np.sinh(a * s) / np.sinh(a) + b * (
        np.tanh(a * (s + 0.5)) / 2 * np.tanh(0.5 * a) - 0.5
    )

    z = eta * (1 + s) + depth_c * s + (depth - depth_c) * C

    out_stdname = _derive_ocean_stdname(eta=eta.attrs, depth=depth.attrs)

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=eta, k=s)

    return z.transpose(*output_order)


def ocean_s_coordinate_g1(s, C, eta, depth, depth_c):
    """Ocean s-coordinate, generic form 1.

    Standard name: ocean_s_coordinate_g1

    Parameters
    ----------
    s : xr.DataArray
        Dimensionless coordinate.
    C : xr.DataArray
        Dimensionless vertical coordinate stretching function.
    eta : xr.DataArray
        Height of the ocean surface (positive upwards) relative to the ocean datum.
    depth : xr.DataArray
        Distance from ocean datum to sea floor (positive value).
    depth_c : xr.DataArray
        Constant (positive value) is a critical depth controlling the stretching.

    Returns
    -------
    xr.DataArray
        A DataArray with the height (positive upwards) relative to ocean datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate_generic_form_1
    """
    S = depth_c * s + (depth - depth_c) * C

    z = S + eta * (1 + s / depth)

    out_stdname = _derive_ocean_stdname(eta=eta.attrs, depth=depth.attrs)

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=eta, k=s)

    return z.transpose(*output_order)


def ocean_s_coordinate_g2(s, C, eta, depth, depth_c):
    """Ocean s-coordinate, generic form 2.

    Standard name: ocean_s_coordinate_g2

    Parameters
    ----------
    s : xr.DataArray
        Dimensionless coordinate.
    C : xr.DataArray
        Dimensionless vertical coordinate stretching function.
    eta : xr.DataArray
        Height of the ocean surface (positive upwards) relative to the ocean datum.
    depth : xr.DataArray
        Distance from ocean datum to sea floor (positive value).
    depth_c : xr.DataArray
        Constant (positive value) is a critical depth controlling the stretching.

    Returns
    -------
    xr.DataArray
        A DataArray with the height (positive upwards) relative to ocean datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate_generic_form_2
    """
    S = (depth_c * s + depth * C) / (depth_c + depth)

    z = eta + (eta + depth) * S

    out_stdname = _derive_ocean_stdname(eta=eta.attrs, depth=depth.attrs)

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=eta, k=s)

    return z.transpose(*output_order)


def ocean_sigma_z_coordinate(sigma, eta, depth, depth_c, nsigma, zlev):
    """Ocean sigma over z coordinate.

    Standard name: ocean_sigma_z_coordinate

    Parameters
    ----------
    sigma : xr.DataArray
        Coordinate defined only for `nsigma` layers nearest the ocean surface.
    eta : xr.DataArray
        Height of the ocean surface (positive upwards) relative to ocean datum.
    depth : xr.DataArray
        Distance from ocean datum to sea floor (positive value).
    depth_c : xr.DataArray
        Constant.
    nsigma : xr.DataArray
        Layers nearest the ocean surface.
    zlev : xr.DataArray
        Coordinate defined only for `nlayer - nsigma` where `nlayer` is the size of the vertical coordinate.

    Returns
    -------
    xr.DataArray
        A DataArray with the height (positive upwards) relative to the ocean datum.

    Notes
    -----
    The description of this type of parametric vertical coordinate is defective in version 1.8 and earlier versions of the standard, in that it does not state what values the vertical coordinate variable should contain. Therefore, in accordance with the rules, all versions of the standard before 1.9 are deprecated for datasets that use the "ocean sigma over z" coordinate.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_sigma_over_z_coordinate
    """
    n, j, i = eta.shape

    k = sigma.shape[0]

    z = xr.DataArray(np.empty((n, k, j, i)), dims=("time", "lev", "lat", "lon"))

    z_sigma = eta + sigma * (np.minimum(depth_c, depth) + eta)

    z = xr.where(~np.isnan(sigma), z_sigma, z)

    z = xr.where(np.isnan(sigma), zlev, z)

    out_stdname = _derive_ocean_stdname(
        eta=eta.attrs, depth=depth.attrs, zlev=zlev.attrs
    )

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("nkji", nji=eta, k=sigma)

    return z.transpose(*output_order)


def ocean_double_sigma_coordinate(sigma, depth, z1, z2, a, href, k_c):
    """Ocean double sigma coordinate.

    Standard name: ocean_double_sigma_coordinate

    Parameters
    ----------
    sigma : xr.DataArray
        Dimensionless coordinate.
    depth : xr.DataArray
        Distance (positive value) from datum to the sea floor.
    z1 : xr.DataArray
        Constant with units of length.
    z2 : xr.DataArray
        Constant with units of length.
    a : xr.DataArray
        Constant with units of length.
    href : xr.DataArray
        Constant with units of length.
    k_c : xr.DataArray

    Returns
    -------
    xr.DataArray
        A DataArray with the height (positive upwards) relative to the datum.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_double_sigma_coordinate
    """
    k = sigma.shape[0]

    j, i = depth.shape

    f = 0.5 * (z1 + z2) + 0.5 * (z1 - z2) * np.tanh(2 * a / (z1 - z2) * (depth - href))

    z = xr.DataArray(np.empty((k, j, i)), dims=("lev", "lat", "lon"), name="z")

    z = xr.where(sigma.k <= k_c, sigma * f, z)

    z = xr.where(sigma.k > k_c, f + (sigma - 1) * (depth - f), z)

    out_stdname = _derive_ocean_stdname(depth=depth.attrs)

    z = z.squeeze().rename("z").assign_attrs(standard_name=out_stdname)

    output_order = derive_dimension_order("kji", ji=depth, k=sigma)

    return z.transpose(*output_order)
