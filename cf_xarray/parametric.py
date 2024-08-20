from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np
import xarray as xr
from xarray import DataArray

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


class ParamerticVerticalCoordinate(ABC):
    @classmethod
    @abstractmethod
    def from_terms(cls, terms: dict):
        pass

    @abstractmethod
    def decode(self):
        pass

    @property
    @abstractmethod
    def computed_standard_name(self):
        pass


class AtmosphereLnPressure(ParamerticVerticalCoordinate):
    """Atmosphere natural log pressure coordinate.

    Standard name: atmosphere_ln_pressure_coordinate

    Parameters
    ----------
    p0 : xr.DataArray
        Reference pressure.
    lev : xr.DataArray
        Vertical dimensionless coordinate.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#atmosphere-natural-log-pressure-coordinate
    """

    def __init__(self, p0, lev):
        self.p0 = p0
        self.lev = lev

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        p = self.p0 * np.exp(-self.lev)

        return p.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self):
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        p0, lev = get_terms(terms, "p0", "lev")

        return cls(p0, lev)


class AtmosphereSigma(ParamerticVerticalCoordinate):
    """Atmosphere sigma coordinate.

    Standard name: atmosphere_sigma_coordinate

    Parameters
    ----------
    sigma : xr.DataArray
        Vertical dimensionless coordinate.
    ps : xr.DataArray
        Horizontal surface pressure.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_atmosphere_sigma_coordinate
    """

    def __init__(self, sigma, ps, ptop):
        self.sigma = sigma
        self.ps = ps
        self.ptop = ptop

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        p = self.ptop + self.sigma * (self.ps - self.ptop)

        return p.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        sigma, ps, ptop = get_terms(terms, "sigma", "ps", "ptop")

        return cls(sigma, ps, ptop)


class AtmosphereHybridSigmaPressure(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_atmosphere_hybrid_sigma_pressure_coordinate
    """

    def __init__(self, b, ps, p0=None, a=None, ap=None):
        self.b = b
        self.ps = ps
        self.p0 = p0
        self.a = a
        self.ap = ap

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        if self.a is None:
            p = self.ap + self.b * self.ps
        else:
            p = self.a * self.p0 + self.b * self.ps

        return p.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        b, ps, p0, a, ap = get_terms(terms, "b", "ps", optional=("p0", "a", "ap"))

        if a is None and ap is None:
            raise KeyError(
                "Optional terms 'a', 'ap' are absent in the dataset, atleast one must be present."
            )

        if a is not None and ap is not None:
            raise Exception(
                "Both optional terms 'a' and 'ap' are present in the dataset, please drop one of them."
            )

        if a is not None and p0 is None:
            raise KeyError(
                "Optional term 'a' is present but 'p0' is absent in the dataset."
            )

        return cls(b, ps, p0, a, ap)


class AtmosphereHybridHeight(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#atmosphere-hybrid-height-coordinate
    """

    def __init__(self, a, b, orog):
        self.a = a
        self.b = b
        self.orog = orog

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z = self.a + self.b * self.orog

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        orog_stdname = self.orog.attrs["standard_name"]

        if orog_stdname == "surface_altitude":
            out_stdname = "altitude"
        elif orog_stdname == "surface_height_above_geopotential_datum":
            out_stdname = "height_above_geopotential_datum"

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        a, b, orog = get_terms(terms, "a", "b", "orog")

        return cls(a, b, orog)


class AtmosphereSleve(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_atmosphere_smooth_level_vertical_sleve_coordinate
    """

    def __init__(self, a, b1, b2, ztop, zsurf1, zsurf2):
        self.a = a
        self.b1 = b1
        self.b2 = b2
        self.ztop = ztop
        self.zsurf1 = zsurf1
        self.zsurf2 = zsurf2

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z = self.a * self.ztop + self.b1 * self.zsurf1 + self.b2 * self.zsurf2

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        ztop_stdname = self.ztop.attrs["standard_name"]

        if ztop_stdname == "altitude_at_top_of_atmosphere_model":
            out_stdname = "altitude"
        elif (
            ztop_stdname == "height_above_geopotential_datum_at_top_of_atmosphere_model"
        ):
            out_stdname = "height_above_geopotential_datum"

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        a, b1, b2, ztop, zsurf1, zsurf2 = get_terms(
            terms, "a", "b1", "b2", "ztop", "zsurf1", "zsurf2"
        )

        return cls(a, b1, b2, ztop, zsurf1, zsurf2)


class OceanSigma(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_sigma_coordinate
    """

    def __init__(self, sigma, eta, depth):
        self.sigma = sigma
        self.eta = eta
        self.depth = depth

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z = self.eta + self.sigma * (self.depth + self.eta)

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        out_stdname = _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        sigma, eta, depth = get_terms(terms, "sigma", "eta", "depth")

        return cls(sigma, eta, depth)


class OceanS(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate
    """

    def __init__(self, s, eta, depth, a, b, depth_c):
        self.s = s
        self.eta = eta
        self.depth = depth
        self.a = a
        self.b = b
        self.depth_c = depth_c

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        C = (1 - self.b) * np.sinh(self.a * self.s) / np.sinh(self.a) + self.b * (
            np.tanh(self.a * (self.s + 0.5)) / 2 * np.tanh(0.5 * self.a) - 0.5
        )

        z = (
            self.eta * (1 + self.s)
            + self.depth_c * self.s
            + (self.depth - self.depth_c) * C
        )

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        s, eta, depth, a, b, depth_c = get_terms(
            terms, "s", "eta", "depth", "a", "b", "depth_c"
        )

        return cls(s, eta, depth, a, b, depth_c)


class OceanSG1(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate_generic_form_1
    """

    def __init__(self, s, c, eta, depth, depth_c):
        self.s = s
        self.c = c
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        S = self.depth_c * self.s + (self.depth - self.depth_c) * self.c

        z = S + self.eta * (1 + self.s / self.depth)

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        s, c, eta, depth, depth_c = get_terms(
            terms, "s", "c", "eta", "depth", "depth_c"
        )

        return cls(s, c, eta, depth, depth_c)


class OceanSG2(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_s_coordinate_generic_form_2
    """

    def __init__(self, s, c, eta, depth, depth_c):
        self.s = s
        self.c = c
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        S = (self.depth_c * self.s + self.depth * self.c) / (self.depth_c + self.depth)

        z = self.eta + (self.eta + self.depth) * S

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        s, c, eta, depth, depth_c = get_terms(
            terms, "s", "c", "eta", "depth", "depth_c"
        )

        return cls(s, c, eta, depth, depth_c)


class OceanSigmaZ(ParamerticVerticalCoordinate):
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

    Notes
    -----
    The description of this type of parametric vertical coordinate is defective in version 1.8 and earlier versions of the standard, in that it does not state what values the vertical coordinate variable should contain. Therefore, in accordance with the rules, all versions of the standard before 1.9 are deprecated for datasets that use the "ocean sigma over z" coordinate.

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_sigma_over_z_coordinate
    """

    def __init__(self, sigma, eta, depth, depth_c, nsigma, zlev):
        self.sigma = sigma
        self.eta = eta
        self.depth = depth
        self.depth_c = depth_c
        self.nsigma = nsigma
        self.zlev = zlev

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z_shape = list(self.eta.shape)

        z_shape.insert(1, self.sigma.shape[0])

        z_dims = list(self.eta.dims)

        z_dims.insert(1, self.sigma.dims[0])

        z = xr.DataArray(np.empty(z_shape), dims=z_dims)

        z_sigma = self.eta + self.sigma * (
            np.minimum(self.depth_c, self.depth) + self.eta
        )

        z = xr.where(~np.isnan(self.sigma), z_sigma, z)

        z = xr.where(np.isnan(self.sigma), self.zlev, z)

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(
            eta=self.eta.attrs, depth=self.depth.attrs, zlev=self.zlev.attrs
        )

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        sigma, eta, depth, depth_c, nsigma, zlev = get_terms(
            terms, "sigma", "eta", "depth", "depth_c", "nsigma", "zlev"
        )

        return cls(sigma, eta, depth, depth_c, nsigma, zlev)


class OceanDoubleSigma(ParamerticVerticalCoordinate):
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

    References
    ----------
    Please refer to the CF conventions document :
      1. https://cfconventions.org/cf-conventions/cf-conventions.html#_ocean_double_sigma_coordinate
    """

    def __init__(self, sigma, depth, z1, z2, a, href, k_c):
        self.sigma = sigma
        self.depth = depth
        self.z1 = z1
        self.z2 = z2
        self.a = a
        self.href = href
        self.k_c = k_c

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        f = 0.5 * (self.z1 + self.z2) + 0.5 * (self.z1 - self.z2) * np.tanh(
            2 * self.a / (self.z1 - self.z2) * (self.depth - self.href)
        )

        # shape k, j, i
        z_shape = self.sigma.shape + self.depth.shape

        z_dims = self.sigma.dims + self.depth.dims

        z = xr.DataArray(np.empty(z_shape), dims=z_dims, name="z")

        z = xr.where(self.sigma.k <= self.k_c, self.sigma * f, z)

        z = xr.where(
            self.sigma.k > self.k_c, f + (self.sigma - 1) * (self.depth - f), z
        )

        return z.squeeze().assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        sigma, depth, z1, z2, a, href, k_c = get_terms(
            terms, "sigma", "depth", "z1", "z2", "a", "href", "k_c"
        )

        return cls(sigma, depth, z1, z2, a, href, k_c)


TRANSFORM_FROM_STDNAME = {
    "atmosphere_ln_pressure_coordinate": AtmosphereLnPressure,
    "atmosphere_sigma_coordinate": AtmosphereSigma,
    "atmosphere_hybrid_sigma_pressure_coordinate": AtmosphereHybridSigmaPressure,
    "atmosphere_hybrid_height_coordinate": AtmosphereHybridHeight,
    "atmosphere_sleve_coordinate": AtmosphereSleve,
    "ocean_sigma_coordinate": OceanSigma,
    "ocean_s_coordinate": OceanS,
    "ocean_s_coordinate_g1": OceanSG1,
    "ocean_s_coordinate_g2": OceanSG2,
    "ocean_sigma_z_coordinate": OceanSigmaZ,
    "ocean_double_sigma_coordinate": OceanDoubleSigma,
}


def get_terms(
    terms: dict[str, DataArray], *required, optional: Sequence[str] = None
) -> DataArray:
    if optional is None:
        optional = []

    selected_terms = []

    for term in required + tuple(optional):
        da = None

        try:
            da = terms[term]
        except KeyError:
            if term not in optional:
                raise KeyError(f"Required term {term} is absent in the dataset.") from None

        selected_terms.append(da)

    return selected_terms
