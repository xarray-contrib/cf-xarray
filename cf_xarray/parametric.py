from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from xarray import DataArray

OCEAN_STDNAME_MAP = {
    "altitude": {
        "zlev": "altitude",
        "eta": "sea_surface_height_above_geoid",
        "depth": "sea_floor_depth_below_geoid",
    },
    "height_above_geopotential_datum": {
        "zlev": "height_above_geopotential_datum",
        "eta": "sea_surface_height_above_geopotential_datum",
        "depth": "sea_floor_depth_below_geopotential_datum",
    },
    "height_above_reference_ellipsoid": {
        "zlev": "height_above_reference_ellipsoid",
        "eta": "sea_surface_height_above_reference_ellipsoid",
        "depth": "sea_floor_depth_below_reference_ellipsoid",
    },
    "height_above_mean_sea_level": {
        "zlev": "height_above_mean_sea_level",
        "eta": "sea_surface_height_above_mean_sea_level",
        "depth": "sea_floor_depth_below_mean_sea_level",
    },
}


def _derive_ocean_stdname(*, zlev=None, eta=None, depth=None):
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
    search_term = ""
    search_vars = {"zlev": zlev, "eta": eta, "depth": depth}

    for x, y in sorted(search_vars.items(), key=lambda x: x[0]):
        if y is None:
            continue

        try:
            search_term = f"{search_term}{y['standard_name']}"
        except TypeError:
            raise ValueError(
                f"The values for {', '.join(sorted(search_vars.keys()))} cannot be `None`."
            ) from None
        except KeyError:
            raise ValueError(
                f"The standard name for the {x!r} variable is not available."
            ) from None

    for x, y in OCEAN_STDNAME_MAP.items():
        check_term = "".join(
            [
                y[i]
                for i, j in sorted(search_vars.items(), key=lambda x: x[0])
                if j is not None
            ]
        )

        if search_term == check_term:
            found_stdname = x

            break

    if found_stdname is None:
        stdnames = ", ".join(
            [
                y["standard_name"]
                for _, y in sorted(search_vars.items(), key=lambda x: x[0])
                if y is not None
            ]
        )

        raise ValueError(
            f"Could not derive standard name from combination of {stdnames}."
        )

    return found_stdname


class ParametricVerticalCoordinate(ABC):
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


@dataclass
class AtmosphereLnPressure(ParametricVerticalCoordinate):
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

    p0: DataArray
    lev: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        p = self.p0 * np.exp(-self.lev)

        return p.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self):
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "p0", "lev"))


@dataclass
class AtmosphereSigma(ParametricVerticalCoordinate):
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

    sigma: DataArray
    ps: DataArray
    ptop: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        p = self.ptop + self.sigma * (self.ps - self.ptop)

        return p.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "sigma", "ps", "ptop"))


@dataclass
class AtmosphereHybridSigmaPressure(ParametricVerticalCoordinate):
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

    b: DataArray
    ps: DataArray
    p0: DataArray
    a: DataArray
    ap: DataArray

    def __post_init__(self):
        if self.a is None and self.ap is None:
            raise KeyError(
                "Optional terms 'a', 'ap' are absent in the dataset, atleast one must be present."
            )

        if self.a is not None and self.ap is not None:
            raise ValueError(
                "Both optional terms 'a' and 'ap' are present in the dataset, please drop one of them."
            )

        if self.a is not None and self.p0 is None:
            raise KeyError(
                "Optional term 'a' is present but 'p0' is absent in the dataset."
            )

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        if self.a is None:
            p = self.ap + self.b * self.ps  # type: ignore[unreachable]
        else:
            p = self.a * self.p0 + self.b * self.ps

        return p.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return "air_pressure"

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "b", "ps", optional=("p0", "a", "ap")))


@dataclass
class AtmosphereHybridHeight(ParametricVerticalCoordinate):
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

    a: DataArray
    b: DataArray
    orog: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z = self.a + self.b * self.orog

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        orog_stdname = self.orog.attrs["standard_name"]

        if orog_stdname == "surface_altitude":
            out_stdname = "altitude"
        elif orog_stdname == "surface_height_above_geopotential_datum":
            out_stdname = "height_above_geopotential_datum"
        else:
            raise ValueError(
                f"Unknown standard name for hybrid height coordinate: {orog_stdname!r}"
            )

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "a", "b", "orog"))


@dataclass
class AtmosphereSleve(ParametricVerticalCoordinate):
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

    a: DataArray
    b1: DataArray
    b2: DataArray
    ztop: DataArray
    zsurf1: DataArray
    zsurf2: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z = self.a * self.ztop + self.b1 * self.zsurf1 + self.b2 * self.zsurf2

        return z.assign_attrs(standard_name=self.computed_standard_name)

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
        else:
            raise ValueError(f"Unknown standard name: {ztop_stdname!r}")

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "a", "b1", "b2", "ztop", "zsurf1", "zsurf2"))


@dataclass
class OceanSigma(ParametricVerticalCoordinate):
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

    sigma: DataArray
    eta: DataArray
    depth: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z = self.eta + self.sigma * (self.depth + self.eta)

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        out_stdname = _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

        return out_stdname

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "sigma", "eta", "depth"))


@dataclass
class OceanS(ParametricVerticalCoordinate):
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

    s: DataArray
    eta: DataArray
    depth: DataArray
    a: DataArray
    b: DataArray
    depth_c: DataArray

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

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "s", "eta", "depth", "a", "b", "depth_c"))


@dataclass
class OceanSG1(ParametricVerticalCoordinate):
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

    s: DataArray
    c: DataArray
    eta: DataArray
    depth: DataArray
    depth_c: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        S = self.depth_c * self.s + (self.depth - self.depth_c) * self.c

        z = S + self.eta * (1 + self.s / self.depth)

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "s", "c", "eta", "depth", "depth_c"))


@dataclass
class OceanSG2(ParametricVerticalCoordinate):
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

    s: DataArray
    c: DataArray
    eta: DataArray
    depth: DataArray
    depth_c: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        S = (self.depth_c * self.s + self.depth * self.c) / (self.depth_c + self.depth)

        z = self.eta + (self.eta + self.depth) * S

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(eta=self.eta.attrs, depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "s", "c", "eta", "depth", "depth_c"))


@dataclass
class OceanSigmaZ(ParametricVerticalCoordinate):
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

    sigma: DataArray
    eta: DataArray
    depth: DataArray
    depth_c: DataArray
    nsigma: DataArray
    zlev: DataArray

    def decode(self) -> xr.DataArray:
        """Decode coordinate.

        Returns
        -------
        xr.DataArray
            Decoded parametric vertical coordinate.
        """
        z_sigma = self.eta + self.sigma * (
            np.minimum(self.depth_c, self.depth) + self.eta
        )

        z = xr.where(np.isnan(self.sigma), self.zlev, z_sigma)

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(
            eta=self.eta.attrs, depth=self.depth.attrs, zlev=self.zlev.attrs
        )

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(
            **get_terms(terms, "sigma", "eta", "depth", "depth_c", "nsigma", "zlev")
        )


@dataclass
class OceanDoubleSigma(ParametricVerticalCoordinate):
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

    sigma: DataArray
    depth: DataArray
    z1: DataArray
    z2: DataArray
    a: DataArray
    href: DataArray
    k_c: DataArray

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

        z = xr.where(
            self.sigma.k <= self.k_c,
            self.sigma * f,
            f + (self.sigma - 1) * (self.depth - f),
        )

        return z.assign_attrs(standard_name=self.computed_standard_name)

    @property
    def computed_standard_name(self) -> str:
        """Computes coordinate standard name."""
        return _derive_ocean_stdname(depth=self.depth.attrs)

    @classmethod
    def from_terms(cls, terms: dict):
        """Create coordinate from terms."""
        return cls(**get_terms(terms, "sigma", "depth", "z1", "z2", "a", "href", "k_c"))


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
    terms: dict[str, DataArray], *required, optional: Sequence[str] | None = None
) -> dict[str, DataArray]:
    if optional is None:
        optional = []

    selected_terms = {}

    for term in required + tuple(optional):
        da = None

        try:
            da = terms[term]
        except KeyError:
            if term not in optional:
                raise KeyError(
                    f"Required term {term} is absent in the dataset."
                ) from None

        selected_terms[term] = da

    return selected_terms  # type: ignore[return-value]
