import abc
from typing import Optional
import numpy as np
import astropy.units as u
from astropy.coordinates import Angle
import astropy.constants as const
from astropy.modeling.blackbody import BlackBody1D
from .calib import Sun
from .data import dataclass_input
from .data.phys import Phys
from .data.ephem import Ephem
from .units.typing import (
    SpectralFluxDensityQuantity,
    SpectralRadianceQuantity,
    SpectralQuantity,
    UnitLike,
)

from numpy import pi


class Surface(abc.ABC):
    """Base class for all surface characteristics.


    Parameters
    ----------
    phys : Phys
        Surface physical parameters, e.g., albedo.

    """

    @dataclass_input()
    def __init__(self, phys: Phys):
        self.phys = phys

    @staticmethod
    def _validate_normal_angle(a: u.Quantity["angle"]) -> Angle:
        """Maps angle to 0 to 180."""
        if a.unit is u.sr:
            wrap = np.pi * u.sr
            # a_max = np.pi / 2 * u.sr
        elif a.unit is u.deg:
            wrap = 180 * u.deg
            # a_max = 90 * u.deg
        else:
            wrap = (180 * u.deg).to(a.unit)
            # a_max = (90 * u.deg).to(a.unit)

        # return np.abs(np.minimum(Angle(a).wrap_at(wrap), a_max))
        return np.abs(Angle(a).wrap_at(wrap))

    @abc.abstractmethod
    def reflectance(
        self,
        i: u.Quantity["angle"],
        e: u.Quantity["angle"],
        phi: u.Quantity["angle"],
    ) -> float:
        r"""Reflectance.

        The surface is illuminated by incident flux density, :math:`F_i`, at an
        angle of :math:`i`, and emitted toward an angle of :math:`e`, measured
        from the surface normal direction.  :math:`\phi` is the
        Sun-target-observer (phase) angle.


        Parameters
        ----------

        """

        pass

    @abc.abstractmethod
    def absorptance(
        self,
        i: u.Quantity["angle"],
    ) -> float:
        r"""Absorption of incident light.

        The surface is illuminated by incident flux density, :math:`F_i`, at an
        angle of :math:`i`, measured from the surface normal direction.


        Parameters
        ----------

        """

        pass

    @abc.abstractmethod
    def emittance(
        self,
        e: u.Quantity["angle"],
        phi: u.Quantity["angle"],
    ) -> float:
        r"""Emission of light.

        The surface is observed at an angle of :math:`e`, measured from the
        surface normal direction, and at a solar phase angle of :math:`phi`.


        Parameters
        ----------

        """

        pass

    @abc.abstractmethod
    def spectral_radiance(
        self,
        F_i: SpectralFluxDensityQuantity,
        n: np.ndarray[3],
        rs: u.Quantity["length"],
        ro: u.Quantity["length"],
    ) -> SpectralRadianceQuantity:
        """Observed radiance from a surface.


        Parameters
        ----------
        F_i : `sbpy.units.Quantity`
            Incident light, spectral flux density.

        n : `numpy.ndarray`
            Surface normal vector.

        rs : `sbpy.units.Quantity`
            Radial vector from the surface to the light source.

        ro : `sbpy.units.Quantity`
            Radial vector from the surface to the observer.

        """

        n_hat = np.linalg.norm(n.value)
        rs_hat = np.linalg.norm(rs.value)
        ro_hat = np.linalg.norm(ro.value)

        i = np.arccos(np.dot(n_hat, rs_hat))
        e = np.arccos(np.dot(n_hat, ro_hat))
        phi = np.arccos(np.dot(rs_hat, ro_hat))

        return F_i * self.reflectance(i, e, phi) / u.sr


class LambertianSurface(Surface):
    """Lambertian surface.


    Parameters
    ----------
    phys : Phys
        Surface physical parameters, e.g., albedo.

    """

    @u.quantity_input
    def reflectance(
        self, i: u.Quantity["angle"], e: u.Quantity["angle"], *args
    ) -> float:
        _i = self._validate_normal_angle(i)
        if u.isclose(_i, 90 * u.deg) or (_i > 90 * u.deg):
            return 0 * u.dimensionless_unscaled

        return self.phys["A"] * np.cos(_i) * self.emittance(e)

    @u.quantity_input
    def absorptance(self, i: u.Quantity["angle"]) -> float:
        _i = self._validate_normal_angle(i)
        if u.isclose(_i, 90 * u.deg) or (_i > 90 * u.deg):
            return 0 * u.dimensionless_unscaled

        return (1 - self.phys["A"]) * np.cos(_i)

    @u.quantity_input
    def emittance(self, e: u.Quantity["angle"]) -> float:
        _e = self._validate_normal_angle(e)
        if u.isclose(_e, 90 * u.deg) or (_e > 90 * u.deg):
            return 0 * u.dimensionless_unscaled

        return np.cos(_e)


class SurfaceReflectance(Surface):
    """Reflectance from a surface illuminated by sunlight."""

    @u.quantity_input
    def radiance(
        self,
        wave_freq: SpectralQuantity,
        n: np.ndarray[3],
        rs: u.Quantity["length"],
        ro: u.Quantity["length"],
    ) -> SpectralFluxDensityQuantity:

        sun = Sun.from_default()
        F_i = sun.observe(wave_freq)
        return super().radiance(F_i, n, rs, ro)


class SurfaceThermalEmission(Surface):
    """Thermal emission from a surface illuminated by sunlight."""

    @u.quantity_input
    def radiance(
        self,
        wave_freq: SpectralQuantity,
        n: np.ndarray[3],
        rs: u.Quantity["length"],
        ro: u.Quantity["length"],
        unit: UnitLike = "W/(m2 um)",
    ) -> SpectralFluxDensityQuantity:

        n_hat = np.linalg.norm(n.value)
        rs_hat = np.linalg.norm(rs.value)
        ro_hat = np.linalg.norm(ro.value)

        i = np.arccos(np.dot(n_hat, rs_hat))
        e = np.arccos(np.dot(n_hat, ro_hat))
        phi = np.arccos(np.dot(rs_hat, ro_hat))

        rh2 = np.dot(rs, rs)
        sun = const.L / (4 * pi * rh2)
        epsilon = self.phys["emissivity"]
        T = (self.absorptance(i) * sun / epsilon / const.sigma_sb) ** (1 / 4)

        return epsilon * BlackBody1D(temperature=T) * self.emittance(e, phi)

    # @u.quantity_input()
    # def surface_radiance(
    #     self,
    #     mu0: u.Quantity["angle"],
    #     mu: u.Quantity["angle"],
    #     F_i: SpectralFluxDensityQuantity,
    # ) -> SpectralRadianceQuantity:
    #     r"""Surface spectral radiance.

    #     The surface is illuminated by the incident flux density, :math:`F_i`, at
    #     an angle of :math:`\theta_i`, and emitted toward an angle of
    #     :math:`\theta_e`.  Angles are measured from the surface normal
    #     direction.

    #     Parameters
    #     ----------
    #     mu0 : `~astropy.units.Quantity`
    #         Cosine of the angle of incidence, measured from the surface normal.

    #     mu : `~astropy.units.Quantity`
    #         Cosine of the angle of emittance, measured from the surface normal.

    #     F_i : `~astropy.units.Quantity`
    #         Incident radiation in units of flux density.

    #     """

    #     if mu0 < 0 or mu < 0:
    #         return 0 * F_i.unit / u.sr

    #     return self.phys["A"] * F_i * mu0 * mu / np.pi / u.sr

    # @u.quantity_input()
    # def surface_flux_density(
    #     self,
    #     theta_e: u.Quantity["angle"],
    #     delta: u.Quantity["length"],
    #     radiance: SpectralRadianceQuantity,
    # ) -> SpectralFluxDensityQuantity:
    #     r"""Observed spectral flux density.

    #     Parameters
    #     ----------
    #     theta_e : `~astropy.units.Quantity`
    #         Angle of emittance, measured from the surface normal.

    #     radiance : `~astropy.units.Quantity`
    #         Surface emitted radiance.

    #     """

    #     _theta_e = np.maximum(
    #         np.minimum(Angle(theta_e).wrap_at(180 * u.deg), 90 * u.deg), -90 * u.deg
    #     )
    #     if np.isclose(np.abs(_theta_e), 90 * u.deg):
    #         return 0 * radiance.unit * u.sr

    #     return radiance * np.cos(np.maximum(_theta_e, 90 * u.deg)) / * u.sr
