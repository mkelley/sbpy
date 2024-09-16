# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Thermal Module

created on June 27, 2017
"""

__all__ = ["ThermalClass", "NonRotThermalModel", "FastRotThermalModel"]

import abc
from typing import Optional, Union
import numpy as np
from scipy.integrate import dblquad
import astropy.units as u
from astropy.units.typing import QuantityLike
import astropy.constants as const
from astropy.coordinates import Angle
from astropy.coordinates import SphericalRepresentation, CartesianRepresentation
from astropy.modeling.models import BlackBody
from ..data import Phys, Obs, Ephem, dataclass_input, quantity_to_dataclass
from ..surface import Surface
from ..units.typing import (
    UnitLike,
    SpectralQuantity,
    SpectralFluxDensityQuantity,
    SpectralRadianceQuantity,
)
from ..exceptions import SbpyWarning


class SurfaceThermalEmission(Surface, abc.ABCMeta):
    """Abstract base class for surface thermal emission.

    The surface is illuminated by sunlight.


    Parameters
    ----------
    phys : `~sbpy.data.Phys` or dictionary-like
        Physical properties of the surface:
            - bolometric (Bond) albedo (e.g., ``A``, default 0.1)
            - infrared emissivity (`emissivity` or ``epsilon``, default 0.9)

    """

    @dataclass_input(phys=Phys)
    def __init__(self, phys: Phys):
        self.phys: Phys = phys

        # apply default values
        if "A" not in phys:
            # Do not use geometric albedo, but raise a warning and use the
            # default bolometric albedo
            if "Ap" in self.phys:
                raise SbpyWarning(
                    "Geometric albedo was provided but ignored.  Instead, a default "
                    "bolometric albedo will be used."
                )
            self.phys["A"] = 0.1
        if "emissivity" not in phys:
            self.phys["emissivity"] = 0.9

    @property
    def bolometric_albedo(self) -> u.Quantity["dimensionless"]:
        return self.phys["A"]

    @property
    def emissivity(self) -> u.Quantity["dimensionless"]:
        return self.phys["emissivity"]

    @abc.abstractmethod
    def T(
        self,
        F_i: SpectralFluxDensityQuantity,
    ) -> u.Quantity["temperature"]:
        """Temperature of the surface given incident sunlight.


        Parameters
        ----------
        F_i : `astropy.units.Quantity`
            Total incident flux density.


        Returns
        -------
        T : `astropy.units.Quantity`

        """

        pass

    @u.quantity_input()
    def surface_radiance(
        self,
        wave_freq: SpectralQuantity,
        theta_i: u.Quantity["angle"],
        F_i: SpectralFluxDensityQuantity,
    ) -> Union[
        u.Quantity["surface_brightness"],
        u.Quantity["surface_brightness_wav"],
    ]:
        unit = F_i.unit / u.sr
        T = self.T(F_i * np.cos(theta_i))

        if u.isclose(T, 0 * u.K):
            radiance = 0.0 * unit
        else:
            radiance = BlackBody(T, scale=1 * unit)(wave_freq)

        return radiance


class InstantaneousThermalEquilibrium(SurfaceThermalEmission):
    """Instantaneous thermal equilibrium with sunlight.

    The surface is illuminated by sunlight.


    Parameters
    ----------
    phys : `~sbpy.data.Phys` or dictionary-like
        Physical properties of the surface:
            - bolometric (Bond) albedo (e.g., ``A``, default 0.1)
            - infrared emissivity (e.g., ``emissivity``, default 0.9)
            - infrared beaming parameter (e.g., ``eta``, default 0.756)
        See :doc:`/sbpy/data/fieldnames` for more on field names.

    """

    @dataclass_input
    def __init__(self, phys: Phys):

        super().__init__(self, phys)

        # apply default values
        if "eta" not in phys:
            self.phys["eta"] = 0.756

        # solar flux density at 1 au
        F_0 = (const.L_sun / (4 * np.pi * const.au**2)).to("W/m2")

        # sub-solar temperature at 1 au
        self.T_sub_solar_0 = (
            ((1 - self.A) * self.F_0 / (self.eta * self.emissivity * const.sigma_sb))
            ** (1 / 4)
        ).to("K")

    @property
    def eta(self) -> u.Quantity["dimensionless"]:
        return self.phys["eta"]

    @u.quantity_input()
    def T(self, F_i: SpectralFluxDensityQuantity) -> u.Quantity["temperature"]:
        """Temperature of the surface given incident sunlight.


        Parameters
        ----------
        F_i : `astropy.units.Quantity`
            Total incident flux density.


        Returns
        -------
        T : `astropy.units.Quantity`

        """

        return self.T_sub_solar_0 * (F_i / self.F_0) ** (1 / 4)


#     def total_spectral_fluxd(
#         self,
#         wave_freq: SpectralUnit,
#         eph: Ephem,
#         unit: Optional[UnitLike] = u.Unit("W/(m2 um)"),
#     ) -> Union[
#         u.Quantity["spectral_flux_density"],
#         u.Quantity["spectral_flux_density_wav"],
#     ]:
#         """Total spectral flux density from object as seen by an observer.


#         Parameters
#         ----------

#         wave_freq : `astropy.unit.Quantity`
#             Wavelength or frequency of calculation.

#         unit : str or `astropy.units.Unit`, optional
#             Units of the return value, e.g., W/(m2 um).


#         Returns
#         -------

#         fluxd : `astropy.unit.Quantity`
#             Spectral flux density in units of `unit`.

#         """

#         _unit = u.Unit(unit)
#         radiance_unit = _unit / u.sr

#         model_xform = self.model_xform(eph)

#         def integrand(sub_lon, sub_lat):
#             """Integration in sub-observer coordinates.

#             lon, lat = 0, 0 is the sub-observer point.

#             """

#             sub_observer = SphericalRepresentation(
#                 lon=sub_lon, lat=sub_lat, distance=1.0
#             )

#             model_coords = sub_observer.transform(model_xform)

#             radiance = self.surface_spectral_radiance(
#                 model_coords.lon,
#                 model_coords.lat,
#                 wave_freq,
#                 eph,
#                 unit=radiance_unit,
#             )

#             return radiance * np.cos(sub_lon) ** 2 * np.cos(sub_lat) * u.sr

#         pi = np.pi
#         result = dblquad(integrand, (-pi / 2, pi / 2), (-pi / 2, pi / 2))
#         breakpoint()
#         return result


# class Sphere:
#     def model_xform():


# class NEATM(
#     InstantaneousEquilibriumTemperature, LambertianSurface, SurfaceThermalEmission
# ):
#     def __init__(self, rh):
#         self.rh = rh

#     def fluxd_surface_point(self, wave, lon, lat):
#         mu = np.cos(lon) * np.cos(lat)
#         return super().fluxd_surface_point(wave, mu)


# class ThermalClass(abc.ABC):
#     """Abstract base class for thermal models.

#     This class implements the basic calculation for thermal models, such as
#     integration of total flux based on a temperature distribution.


#     Parameters
#     ----------
#     rh : `~sbpy.data.Ephem`, dictionary-like, or `~astropy.units.Quantity`
#         Heliocentric distance to evaluate the model.

#     phys : `~sbpy.data.Phys` or dictionary-like
#         Physical properties of the object:
#           - size as radius or diameter (e.g., `R` or `D`, required)
#           - bolometric (Bond) albdo (e.g., `A`, default 0.1)
#           - emissivity (`emissivity` or `epsilon`, default 0.9)
#           - infrared beaming parameter (`eta`, default 1.0)
#         Refer to the data class :ref:`field name documentation <field name
#         list>` for details on parameter names and units.

#     """

#     @quantity_to_dataclass(eph=(Ephem, "rh"))
#     @dataclass_input(phys=Phys)
#     def __init__(self, eph: Ephem, phys: Phys):
#         self.r: u.Quantity["length"] = u.Quantity(eph["rh"], u.au)
#         self.phys: Phys = phys

#         # apply default values
#         if "A" not in phys:
#             self.phys["A"] = 0.1
#         if "emissivity" not in phys:
#             self.phys["emissivity"] = 0.9
#         if "eta" not in phys:
#             self.phys["eta"] = 1.0

#     @property
#     def bolometric_albedo(self) -> u.Quantity["dimensionless"]:
#         return self.phys["A"]

#     @property
#     def emissivity(self) -> u.Quantity["dimensionless"]:
#         return self.phys["emissivity"]

#     @property
#     def eta(self) -> u.Quantity["dimensionless"]:
#         return self.phys["eta"]

#     def T(
#         self, lon: u.Quantity["angle"], lat: u.Quantity["angle"]
#     ) -> u.Quantity["temperature"]:
#         """Temperature at a point on the surface of an object.


#         Parameters
#         ----------
#         lon : `astropy.coordinates.Angle` or equivalent
#             Longitude.

#         lat : `astropy.coordinates.Angle` or equivalent
#             Latitude.

#         """

#         # Needs to be overridden in subclasses.  This function needs to be able
#         # to return a value for the full range of lon and lat, i.e., include the
#         # night side of an object.

#         pass

#     @u.quantity_input()
#     def fluxd_surface_point(
#         self,
#         wave_freq: SpectralUnit,
#         lon: u.Quantity["angle"],
#         lat: u.Quantity["angle"],
#         unit: Optional[Union[str, u.Unit]] = u.Unit("W/(m2 um)"),
#     ) -> Union[
#         u.Quantity["spectral_flux_density"],
#         u.Quantity["spectral_flux_density_wav"],
#     ]:
#         """Spectral radiance of a point on the surface.


#         Parameters
#         ----------

#         wave_freq : `astropy.unit.Quantity`
#             Wavelength or frequency of calculation.

#         lon : `astropy.units.Quantity` or `astropy.coordinates.Angle`
#             Body-fixed longitude.

#         lat : `astropy.units.Quantity` or `astropy.coordinates.Angle`
#             Body-fixed latitude.

#         unit : str or `astropy.units.Unit`
#             Spectral radiance units of the integral function.


#         Returns
#         -------

#         fluxd : `astropy.unit.Quantity`
#             Spectral flux density in units of `unit`.

#         """

#         T = self.T(lon, lat)
#         unit = u.Unit(unit)

#         if u.isclose(T, 0 * u.K):
#             fluxd = 0.0 * unit
#         else:
#             fluxd = BlackBody(T, scale=1 * unit)(wave_freq)

#         return fluxd

#     @u.quantity_input()
#     def fluxd_total(
#         self,
#         wave_freq: SpectralUnit,
#         lon: u.Quantity["angle"],
#         lat: u.Quantity["angle"],
#         unit: Optional[Union[str, u.Unit]] = u.Unit("W/(m2 um)"),
#         body_xform: Optional[np.ndarray[(3, 3), float]] = None,
#     ) -> Union[
#         u.Quantity["spectral_flux_density"],
#         u.Quantity["spectral_flux_density_wav"],
#     ]:
#         """Total spectral radiance as seen by an observer.


#         Parameters
#         ----------

#         wave_freq : `astropy.unit.Quantity`
#             Wavelength or frequency of calculation.

#         lon : `astropy.units.Quantity` or `astropy.coordinates.Angle`
#             Sub-observer longitude.

#         lat : `astropy.units.Quantity` or `astropy.coordinates.Angle`
#             Sub-observer latitude.

#         unit : str or `astropy.units.Unit`
#             Spectral radiance units of the integral function.

#         body_xform : `numpy.ndarray`, optional
#             3D transformation matrix, shape=(3, 3), to convert sub-observer
#             coordinates into body-fixed coordinates.  By default
#             `sub_observer_to_body_fixed` is used.  See for
#             `sub_observer_to_body_fixed` more details.


#         Returns
#         -------

#         fluxd : `astropy.unit.Quantity`
#             Spectral flux density in units of `unit`.

#         """

#         # For spectral flux per unit frequency integrate in Hz
#         # For spectral flux per unit wavelength integrate in m

#         _unit = u.Unit(unit)
#         if unit.is_equivalent("W/(m2 Hz)"):
#             _unit = u.Unit("W/(m2 Hz)")
#             x = wave_freq.to("Hz", u.spectral())
#         elif unit.is_equivalent("W/m3"):
#             _unit = u.Unit("W/m3")
#             x = wave_freq.to("m", u.spectral())
#         else:
#             raise ValueError("`unit` must be equivalent to W/(m2 Hz) or W/m3")

#         if body_xform is None:
#             body_xform = self.sub_observer_to_body_fixed(lon, lat)

#         def integrand(sub_lon, sub_lat):
#             sub_observer = SphericalRepresentation(
#                 lon=sub_lon, lat=sub_lat, distance=1.0
#             )
#             body_fixed = sub_observer.transform(body_xform)
#             return (
#                 self.fluxd_surface_point(
#                     wave_freq, body_fixed.lon, body_fixed.lat, unit=_unit
#                 ).value
#                 * np.cos(sub_lon) ** 2
#                 * np.cos(sub_lat)
#             )

#         pi = np.pi
#         result = dblquad(integrand, (-pi / 2, pi / 2), (-pi / 2, pi / 2))
#         breakpoint()

#         #

#     @staticmethod
#     @u.quantity_input()
#     def sub_observer_to_body_fixed(
#         sub_lon: u.Quantity["angle"], sub_lat: u.Quantity["angle"]
#     ) -> np.ndarray[(3, 3)]:
#         """Transformation matrix from sub-observer to body-fixed frame.

#         The numerical integration to calculate total flux density is performed
#         in a reference frame where the sub-observer point is at ``(lon, lat) =
#         (0, 0)``.  This matrix supports the transformation from this frame to
#         the body-fixed frame to facilitate the calculation of surface
#         temperature.

#         .. math::

#             b = M \dot o

#         Where ``b`` is a vector in the body fixed frame, ``M`` is the
#         transformation matrix, and ``o`` is a vector in the sub-observer frame.


#         Parameters
#         ----------

#         sub_lon : `sbpy.units.Quantity`
#             Sub-observer longitude.

#         sub_lat : `sbpy.units.Quantity`
#             Sub-observer latitude.


#         Returns
#         -------

#         M : `numpy.ndarray`
#             The transformation matrix.

#         """

#         sub_observer = SphericalRepresentation(lon=sub_lon, lat=sub_lat, distance=1.0)
#         if u.isclose(sub_observer.lat, 90 * u.deg):  # north pole
#             M = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
#         elif u.isclose(sub_observer.lat, -90 * u.deg):  # south pole
#             M = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
#         else:  # anywhere else
#             M = twovec(sub_observer.to_cartesian().get_xyz(), 0, [0, 0, 1], 2)

#         return M

# los = sph2xyz(sublon.to_value("rad"), sublat.to_value("rad"))
# if np.isclose(np.linalg.norm(np.cross(sub_observer, [0, 0, 1])), 0):
#     if sub_lat.value > 0:
#         m = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
#     else:
#         m = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
# else:
#     m = twovec(
#         sph2xyz(sub_lon.to_value("rad"), sub_lat.to_value("rad")), 0, [0, 0, 1], 2
#     ).T

# @u.quantity_input(
#     wave_freq=u.m, delta=u.au, lon=u.deg, lat=u.deg, equivalencies=u.spectral()
# )
# def fluxd_surface_point(
#     self,
#     wave_freq,
#     delta,
#     sublon,
#     sublat,
#     unit="W m-2 um-1",
#     error=False,
#     epsrel=1e-3,
#     **kwargs
# ):
#     """Model thermal flux density of an object.

#     Parameters
#     ----------
#     wave_freq : u.Quantity
#         Wavelength or frequency of observations
#     delta : `~sbpy.data.Ephem`, dict_like, number, or
#         `~astropy.units.Quantity`
#         If `~sbpy.data.Ephem` or dict_like, ephemerides of the object that
#         can include the observer distance via keywords `delta`.  If float
#         or `~astropy.units.Quantity`, then the observer distance of an
#         object.  If observer distance is not found, then it will be
#         assumed to be 1 au. If no unit is provided via type
#         `~astropy.units.Quantity`, then au is assumed
#     sublon : u.Quantity
#         Observer longitude in target-fixed frame
#     sublat : u.Quantity
#         Observer latitude in target-fixed frame
#     unit : str, u.Unit, optional
#         Specify the unit of returned flux density
#     error : bool, optional
#         Return error of computed flux density
#     epsrel : float, optional
#         Relative precision of the nunerical integration.
#     **kwargs : Other keywords accepted by `scipy.integrate.dblquad`
#         Including `epsabs`, `epsrel`

#     Returns
#     -------
#     u.Quantity : Integrated flux density if `error = False`,
#         or flux density and numerical error if `error = True`.
#     """
#     delta_ = u.Quantity(delta, u.km)
#     unit = unit + " sr-1"
#     m = self.sub_observer_to_body_fixed(sublon, sublat)
#     f = dblquad(
#         self.fluxd_surface_point,
#         -np.pi / 2,
#         np.pi / 2,
#         lambda x: -np.pi / 2,
#         lambda x: np.pi / 2,
#         args=(m, unit, wave_freq),
#         epsrel=epsrel,
#         **kwargs
#     )
#     flx = (
#         u.Quantity(f, unit)
#         * ((self.R / delta_) ** 2).to("sr", u.dimensionless_angles())
#         * self.emissivity
#     )
#     if error:
#         return flx
#     else:
#         return flx[0]


#    def flux(phys, eph, lam):
#        """Model flux density for a given wavelength `lam`, or a list/array thereof
#
#        Parameters
#        ----------
#        phys : `sbpy.data.Phys` instance, mandatory
#            provide physical properties
#        eph : `sbpy.data.Ephem` instance, mandatory
#            provide object ephemerides
#        lam : `astropy.units` quantity or list-like, mandatory
#            wavelength or list thereof
#
#        Examples
#        --------
#        >>> from astropy.time import Time
#        >>> from astropy import units as u
#        >>> from sbpy.thermal import STM
#        >>> from sbpy.data import Ephem, Phys
#        >>> epoch = Time('2019-03-12 12:30:00', scale='utc')
#        >>> eph = Ephem.from_horizons('2015 HW', location='568', epochs=epoch) # doctest: +REMOTE_DATA
#        >>> phys = PhysProp('diam'=0.3*u.km, 'pv'=0.3) # doctest: +SKIP
#        >>> lam = np.arange(1, 20, 5)*u.micron # doctest: +SKIP
#        >>> flux = STM.flux(phys, eph, lam) # doctest: +SKIP
#
#        not yet implemented
#
#        """

#    def fit(self, eph):
#        """Fit thermal model to observations stored in `sbpy.data.Ephem` instance
#
#        Parameters
#        ----------
#        eph : `sbpy.data.Ephem` instance, mandatory
#            provide object ephemerides and flux measurements
#
#        Examples
#        --------
#        >>> from sbpy.thermal import STM
#        >>> stmfit = STM.fit(eph) # doctest: +SKIP
#
#        not yet implemented
#
#        """


# class TemperatureDistribution(abc.ABC):
#     """Surface temperature distribution."""


# class NonRotThermalModel(ThermalClass):
#     """Non-rotating object temperature distribution, i.e., STM, NEATM."""

#     @property
#     def T_sub_solar(self):
#         f_sun = const.L_sun / (4 * np.pi * self.r**2)
#         return (
#             ((1 - self.A) * f_sun / (self.beaming * self.emissivity * const.sigma_sb))
#             ** 0.25
#         ).decompose()

#     @u.quantity_input(lon=u.deg, lat=u.deg)
#     def T(self, lon: u.Quantity["angle"], lat: u.Quantity["angle"]):
#         """Surface temperature at specific (lat, lon)

#         lon : u.Quantity in units equivalent to deg
#             Longitude
#         lat : u.Quantity in units equivalent to deg
#             Latitude

#         Returns
#         -------
#         u.Quantity : Surface temperature.
#         """
#         coslon = np.cos(lon)
#         coslat = np.cos(lat)
#         prec = np.finfo(coslat.value).resolution
#         if (abs(coslon) < prec) or (abs(coslat) < prec) or (coslon < 0):
#             return 0 * u.K
#         else:
#             return self.T_sub_solar * (coslon * coslat) ** 0.25


# NonRotThermalModel.__doc__ += "\n".join(ThermalClass.splitlines()[1:])


# class FastRotThermalModel(ThermalClass):
#     """Fast-rotating object temperature distribution, i.e., FRM"""

#     @property
#     def Tss(self):
#         f_sun = const.L_sun / (4 * np.pi * self.r**2)
#         return (
#             ((1 - self.A) * f_sun / (np.pi * self.emissivity * const.sigma_sb)) ** 0.25
#         ).decompose()

#     @u.quantity_input(lon=u.deg, lat=u.deg)
#     def T(self, lon, lat):
#         """Surface temperature at specific (lat, lon)

#         lon : u.Quantity in units equivalent to deg
#             Longitude
#         lat : u.Quantity in units equivalent to deg
#             Latitude

#         Returns
#         -------
#         u.Quantity : Surface temperature.
#         """
#         coslat = np.cos(lat)
#         return self.Tss * coslat**0.25


def twovec(axdef, indexa, plndef, indexp):
    """Transformation matrix to a new coordinate defined by two input vectors.

    Parameters
    ----------
    axdef : array-like float containing 3 elements
        The vector (x, y, z) that defines one of the principal axes of the new
        coordinate frame.
    indexa : int 0, 1, or 2
        Specify which of the three coordinate axes is defined by `axdef`.  0 for
        x-axis, 1 for y-axis, and 2 for z-axis
    plndef : array-like float containing 3 elements
        The vector (x, y, z) that defines (with `axdef`) a principal plane of
        the new coordinate frame.
    indexp : int 0, 1, or 2
        Specify the second axis of the principal frame determined by `axdef` and
        `plndef`

    Returns
    -------
    numpy array of shape 3x3 The transformation matrix that convert a vector
    from the old coordinate to the coordinate frame defined by the input vectors
    via a dot product.

    Notes
    -----
    This routine is directly translated form SPICE lib routine twovec.f (cf.
    SPICE manual http://www.nis.lanl.gov/~esm/idl/spice-dlm/spice-t.html#TWOVEC)

    The indexing of array elements are different in FORTRAN (that SPICE is
    originally based) from Python.  Here 0-based index is used.

    Note that the original twovec.f in SPICE toolkit returns a matrix that
    converts a vector in the new frame to the original frame, opposite to what
    is implemented here.

    """

    axdef = np.asarray(axdef).flatten()
    plndef = np.asarray(plndef).flatten()

    if np.linalg.norm(np.cross(axdef, plndef)) == 0:
        raise RuntimeError(
            "The input vectors AXDEF and PLNDEF are linearly"
            " correlated and can't define a coordinate frame."
        )

    M = np.eye(3)
    i1 = indexa % 3
    i2 = (indexa + 1) % 3
    i3 = (indexa + 2) % 3

    M[i1, :] = axdef / np.linalg.norm(axdef)
    if indexp == i2:
        xv = np.cross(axdef, plndef)
        M[i3, :] = xv / np.linalg.norm(xv)
        xv = np.cross(xv, axdef)
        M[i2, :] = xv / np.linalg.norm(xv)
    else:
        xv = np.cross(plndef, axdef)
        M[i2, :] = xv / np.linalg.norm(xv)
        xv = np.cross(axdef, xv)
        M[i3, :] = xv / np.linalg.norm(xv)

    return M
