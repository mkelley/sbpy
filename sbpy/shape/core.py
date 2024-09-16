# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Shape Module

created on June 26, 2017
"""

import abc
import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord, SphericalRepresentation
from ..data import dataclass_input, Phys, Ephem
from ..units.typing import SpectralFluxDensityQuantity, SpectralQuantity, UnitLike
from ..utils.math import twovec


class Shape(abc.ABC):
    @dataclass_input
    def __init__(self, phys: Phys):
        self.phys: Phys = phys

    @abc.abstractmethod
    def total_flux_density(self, eph: Ephem) -> SpectralFluxDensityQuantity:
        pass


class Sphere(Shape):

    def __init__(self, phys: Phys, pole: SkyCoord):
        super().__init__(self, phys)
        self.pole: SkyCoord = pole

    @property
    def radius(self) -> u.Quantity["length"]:
        return self.phys["R"]

    @property
    def diameter(self) -> u.Quantity["length"]:
        return self.phys["D"]

    @dataclass_input()
    def total_flux_density(
        self,
        wave_freq: SpectralQuantity,
        eph: Ephem,
        unit: UnitLike = "W/(m2 um)",
    ):

        radiance_unit = u.Unit(unit) / u.sr

        target_observer = SkyCoord(
            ra=eph["ra"] + 180 * u.deg, dec=-eph["dec"], distance=eph["Delta"]
        )

        if u.isclose(target_observer.lat, 90 * u.deg):  # north pole
            M = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
        elif u.isclose(target_observer.lat, -90 * u.deg):  # south pole
            M = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        else:  # anywhere else
            M = twovec(target_observer.to_cartesian().get_xyz(), 0, [0, 0, 1], 2)

        def integrand(sub_lon, sub_lat):
            """Integration in sub-observer coordinates.

            lon, lat = 0, 0 is the sub-observer point.

            """

            sub_observer = SphericalRepresentation(lon=sub_lon, lat=sub_lat)
            body_coords = target_observer.transform(M)

            return (
                self.surface_spectral_radiance(unit=radiance_unit)
                * np.cos(sub_lon) ** 2
                * np.cos(sub_lat)
                * u.sr
            )


# __all__ = ["ModelClass", "Kaasalainen", "Lightcurve"]


# class ModelClass:

#     def __init__(self):
#         self.shape = None

#     def load_obj(self, filename):
#         """Load .OBJ shape model"""

#     def get_facet(self, facetid):
#         """Retrieve information for one specific surface facet"""

#     def iof(self, facetid, eph):
#         """Derive I/F for one specific surface facet"""

#     def integrated_flux(self, eph):
#         """Derive surface-integrated flux"""

#     def lightcurve(self, eph, time):
#         """Derive lightcurve"""


# class Kaasalainen(ModelClass):

#     def __init__(self):
#         self.properties = None
#         # setup model properties

#     def convexinv(self, eph):
#         """Obtain shape model through lightcurve inversion, optimizing all
#         parameters and uses spherical harmonics functions for shape
#         representation"""

#     def conjgradinv(self, eph):
#         """Obtain shape model through lightcurve inversion, optimizing only
#         shape and uses directly facet areas as parameters"""


# class Lightcurve:

#     def __init__(self, eph):
#         self.eph = eph
#         self.fouriercoeff = None
#         self.period = None
#         self.pole = (0, 90)  # ecliptic coordinates

#     def axis_ratio(self):
#         """Derive axis ratio from lightcurve amplitude"""

#     def derive_period(self, method="lomb-scargle"):
#         """Derive lightcurve period using different methods"""

#     def fit_fourier(self, order):
#         """Fit Fourier coefficients to lightcurve"""

#     def fit_pole(self):
#         """Fit pole orientation"""

#     def fit(self):
#         """Fit period, pole orientation, and Fourier coefficients at the same time"""

#     def simulate(self):
#         """Simulate a lightcurve from period, Fourier coefficients, pole orientation"""
