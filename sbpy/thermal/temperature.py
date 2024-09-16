import abc
import numpy as np
import astropy.units as u
import astropy.constants as const
from ..data.phys import Phys
from ..data import dataclass_input


class TemperatureDistribution(abs.ABC):
    """Abstract base class for temperature distributions.


    Parameters
    ----------

    phys : `~sbpy.data.Phys` or dictionary-like
        Physical properties of the object:
            - bolometric (Bond) albedo
            - infrared emissivity
            - infrared beaming parameter

    """

    @dataclass_input(phys=Phys)
    def __init__(self, phys: Phys):
        self.phys: Phys = phys

    @abc.abstractmethod
    def T(
        self,
        rh: u.Quantity["length"],
        lon: u.Quantity["angle"],
        lat: u.Quantity["angle"],
    ) -> u.Quantity["temperature"]:
        """Temperature at a point on the surface of an object.


        Parameters
        ----------
        rh : `astropy.units.Quantity`
            Heliocentric distance.

        lon : `astropy.units.Quantity`
            Longitude.

        lat : `astropy.units.Quantity`
            Latitude.


        Returns
        -------
        T : `astropy.units.Quantity`

        """

        pass


class InstantaneousEquilibriumTemperature(TemperatureDistribution):
    """Temperature of a surface in instantaneous equilibrium with sunlight."""

    @u.quantity_input()
    def T_sub_solar(self, rh: u.Quantity["length"]):
        """Sub-solar point temperature.


        Parameters
        ----------
        rh : `astropy.units.Quantity`
            Heliocentric distance.

        """

        A = self.phys["A"]
        eta = self.phys["eta"]
        epsilon = self.phys["emissivity"]
        F_sun = const.L_sun / (4 * np.pi * rh**2)

        return ((1 - A) * F_sun / (eta * epsilon * const.sigma_sb)).to("K4") ** (1 / 4)

    @u.quantity_input()
    def T(
        self,
        rh: u.Quantity["length"],
        lon: u.Quantity["angle"],
        lat: u.Quantity["angle"],
    ) -> u.Quantity["temperature"]:

        coslon = np.cos(lon)
        coslat = np.cos(lat)

        prec = np.finfo(coslat.value).resolution
        if (abs(coslon) < prec) or (abs(coslat) < prec) or (coslon < 0):
            T = 0 * u.K
        else:
            T = self.T_sub_solar(rh) * (coslon * coslat) ** 0.25

        return T


class FastRotatorTemperature(TemperatureDistribution):
    """Temperature distribution of a fast rotating sphere.


    Parameters
    ----------

    phys : `~sbpy.data.Phys` or dictionary-like
        Physical properties of the object:
            - bolometric (Bond) albedo
            - infrared beaming parameter

    """

    @u.quantity_input()
    def T_sub_solar(self, rh: u.Quantity["length"]):
        """Sub-solar point temperature.


        Parameters
        ----------
        rh : `astropy.units.Quantity`
            Heliocentric distance.

        """

        A = self.phys["A"]
        epsilon = self.phys["emissivity"]
        F_sun = const.L_sun / (4 * np.pi * rh**2)

        return ((1 - A) * F_sun / (np.pi * epsilon * const.sigma_sb)).to("K4") ** (
            1 / 4
        )

    @u.quantity_input()
    def T(
        self,
        rh: u.Quantity["length"],
        lon: u.Quantity["angle"],
        lat: u.Quantity["angle"],
    ) -> u.Quantity["temperature"]:

        coslat = np.cos(lat)
        T = self.T_sub_solar(rh) * coslat ** (1 / 4)

        return T
