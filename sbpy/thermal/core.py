# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy Thermal Module

created on June 27, 2017
"""

__all__ = ["ThermalClass", "NonRotThermalModel", "FastRotThermalModel"]

import abc
import numpy as np
from numpy.linalg import norm
from scipy.integrate import dblquad
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import Angle
from astropy.modeling.models import BlackBody
from ..data import Phys, Obs, Ephem, dataclass_input, quantity_to_dataclass

__doctest_requires__ = {"ThermalClass.flux": "astroquery"}


class ThermalClass(abc.ABC):
    """Abstract base class for thermal models.

    This class implements the basic calculation for thermal models, such as
    integration of total flux based on a temperature distribution.


    Parameters
    ----------
    rh : `~sbpy.data.Ephem`, dictionary-like, or `~astropy.units.Quantity`
        Heliocentric distance to evaluate the model.

    phys : `~sbpy.data.Phys` or dictionary-like
        Physical properties of the object:
          - size as radius or diameter (e.g., `R` or `D`, required)
          - bolometric (Bond) albdo (e.g., `A`, default 0.1)
          - emissivity (`emissivity` or `epsilon`, default 0.9)
          - infrared beaming parameter (`eta`, default 1.0)
        Refer to the data class :ref:`field name documentation <field name
        list>` for details on parameter names and units.

    """

    @quantity_to_dataclass(eph=(Ephem, "rh"))
    @dataclass_input(phys=Phys)
    def __init__(self, eph: Ephem, phys: Phys):
        self.r = u.Quantity(eph["rh"], u.au)
        self.phys = phys

        # apply default values
        if "A" not in phys:
            self.phys["A"] = 0.1
        if "emissivity" not in phys:
            self.phys["emissivity"] = 0.9
        if "eta" not in phys:
            self.phys["eta"] = 1.0

    @property
    def bolometric_albedo(self):
        return self.phys["A"]

    @property
    def emissivity(self):
        return self.phys["emissivity"]

    @property
    def eta(self):
        return self.phys["eta"]

    @abc.abstractmethod
    def T(self, lon: Angle, lat: Angle):
        """Temperature at a point on the surface of an object.


        Parameters
        ----------
        lon : `astropy.coordinates.Angle` or equivalent
            Longitude.

        lat : `astropy.coordinates.Angle` or equivalent
            Latitude.

        """

        # Needs to be overridden in subclasses.  This function needs to be able
        # to return a valid quantity for the full range of lon and lat, i.e.,
        # include the night side of an object.

        pass

    @u.quantity_input(wave_freq=u.m, equivalencies=u.spectral())
    def _int_func(self, lon, lat, body_xform, unit, wave_freq):
        """Integral function for `fluxd`.

        Parameters
        ----------
        lon : float
            Sub-observer longitude in radians.

        lat : float
            Sub-observer latitude in radians.

        body_xform : numpy array of shape (3, 3)
            Transformation matrix to convert a vector in the model frame to the
            body-fixed frame.  The model frame defines the sub-observer point to
            be `lon = 0`, `lat = 0`.  This matrix can be calculated with private
            method `_transfer_to_bodyframe`.

        unit : str or astropy.units.Unit
            Unit of the integral function.

        wave_freq : u.Quantity
            Wavelength or frequency of calculation

        Returns
        -------
        fluxd : float
            Spectral flux density in units of `unit`.

        """

        body_lon, body_lat = xyz2sph(*body_xform.dot(sph2xyz(lon, lat)))

        T = self.T(body_lon * u.rad, body_lat * u.rad)
        if np.isclose(T, 0 * u.K):
            return 0.0
        else:
            # the integral term needs to include a correction for latitude
            # with cos(lat), and a Lambertian emission term cos(lat) + cos(lon)
            coslat = np.cos(lat)
            coslon = np.cos(lon)
            f = BlackBody(T)(wave_freq) * coslat * coslat * coslon
            return f.to_value(unit, u.spectral_density(wave_freq))

    @staticmethod
    @u.quantity_input(sublon=u.deg, sublat=u.deg)
    def _xform_to_bodyframe(sublon, sublat):
        """Calculate transformation matrix.

        The numerical integration to calculate total flux density is performed
        in a reference frame where the sub-observer point is at
        (lon, lat) = (0, 0).  This matrix supports the transformation from
        this frame to the body-fixed frame to facilitate the calculation of
        surface temperature.
        """
        los = sph2xyz(sublon.to_value("rad"), sublat.to_value("rad"))
        if np.isclose(norm(np.cross(los, [0, 0, 1])), 0):
            if sublat.value > 0:
                m = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            else:
                m = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        else:
            m = twovec(
                sph2xyz(sublon.to_value("rad"), sublat.to_value("rad")), 0, [0, 0, 1], 2
            ).T
        return m

    @u.quantity_input(
        wave_freq=u.m, delta=u.au, lon=u.deg, lat=u.deg, equivalencies=u.spectral()
    )
    def fluxd(
        self,
        wave_freq,
        delta,
        sublon,
        sublat,
        unit="W m-2 um-1",
        error=False,
        epsrel=1e-3,
        **kwargs
    ):
        """Model thermal flux density of an object.

        Parameters
        ----------
        wave_freq : u.Quantity
            Wavelength or frequency of observations
        delta : `~sbpy.data.Ephem`, dict_like, number, or
            `~astropy.units.Quantity`
            If `~sbpy.data.Ephem` or dict_like, ephemerides of the object that
            can include the observer distance via keywords `delta`.  If float
            or `~astropy.units.Quantity`, then the observer distance of an
            object.  If observer distance is not found, then it will be
            assumed to be 1 au. If no unit is provided via type
            `~astropy.units.Quantity`, then au is assumed
        sublon : u.Quantity
            Observer longitude in target-fixed frame
        sublat : u.Quantity
            Observer latitude in target-fixed frame
        unit : str, u.Unit, optional
            Specify the unit of returned flux density
        error : bool, optional
            Return error of computed flux density
        epsrel : float, optional
            Relative precision of the nunerical integration.
        **kwargs : Other keywords accepted by `scipy.integrate.dblquad`
            Including `epsabs`, `epsrel`

        Returns
        -------
        u.Quantity : Integrated flux density if `error = False`,
            or flux density and numerical error if `error = True`.
        """
        delta_ = u.Quantity(delta, u.km)
        unit = unit + " sr-1"
        m = self._xform_to_bodyframe(sublon, sublat)
        f = dblquad(
            self._int_func,
            -np.pi / 2,
            np.pi / 2,
            lambda x: -np.pi / 2,
            lambda x: np.pi / 2,
            args=(m, unit, wave_freq),
            epsrel=epsrel,
            **kwargs
        )
        flx = (
            u.Quantity(f, unit)
            * ((self.R / delta_) ** 2).to("sr", u.dimensionless_angles())
            * self.emissivity
        )
        if error:
            return flx
        else:
            return flx[0]


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


class NonRotThermalModel(ThermalClass):
    """Non-rotating object temperature distribution, i.e., STM, NEATM"""

    @property
    def Tss(self):
        f_sun = const.L_sun / (4 * np.pi * self.r**2)
        return (
            ((1 - self.A) * f_sun / (self.beaming * self.emissivity * const.sigma_sb))
            ** 0.25
        ).decompose()

    @u.quantity_input(lon=u.deg, lat=u.deg)
    def T(self, lon, lat):
        """Surface temperature at specific (lat, lon)

        lon : u.Quantity in units equivalent to deg
            Longitude
        lat : u.Quantity in units equivalent to deg
            Latitude

        Returns
        -------
        u.Quantity : Surface temperature.
        """
        coslon = np.cos(lon)
        coslat = np.cos(lat)
        prec = np.finfo(coslat.value).resolution
        if (abs(coslon) < prec) or (abs(coslat) < prec) or (coslon < 0):
            return 0 * u.K
        else:
            return self.Tss * (coslon * coslat) ** 0.25


class FastRotThermalModel(ThermalClass):
    """Fast-rotating object temperature distribution, i.e., FRM"""

    @property
    def Tss(self):
        f_sun = const.L_sun / (4 * np.pi * self.r**2)
        return (
            ((1 - self.A) * f_sun / (np.pi * self.emissivity * const.sigma_sb)) ** 0.25
        ).decompose()

    @u.quantity_input(lon=u.deg, lat=u.deg)
    def T(self, lon, lat):
        """Surface temperature at specific (lat, lon)

        lon : u.Quantity in units equivalent to deg
            Longitude
        lat : u.Quantity in units equivalent to deg
            Latitude

        Returns
        -------
        u.Quantity : Surface temperature.
        """
        coslat = np.cos(lat)
        return self.Tss * coslat**0.25


def twovec(axdef, indexa, plndef, indexp):
    """Transformation matrix to a new coordinate defined by two input vectors.

    Parameters
    ----------
    axdef : array-like float containing 3 elements
        The vector (x, y, z) that defines one of the principal axes of the new
        coordinate frame.
    indexa : int 0, 1, or 2
        Specify which of the three coordinate axes is defined by `axdef`.  0
        for x-axis, 1 for y-axis, and 2 for z-axis
    plndef : array-like float containing 3 elements
        The vector (x, y, z) that defines (with `axdef`) a principal plane of
        the new coordinate frame.
    indexp : int 0, 1, or 2
        Specify the second axis of the principal frame determined by `axdef`
        and `plndef`

    Returns
    -------
    numpy array of shape 3x3
    The transformation matrix that convert a vector from the old coordinate to
    the coordinate frame defined by the input vectors via a dot product.

    Notes
    -----
    This routine is directly translated form SPICE lib routine twovec.f
    (cf. SPICE manual
    http://www.nis.lanl.gov/~esm/idl/spice-dlm/spice-t.html#TWOVEC)

    The indexing of array elements are different in FORTRAN (that SPICE
    is originally based) from Python.  Here 0-based index is used.

    Note that the original twovec.f in SPICE toolkit returns matrix that
    converts a vector in the new frame to the original frame, opposite to
    what is implemented here.
    """

    axdef = np.asarray(axdef).flatten()
    plndef = np.asarray(plndef).flatten()

    if norm(np.cross(axdef, plndef)) == 0:
        raise RuntimeError(
            "The input vectors AXDEF and PLNDEF are linearly"
            " correlated and can't define a coordinate frame."
        )

    M = np.eye(3)
    i1 = indexa % 3
    i2 = (indexa + 1) % 3
    i3 = (indexa + 2) % 3

    M[i1, :] = axdef / norm(axdef)
    if indexp == i2:
        xv = np.cross(axdef, plndef)
        M[i3, :] = xv / norm(xv)
        xv = np.cross(xv, axdef)
        M[i2, :] = xv / norm(xv)
    else:
        xv = np.cross(plndef, axdef)
        M[i2, :] = xv / norm(xv)
        xv = np.cross(axdef, xv)
        M[i3, :] = xv / norm(xv)

    return M


def xyz2sph(x, y, z):
    """Convert (x, y, z) to (lon, lat)."""
    x_ = np.asanyarray(x)
    y_ = np.asanyarray(y)
    z_ = np.asanyarray(z)
    lon = np.arctan2(y_, x_)
    complete_angle = (
        u.Quantity(2 * np.pi, u.rad) if isinstance(lon, u.Quantity) else 2 * np.pi
    )
    lon = (lon + complete_angle) % complete_angle
    lat = np.arctan2(z_, np.sqrt(x_ * x_ + y_ * y_))
    return np.stack([lon, lat])


def sph2xyz(lon, lat, r=1.0):
    """Convert (lon, lat) to (x, y, z), with a default length of unity."""
    if r is None:
        r = 1.0 * u.dimensionless_unscaled if isinstance(lon, u.Quantity) else 1.0
    lon_ = np.asanyarray(lon)
    lat_ = np.asanyarray(lat)
    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)
    return np.stack([x, y, z])
