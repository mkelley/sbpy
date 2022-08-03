# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy activity.dust.syndynes
===========================

Generate cometary dust syndynes and synchrones.

"""

__all__ = [
    'State',
    'Syndynes',
]

from typing import Union, Optional, TypeVar

import numpy as np
import erfa
from astropy.time import Time, TimeFromEpoch
import astropy.units as u
from astropy.coordinates import SkyCoord, BaseCoordinateFrame

import sbpy.data as sbd
from sbpy.data.ephem import Ephem


class SpiceEphemerisTime(TimeFromEpoch):
    """Number of seconds since J2000.0 epoch.

    This is equivalent to the NAIF SPICE Ephemeris Time when the Barycentric
    Dynamical Time (TDB) scale is used.

    """
    name = 'et'
    unit = 1.0 / erfa.DAYSEC  # in days (1 day == 86400 seconds)
    epoch_val = '2000-01-01 12:00:00'
    epoch_val2 = None
    epoch_scale = 'tdb'
    epoch_format = 'iso'


StateType = TypeVar('StateType', bound='State')


class State:
    """Dynamical state of an asteroid, comet, dust grain, etc.

    Take care that the input coordinates are in the correct reference frame,
    e.g., if using ICRF, then the coordinates should be with respect to the
    solar system barycenter.


    Parameters
    ----------
    coords : ~astropy.coordinates.SkyCoord
        Requires position, velocity, and time.


    Examples
    --------

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from sbpy.activity.dust import State
    >>> r = [1e9, 1e9, 0] * u.km
    >>> v = [0, 0, 10] * u.km / u.s
    >>> t = Time('2022-07-24', scale='tdb')
    >>> state = State.from_vectors(r, v, t)


    Notes
    -----

    State is internally stored in units of km, km / s, and TDB seconds past
    J2000.0 epoch.  The internal reference frame is
    `~astropy.coordinates.HeliocentricEclipticIAU76` to match what JPL/Horizons
    and the NAIF SPICE toolkit use.

    """

    def __init__(self, coords: SkyCoord):
        """Initialize from astropy `~astropy.coordinates.SkyCoord`.


        Parameters
        ----------
        coords: ~astropy.coordinates.SkyCoord
            The object state.  Must have position and velocity (e.g., ``x``,
            ``y``, ``z``, ``v_x``, ``v_y``, ``v_z``), ``obstime``, and be
            convertible to cartesian (3D) coordinates.

        """

        out_coords = coords.transform_to('heliocentriceclipticiau76')
        out_coords.representation_type = 'cartesian'
        self.r = u.Quantity([out_coords.x, out_coords.y, out_coords.z])
        self.v = u.Quantity([out_coords.v_x, out_coords.v_y, out_coords.v_z])
        self.t = out_coords.obstime

    @property
    def r(self):
        """Position vector in the internal reference frame."""
        return u.Quantity(self._r, u.km)

    @r.setter
    @u.quantity_input
    def r(self, r: u.Quantity[u.m]):
        self._r = r.to_value(u.km)

    @property
    def v(self):
        """Velocity vector in the internal reference frame."""
        return u.Quantity(self._v, u.km / u.s)

    @v.setter
    @u.quantity_input
    def v(self, v: u.Quantity[u.m / u.s]):
        self._v = v.to_value(u.km / u.s)

    @property
    def t(self):
        """Time in the internal scale and format."""
        return Time(self._t, format='et', scale='tdb')

    @t.setter
    def t(self, t):
        self._t = t.tdb.to_value('et')

    @property
    def coords(self):
        """State as a `~astropy.coordinates.SkyCoords` object."""
        return SkyCoord(x=self.r[0], y=self.r[1], z=self.r[2],
                        v_x=self.v[0], v_y=self.v[1], v_z=self.v[2],
                        obstime=self.t, frame='heliocentriceclipticiau76',
                        representation_type='cartesian')

    @classmethod
    @sbd.dataclass_input
    def from_ephem(cls, eph: Ephem) -> StateType:
        """Initialize from an `~sbpy.data.Ephem` object.


        Parameters
        ----------
        eph : ~sbpy.data.ephem.Ephem
            Ephemeris object, must have 'time', 'x', 'y', 'z', 'vx', 'vy', and
            'vz' fields, and ``Ephem.frame`` must be defined.

        """

        coords = SkyCoord(x=eph['x'], y=eph['y'], z=eph['z'],
                          v_x=eph['vx'], v_y=eph['vy'], v_z=eph['vz'],
                          obstime=t, representation_type='cartesian',
                          frame=eph.frame)
        return cls.from_skycoord(coords)

    @classmethod
    def from_vectors(cls,
                     r: u.Quantity[u.m],
                     v: u.Quantity[u.m / u.s],
                     t: Time,
                     frame: Optional[Union[BaseCoordinateFrame, str]] = None
                     ) -> None:
        """Initialize state from vectors.


        Parameters
        ----------
        r : ~astropy.units.Quantity
            Position.

        v : ~astropy.units.Quantity
            Velocity.

        t : ~astropy.time.Time
            Time.

        frame : `~astropy.coordinates.BaseCoordinateFrame` class or string,
        optional
            Coordinate frame for ``r`` and ``v``. Defaults to
            `~astropy.coordinates.HeliocentricEclipticIAU76` if given as
            ``None``.

        """

        frame = 'heliocentriceclipticiau76' if frame is None else frame
        coords = SkyCoord(x=r[0], y=r[1], z=r[2], v_x=v[0], v_y=v[1],
                          v_z=v[2], obstime=t, frame=frame,
                          representation_type='cartesian')
        return cls(coords)


class Syndynes:
    """Syndyne / synchrone generator for cometary dust.


    Dust is parameterized with ``beta``, the ratio of the force from solar
    radiation pressure (:math:`F_r`) to that from solar gravity (:math:`F_g`):

    .. math:: \beta = \frac{F_r}{F_g}

    For spherical dust grains, ``beta`` reduces to:

    .. math:: \beta = \frac{0.57 Q_{pr}}{\rho a}

    where :math:`Q_{pr}` is the radiation pressure efficiency averaged over the
    solar spectrum, :math:`\rho` is the mass density of the grain (g/cm3), and
    :math:`a` is the grain radius (μm) (Burns et al. 1979).


    Parameters
    ----------
    state_obs : State
        State vector (i.e., position and velocity at time) of the object
        producing dust at the time of the observation.

    betas : ~numpy.ndarray, optional
        Array of beta-parameters to be simulated (dimensionless).

    ages : ~astropy.units.Quantity, optional
        Array of particle ages (time).

    """

    def __init__(self, state_obs: State,
                 betas: Optional[Union[np.ndarray, u.Quantity]],
                 ages: Optional[u.Quantity]) -> None:
        self.state_obs = state_obs
        self.betas = betas
        self.ages = ages
