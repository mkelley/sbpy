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

from typing import Union, Optional

import numpy as np
import erfa
from astropy.time import Time, TimeFromEpoch
import astropy.units as u

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


class State:
    """Dynamical state of an asteroid, comet, dust grain, etc.


    Parameters
    ----------
    r : ~astropy.units.Quantity
        Position.

    v : ~astropy.units.Quantity
        Velocity.

    t : ~astropy.time.Time
        Time.


    Examples
    --------

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from sbpy.activity.dust import State
    >>> r = [1e9, 1e9, 0] * u.km
    >>> v = [0, 0, 10] * u.km / u.s
    >>> t = Time('2022-07-24', scale='tdb')
    >>> state = State(r, v, t)


    Notes
    -----
    State is internally stored in units of km, km / s, and TDB seconds past
    J2000.0 epoch.

    """

    @u.quantity_input
    def __init__(self, r: u.Quantity[u.m], v: u.Quantity[u.m / u.s],
                 t: Time) -> None:
        self.r = r
        self.v = v
        self.t = t

    @property
    def r(self):
        return u.Quantity(self._r, u.km)

    @r.setter
    @u.quantity_input
    def r(self, r: u.Quantity[u.m]):
        self._r = r.to_value(u.km)

    @property
    def v(self):
        return u.Quantity(self._v, u.km / u.s)

    @v.setter
    @u.quantity_input
    def v(self, v: u.Quantity[u.m / u.s]):
        self._v = v.to_value(u.km / u.s)

    @property
    def t(self):
        return Time(self._t, format='et', scale='tdb')

    @t.setter
    def t(self, t):
        self._t = t.tdb.to_value('et')

    @classmethod
    @sbd.dataclass_input(eph=Ephem)
    def from_ephem(cls, eph: Ephem):
        """Initialize from an ephemeris object.


        Parameters
        ----------
        eph : ~sbpy.data.ephem.Ephem
            Ephemeris object, must have 'time', 'x', 'y', 'z', 'vx', 'vy', and
            'vz' fields.

        """

        r = u.Quantity([eph['x'], eph['y'], eph['z']])
        v = u.Quantity([eph['vx'], eph['vy'], eph['vz']])
        t = eph['date']


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
