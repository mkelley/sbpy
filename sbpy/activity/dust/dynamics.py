# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy activity.dust.dynamics
===========================

Dust dynamical models.

"""

__all__ = [
    "FreeExpansion",
    "SolarGravity",
    "SolarGravityAndRadiation",
]

from typing import Union, Optional, TypeVar

import numpy as np
from scipy.integrate import solve_ivp

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import SkyCoord, BaseCoordinateFrame
import astropy.constants as const

from ... import data as sbd
from ...data.ephem import Ephem
from ...exceptions import SbpyException
from ... import time


class SolverFailed(SbpyException):
    pass


StateType = TypeVar("StateType", bound="State")


class State:
    """Dynamical state of an asteroid, comet, dust grain, etc.

    Take care that the input coordinates are in the correct reference frame,
    e.g., if using ICRF, then the coordinates should be with respect to the
    solar system barycenter.


    Parameters
    ----------
    r : ~astropy.units.Quantity
        Position (x, y, z), shape = (3,) or (N, 3).

    v : ~astropy.units.Quantity
        Velocity (x, y, z), shape = (3,) or (N, 3).

    t : ~astropy.time.Time
        Time, a scaler or shape = (N,).

    frame : `~astropy.coordinates.BaseCoordinateFrame` class or string, optional
        Coordinate frame for ``r`` and ``v``. Defaults to
        `~astropy.coordinates.HeliocentricEclipticIAU76` if given as ``None``.


    Examples
    --------

    >>> from astropy.time import Time
    >>> import astropy.units as u
    >>> from sbpy.activity.dust import State
    >>> r = [1e9, 1e9, 0] * u.km
    >>> v = [0, 0, 10] * u.km / u.s
    >>> t = Time("2022-07-24", scale="tdb")
    >>> state = State(r, v, t)


    Notes
    -----

    State is internally stored in units of km, km / s, and TDB seconds past
    J2000.0 epoch.  The internal reference frame is
    `~astropy.coordinates.HeliocentricEclipticIAU76` to match what JPL/Horizons
    and the NAIF SPICE toolkit use.

    """

    def __init__(
        self,
        r: u.Quantity[u.m],
        v: u.Quantity[u.m / u.s],
        t: Time,
        frame: Optional[Union[BaseCoordinateFrame, str]] = None,
    ) -> None:
        self.r = u.Quantity(r, "km")
        self.v = u.Quantity(v, "km/s")
        self.t = Time(t, format="et", scale="tdb")

        if not (self.r.shape[0] == self.v.shape[0] == self.t.shape[0]):
            raise ValueError("Mismatch between lengths of vectors.")

        self.frame = "heliocentriceclipticiau76" if frame is None else frame

    def __len__(self):
        """Number of state vectors in this object."""
        return self._r.shape[1]

    @property
    def r(self) -> u.Quantity[u.km]:
        """Position vector in the internal reference frame."""
        return u.Quantity(self._r, u.km)

    @r.setter
    @u.quantity_input
    def r(self, r: u.Quantity[u.m]):
        self._r = r.to_value(u.km).reshape((-1, 3))

    @property
    def v(self) -> u.Quantity[u.km / u.s]:
        """Velocity vector in the internal reference frame."""
        return u.Quantity(self._v, u.km / u.s)

    @v.setter
    @u.quantity_input
    def v(self, v: u.Quantity[u.m / u.s]):
        self._v = v.to_value(u.km / u.s).reshape((-1, 3))

    @property
    def rv(self) -> np.ndarray:
        """Position in km, and velocity in km/s."""
        return np.hstack((self._r, self._v))

    @property
    def t(self) -> Time:
        """Time in the internal scale and format."""
        return Time(self._t, format="et", scale="tdb")

    @t.setter
    def t(self, t):
        self._t = t.tdb.to_value("et").reshape((-1,))

    @property
    def coords(self) -> SkyCoord:
        """State as a `~astropy.coordinates.SkyCoords` object."""
        return SkyCoord(
            x=self.r[0],
            y=self.r[1],
            z=self.r[2],
            v_x=self.v[0],
            v_y=self.v[1],
            v_z=self.v[2],
            obstime=self.t,
            frame="heliocentriceclipticiau76",
            representation_type="cartesian",
        )

    @classmethod
    def from_skycoord(cls, coords: SkyCoord) -> StateType:
        """Initialize from astropy `~astropy.coordinates.SkyCoord`.


        Parameters
        ----------
        coords: ~astropy.coordinates.SkyCoord
            The object state.  Must have position and velocity, ``obstime``, and
            be convertible to cartesian (3D) coordinates.

        """

        out_coords = coords.transform_to("heliocentriceclipticiau76")
        out_coords.representation_type = "cartesian"
        r: u.Quantity = u.Quantity([out_coords.x, out_coords.y, out_coords.z])
        v: u.Quantity = u.Quantity([out_coords.v_x, out_coords.v_y, out_coords.v_z])
        t: Time = out_coords.obstime
        return cls.from_vectors(r, v, t)

    @classmethod
    @sbd.dataclass_input
    def from_ephem(cls, eph: Ephem) -> StateType:
        """Initialize from an `~sbpy.data.Ephem` object.


        Parameters
        ----------
        eph : ~sbpy.data.ephem.Ephem
            Ephemeris object, must have "time", "x", "y", "z", "vx", "vy", and
            "vz" fields, and ``Ephem.frame`` must be defined.

        """

        coords = SkyCoord(
            x=eph["x"],
            y=eph["y"],
            z=eph["z"],
            v_x=eph["vx"],
            v_y=eph["vy"],
            v_z=eph["vz"],
            obstime=eph["date"],
            representation_type="cartesian",
            frame=eph.frame,
        )
        return cls.from_skycoord(coords)


class DynamicalModel:
    """Super-class for dynamical models."""


class SolarGravityAndRadiation(DynamicalModel):
    """Solve equations of motion considering radiation force.


    Dust is parameterized with ``beta``, the ratio of the force from solar
    radiation pressure (:math:`F_r`) to that from solar gravity (:math:`F_g`):

    .. math:: \beta = \frac{F_r}{F_g}

    For spherical dust grains, ``beta`` reduces to:

    .. math:: \beta = \frac{0.57 Q_{pr}}{\rho a}

    where :math:`Q_{pr}` is the radiation pressure efficiency averaged over the
    solar spectrum, :math:`\rho` is the mass density of the grain (g/cm3), and
    :math:`a` is the grain radius (μm) (Burns et al. 1979).

    The radiation force up to Poynting-Roberson drag is considered.

    """

    # Constants for quick reference
    _C: float = const.c.to_value("km/s")
    _GM: float = (const.G * const.M_sun).to_value("km3/s2")

    @classmethod
    def dx_dt(cls, t: float, rv: np.ndarray, beta: float) -> np.ndarray:
        """Derivative of position and velocity.


        Parameters
        ----------
        t : float
            Time, s.  Not used.

        rv : ndarray
            First three elements are the position vector at time ``t``, km. The
            next three elements are the velocity vector at time ``t``, km/s.

        beta : float
            Radiation force efficiency factor: $F_{rad} / F_{gravity}$.


        Returns
        -------
        dx_dt : ndarray
            First three elements for $dr/dt$, next three for $dv/dt$.

        """

        r = rv[:3]
        v = rv[3:]

        r2 = (r**2).sum()
        r1 = np.sqrt(r2)
        r3 = r1 * r2
        GM_r3 = cls._GM / r3

        # radiation force up to Poynting-Robertson drag
        rhat = r / r2
        vr_c = v * rhat / cls._C
        betamu_r2 = beta * cls._GM / r2

        dx_dt = np.empty(6)
        dx_dt[:3] = v
        dx_dt[3:] = -r * GM_r3 + betamu_r2 * ((1 - vr_c) * rhat - v / cls._C)

        return dx_dt

    @classmethod
    def df_drv(cls, t: float, rv: np.ndarray, beta: float) -> np.ndarray:
        """Jacobian matrix, :math:`df/drv`.


        Parameters
        ----------
        t : float
            Time, s.  Not used.

        rv : ndarray
            First three elements are the position vector at time ``t``, km. The
            next three elements are the velocity vector at time ``t``, km/s.

        beta : float
            Radiation force efficiency factor: :math:`F_{rad} / F_{gravity}`.


        Returns
        -------
        df_drv : ndarray
            First three elements for $df/dr$, next three for :math:`df/dv`.

        """

        r = rv[:3]
        v = rv[3:]
        r2 = (r**2).sum()
        r1 = np.sqrt(r2)
        r3 = r1 * r2
        GM_r3 = cls._GM * (1 - beta) / r3
        GM_r5 = GM_r3 / r2

        # df_drv[i, j] = df_i/drv_j
        df_drv = np.zeros((6, 6))

        df_drv[0, 3] = 1
        df_drv[1, 4] = 1
        df_drv[2, 5] = 1

        df_drv[3, 0] = GM_r5 * (r2 - 3 * r[0] * r[0])
        df_drv[3, 1] = -GM_r5 * 3 * r[0] * r[1]
        df_drv[3, 2] = -GM_r5 * 3 * r[0] * r[2]

        df_drv[4, 0] = -GM_r5 * 3 * r[1] * r[0]
        df_drv[4, 1] = GM_r5 * (r2 - 3 * r[1] * r[1])
        df_drv[4, 2] = -GM_r5 * 3 * r[1] * r[2]

        df_drv[5, 0] = -GM_r5 * 3 * r[2] * r[0]
        df_drv[5, 1] = -GM_r5 * 3 * r[2] * r[1]
        df_drv[5, 2] = GM_r5 * (r2 - 3 * r[2] * r[2])

        return df_drv

    @classmethod
    def solve(
        cls,
        initial: State,
        t_f: Time,
        beta: float,
        max_step: Optional[u.Quantity[u.s]] = None,
    ) -> State:
        """Solve the equations of motion for a single particle.


        Parameters
        ----------
        initial : State
            Initial state (position and velocity at time) of the particle.

        t_f : Time
            Time at which the solution is desired.

        beta : float
            Radiation pressure efficiency factor.

        max_step : `astropy.units.Quantity`, optional
            Maximum integration step size.


        Returns
        -------
        final : State

        """

        final = State([0, 0, 0], [0, 0, 0], t_f, frame=initial.frame)

        if cls._GM > 0:
            # period for a circular orbit at this distance
            r1: float = np.sqrt(np.sum(initial.r.value**2))
            s: float = np.sqrt(cls._GM / r1)
            period0: float = 2 * np.pi * r1 / s
            max_step = period0 / 100 if max_step is None else max_step.to_value("s")
        else:
            max_step = np.inf

        result = solve_ivp(
            cls.dx_dt,
            (initial.t.et[0], final.t.et[0]),
            initial.rv[0],
            args=(beta,),
            jac=cls.df_drv,
            max_step=max_step,
            method="Radau",
        )

        if not result.success:
            raise SolverFailed(result.message)

        final.r = result.y[:3, -1] * u.km
        final.v = result.y[3:, -1] * u.km / u.s
        return final


class SolarGravity(SolarGravityAndRadiation):
    """Solve equations of motion for a particle orbiting the Sun."""

    @classmethod
    def solve(
        cls, initial: State, t_f: Time, max_step: Optional[u.Quantity[u.s]] = None
    ) -> State:
        """Solve the equations of motion for a single particle.


        Parameters
        ----------
        initial : State
            Initial state (position and velocity at time) of the particle.

        t_f : Time
            Time at which the solution is desired.

        max_step : `astropy.units.Quantity`, optional
            Maximum integration step size.

        Returns
        -------
        final : State

        """

        return super().solve(initial, t_f, 0, max_step=max_step)


class FreeExpansion(SolarGravity):
    """Solve equations of motion for a particle in free space."""

    _GM = 0
