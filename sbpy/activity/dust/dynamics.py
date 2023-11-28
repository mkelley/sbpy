# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy activity.dust.dynamics
===========================

Dust dynamical models.

"""

__all__ = [
    "State",
    "FreeExpansion",
    "SolarGravity",
    "SolarGravityAndRadiationPressure",
]

from typing import Iterable, Union, Optional, TypeVar

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
    J2000.0 epoch.

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

        if (self.r.shape != self.v.shape) or (len(self) != len(self.t)):
            raise ValueError("Mismatch between lengths of vectors.")

        # use astropy to convert between reference frames
        self.r = SkyCoord(
            x=self.r[..., 0],
            y=self.r[..., 1],
            z=self.r[..., 2],
            v_x=self.v[..., 0],
            v_y=self.v[..., 1],
            v_z=self.v[..., 2],
            obstime=self.t,
            frame="heliocentriceclipticiau76" if frame is None else frame,
            representation_type="cartesian",
        )

    def __len__(self):
        """Number of state vectors in this object."""
        if self._r.ndim == 1:
            return 1
        else:
            return self._r.shape[0]

    def __getitem__(self, k: Union[int, tuple, slice]) -> StateType:
        """Get the state(s) at ``k``."""
        return State(self.r[k], self.v[k], self.t[k], frame=self.frame)

    @property
    def r(self) -> u.Quantity[u.km]:
        """Position vector."""
        return u.Quantity(self._r, u.km)

    @r.setter
    @u.quantity_input
    def r(self, r: u.Quantity[u.km]):
        if r.ndim > 3 or r.shape[r.ndim - 1] != 3:
            raise ValueError("Must have shape (3,) or (N, 3).")
        self._r = r.to_value(u.km)

    @property
    def x(self) -> u.Quantity[u.km]:
        """x component of the position vector."""
        return self.r[..., 0]

    @property
    def y(self) -> u.Quantity[u.km]:
        """y component of the position vector."""
        return self.r[..., 1]

    @property
    def z(self) -> u.Quantity[u.km]:
        """z component of the position vector."""
        return self.r[..., 2]

    @property
    def v(self) -> u.Quantity[u.km / u.s]:
        """Velocity vector."""
        return u.Quantity(self._v, u.km / u.s)

    @v.setter
    @u.quantity_input
    def v(self, v: u.Quantity[u.km / u.s]):
        if v.ndim > 3 or v.shape[v.ndim - 1] != 3:
            raise ValueError("Must have shape (3,) or (N, 3).")
        self._v = v.to_value(u.km / u.s)

    @property
    def v_x(self) -> u.Quantity[u.km / u.s]:
        """x component of the velocity vector."""
        return self.v[..., 0]

    @property
    def v_y(self) -> u.Quantity[u.km / u.s]:
        """y component of the velocity vector."""
        return self.v[..., 1]

    @property
    def v_z(self) -> u.Quantity[u.km / u.s]:
        """z component of the velocity vector."""
        return self.v[..., 2]

    @property
    def rv(self) -> np.ndarray:
        """Position in km, and velocity in km/s."""
        if self._r.ndim == 1:
            return np.r_[self._r, self._v]
        else:
            return np.hstack((self._r, self._v))

    @property
    def t(self) -> Time:
        """Time in the internal scale and format."""
        return Time(self._t, format="et", scale="tdb")

    @t.setter
    def t(self, t):
        self._t = t.tdb.to_value("et").reshape((-1,))

    @property
    def skycoord(self) -> SkyCoord:
        """State as a `~astropy.coordinates.SkyCoord` object."""
        return SkyCoord(
            x=self.x,
            y=self.y,
            z=self.z,
            v_x=self.v_x,
            v_y=self.v_y,
            v_z=self.v_z,
            obstime=self.t,
            frame=self.frame,
            representation_type="cartesian",
        )

    def observe(self, observer: StateType):
        """Observe and return a `~astropy.coordinates.SkyCoord` object."""

        target: State = State.from_skycoord(self.skycoord.transform_to(observer.frame))
        return SkyCoord(
            x=target.x - observer.x,
            y=target.y - observer.y,
            z=target.z - observer.z,
            v_x=target.v_x - observer.v_x,
            v_y=target.v_y - observer.v_y,
            v_z=target.v_z - observer.v_z,
            obstime=self.t,
            frame=observer.frame,
            representation_type="cartesian",
        )

    @classmethod
    def from_states(cls, states: Iterable[StateType]) -> StateType:
        """Initialize from a list of states.

        The coordinate frames must be identical.


        Parameters
        ----------
        states : array

        """

        frames: set = set([state.frame for state in states])
        if len(frames) != 1:
            raise ValueError("The coordinate frames must be identical.")

        r: np.ndarray = np.array([state.r for state in states])
        v: np.ndarray = np.array([state.v for state in states])
        t: Time = Time([state.t.tdb.et for state in states], scale="tdb", format="et")

        return State(r, v, t, frame=list(frames)[0])

    @classmethod
    def from_skycoord(cls, coords: SkyCoord) -> StateType:
        """Initialize from astropy `~astropy.coordinates.SkyCoord`.


        Parameters
        ----------
        coords: ~astropy.coordinates.SkyCoord
            The object state.  Must have position and velocity, ``obstime``, and
            be convertible to cartesian (3D) coordinates.

        """

        r: u.Quantity = u.Quantity(
            [coords.cartesian.x, coords.cartesian.y, coords.cartesian.z]
        ).T
        v: u.Quantity = u.Quantity(
            [
                coords.cartesian.differentials["s"].d_x,
                coords.cartesian.differentials["s"].d_y,
                coords.cartesian.differentials["s"].d_z,
            ]
        ).T
        t: Time = coords.obstime
        return cls(r, v, t, frame=coords.frame)

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


class SolarGravityAndRadiationPressure(DynamicalModel):
    """Solve equations of motion considering radiation force.


    Dust is parameterized with ``beta``, the ratio of the force from solar
    radiation pressure (:math:`F_r`) to that from solar gravity (:math:`F_g`):

    .. math::
        \\beta = \\frac{F_r}{F_g}

    For spherical dust grains, ``beta`` reduces to:

    .. math::
        \\beta = \\frac{0.57 Q_{pr}}{\\rho a}

    where :math:`Q_{pr}` is the radiation pressure efficiency averaged over the
    solar spectrum, :math:`\\rho` is the mass density of the grain (g/cm3), and
    :math:`a` is the grain radius (μm) (Burns et al. 1979).

    Only Newtonian gravity and radiation pressure are considered.
    Poynting-Roberston drag and general relativity are not included.

    """

    # Constants for quick reference
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
            Radiation force efficiency factor: :math:`F_{rad} / F_{gravity}`.


        Returns
        -------
        dx_dt : ndarray
            First three elements for :math:`dr/dt`, next three for :math:`dv/dt`.

        """

        r = rv[:3]
        v = rv[3:]

        r2 = (r**2).sum()
        r1 = np.sqrt(r2)
        r3 = r2 * r1
        GM_r3 = cls._GM / r3 * (1 - beta)

        dx_dt = np.empty(6)
        dx_dt[:3] = v
        dx_dt[3:] = -r * GM_r3

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
            First three elements for :math:`df/dr`, next three for
            :math:`df/dv`.

        """

        r = rv[:3]
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
        **kwargs,
    ) -> State:
        """Solve the equations of motion for a single particle.

        The solution is calculated with `scipy.integrate.solve_ivp`.  The
        default parameters are tuned for precision, but your requirements may
        need different values.


        Parameters
        ----------
        initial : State
            Initial state (position and velocity at time) of the particle.

        t_f : Time
            Time at which the solution is desired.

        beta : float
            Radiation pressure efficiency factor.

        **kwargs
            Keyword arguments for `scipy.integrate.solve_ivp`.  Units are
            seconds, km, and km/s, e.g., ``max_step`` is a float value in units
            of seconds.  For relative and absolute tolerance keywords, ``rtol``
            and ``atol``, 6-element arrays may be used, where the first three
            elements are for position, and the last three are for velocity.


        Returns
        -------
        final : State

        """

        final = State([0, 0, 0], [0, 0, 0], t_f, frame=initial.frame)
        # jac_sparsity: np.ndarray = np.zeros((6, 6))
        # jac_sparsity[0, 3:] = 1
        # jac_sparsity[3:, :3] = 1

        ivp_kwargs = dict(
            rtol=1e-8,
            atol=[1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-10],
            jac=cls.df_drv,
            # jac_sparsity=jac_sparsity,  # not used for all methods
            method="LSODA",
        )
        ivp_kwargs.update(kwargs)

        result = solve_ivp(
            cls.dx_dt,
            (initial.t.et[0], final.t.et[0]),
            initial.rv,
            args=(beta,),
            **ivp_kwargs,
        )

        if not result.success:
            raise SolverFailed(result.message)

        final.r = result.y[:3, -1] * u.km
        final.v = result.y[3:, -1] * u.km / u.s
        return final


class SolarGravity(SolarGravityAndRadiationPressure):
    """Solve equations of motion for a particle orbiting the Sun."""

    @classmethod
    def solve(cls, initial: State, t_f: Time) -> State:
        """Solve the equations of motion for a single particle.


        Parameters
        ----------
        initial : State
            Initial state (position and velocity at time) of the particle.

        t_f : Time
            Time at which the solution is desired.

        Returns
        -------
        final : State

        """

        return super().solve(initial, t_f, 0)


class FreeExpansion(SolarGravity):
    """Solve equations of motion for a particle in free space."""

    _GM = 0
