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

import abc
from typing import Iterable, Union, Optional, Tuple, TypeVar

import numpy as np

try:
    from scipy.integrate import solve_ivp
except ImportError:
    pass

from astropy.time import Time
import astropy.units as u
from astropy.coordinates import (
    SkyCoord,
    BaseCoordinateFrame,
    SphericalRepresentation,
    SphericalDifferential,
)
import astropy.constants as const

from ... import data as sbd
from ...data.ephem import Ephem
from ...exceptions import SbpyException
from ...utils.decorators import requires
from ... import time  # noqa: F401


class SolverFailed(SbpyException):
    """DynamicalModel solver failed."""


StateType = TypeVar("StateType", bound="State")
FrameType = TypeVar("FrameType", str, BaseCoordinateFrame)


class State:
    """Dynamical state of an asteroid, comet, dust grain, etc.

    Take care that the input coordinates are in the correct reference frame,
    e.g., if using ICRF, then the coordinates should be with respect to the
    solar system barycenter.


    Parameters
    ----------
    r : `~astropy.units.Quantity`
        Position (x, y, z), shape = (3,) or (N, 3).

    v : `~astropy.units.Quantity`
        Velocity (x, y, z), shape = (3,) or (N, 3).

    t : `~astropy.time.Time`
        Time, a scalar or shape = (N,).

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
        r: u.Quantity,
        v: u.Quantity,
        t: Time,
        frame: Optional[FrameType] = None,
    ) -> None:
        self.r = u.Quantity(r, "km")
        self.v = u.Quantity(v, "km/s")
        self.t = Time([t] * len(self)) if t.ndim == 0 else t
        self.frame = frame

        if (self.r.shape != self.v.shape) or (len(self) != len(self.t)):
            raise ValueError("Mismatch between lengths of vectors.")

    def __repr__(self) -> str:
        return (
            f"<{type(self).__name__} ({self.frame}):\n"
            + f" r\n    {self.r}\n"
            + f" v\n    {self.v}\n"
            + f" t\n    {self.t}>"
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

    def __add__(self, other: StateType) -> StateType:
        """Vector addition of two states.

        Time is taken from the left operand.

        """

        return State(
            self.r + other.r,
            self.v + other.v,
            self.t,
            frame=self.frame,
        )

    def __sub__(self, other: StateType) -> StateType:
        """Vector subtraction of two states.

        Time is taken from the left operand.

        """

        return self + -other

    def __neg__(self) -> StateType:
        """Invert the direction of the state vector position and velocity."""
        return State(
            -self.r,
            -self.v,
            self.t,
            frame=self.frame,
        )

    def __abs__(self) -> Tuple[u.Quantity, u.Quantity]:
        """Return the magnitude of the position and velocity."""
        r = np.sqrt(np.sum(self.r**2, axis=-1))
        v = np.sqrt(np.sum(self.v**2, axis=-1))
        return r, v

    @property
    def r(self) -> u.Quantity:
        """Position vector."""
        return u.Quantity(self._r, u.km)

    @r.setter
    @u.quantity_input
    def r(self, r: u.Quantity):
        if r.ndim > 3 or r.shape[r.ndim - 1] != 3:
            raise ValueError("Must have shape (3,) or (N, 3).")
        self._r = r.to_value(u.km)

    @property
    def x(self) -> u.Quantity:
        """x component of the position vector."""
        return self.r[..., 0]

    @property
    def y(self) -> u.Quantity:
        """y component of the position vector."""
        return self.r[..., 1]

    @property
    def z(self) -> u.Quantity:
        """z component of the position vector."""
        return self.r[..., 2]

    @property
    def v(self) -> u.Quantity:
        """Velocity vector."""
        return u.Quantity(self._v, u.km / u.s)

    @v.setter
    @u.quantity_input
    def v(self, v: u.Quantity):
        if v.ndim > 3 or v.shape[v.ndim - 1] != 3:
            raise ValueError("Must have shape (3,) or (N, 3).")
        self._v = v.to_value(u.km / u.s)

    @property
    def v_x(self) -> u.Quantity:
        """x component of the velocity vector."""
        return self.v[..., 0]

    @property
    def v_y(self) -> u.Quantity:
        """y component of the velocity vector."""
        return self.v[..., 1]

    @property
    def v_z(self) -> u.Quantity:
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
        """Time."""
        return Time(self._t, format="et", scale="tdb")

    @t.setter
    def t(self, t):
        self._t = t.tdb.to_value("et").reshape((-1,))

    def to_skycoord(self) -> SkyCoord:
        """State as a `~astropy.coordinates.SkyCoord` object."""

        kwargs: dict = {}
        if isinstance(self.frame, BaseCoordinateFrame):
            kwargs["frame"] = self.frame.copy()
            kwargs["frame"].representation_type = "cartesian"
        else:
            kwargs["frame"] = self.frame
            kwargs["representation_type"] = "cartesian"

        # when frame instances have obstime, SkyCoord will not accept it as a
        # separate parameter
        if not hasattr(kwargs["frame"], "obstime"):
            kwargs["obstime"] = self.t[0]

        return SkyCoord(
            x=self.x,
            y=self.y,
            z=self.z,
            v_x=self.v_x,
            v_y=self.v_y,
            v_z=self.v_z,
            **kwargs,
        )

    def transform_to(self, frame: FrameType) -> StateType:
        """Transform state into another reference frame.


        Parameters
        ----------
        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform into this reference frame.


        Returns
        -------
        state : State
            The transformed state.

        """

        return State.from_skycoord(self.to_skycoord().transform_to(frame))

    def observe(
        self,
        target: StateType,
        frame: Optional[FrameType] = None,
    ) -> SkyCoord:
        """Project a target's position on to the sky.


        Parameters
        ----------
        target : State
            The target to observe.

        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform the coordinates into this reference frame.


        Returns
        -------
        coords : SkyCoord

        """

        frame: FrameType = self.frame if frame is None else frame

        # transform into the requested reference frame, then difference
        target_in_frame: State = target.transform_to(frame)
        observer_in_frame: State = self.transform_to(frame)

        coords: SkyCoord = (target_in_frame - observer_in_frame).to_skycoord()
        coords.representation_type = "spherical"
        return coords

    @classmethod
    def from_states(cls, states: Iterable[StateType]) -> StateType:
        """Initialize from a list of states.

        The coordinate frames must be identical.


        Parameters
        ----------
        states : array

        """

        frames: set = set([str(state.frame).lower() for state in states])
        if len(frames) != 1:
            raise ValueError("The coordinate frames must be identical.")

        r: np.ndarray = np.array([state.r for state in states])
        v: np.ndarray = np.array([state.v for state in states])
        t: Time = Time(
            [state.t.tdb.et for state in states],
            scale="tdb",
            format="et",
        )

        return State(r, v, t, frame=list(frames)[0])

    @classmethod
    def from_skycoord(cls, coords: SkyCoord) -> StateType:
        """Initialize from astropy `~astropy.coordinates.SkyCoord`.


        Parameters
        ----------
        coords: ~astropy.coordinates.SkyCoord
            The object state.  Must have position and velocity, ``obstime``,
            and be convertible to cartesian (3D) coordinates.

        """

        _coords: SkyCoord = coords.copy()
        _coords.representation_type = "cartesian"

        r: u.Quantity = u.Quantity([_coords.x, _coords.y, _coords.z]).T
        v: u.Quantity = u.Quantity([_coords.v_x, _coords.v_y, _coords.v_z]).T
        t: Time = coords.obstime
        return cls(r, v, t, frame=coords.frame)

    @classmethod
    @sbd.dataclass_input
    def from_ephem(
        cls,
        eph: Ephem,
        frame: Optional[FrameType] = None,
    ) -> StateType:
        """Initialize from an `~sbpy.data.Ephem` object.


        Parameters
        ----------
        eph : ~sbpy.data.ephem.Ephem
            Ephemeris object, must have time, position, and velocity.  Position
            and velocity may be specified using ("x", "y", "z", "vx", "vy", and
            "vz"), or ("ra", "dec", "Delta", "RA*cos(Dec)_rate", "Dec_rate",
            and "deltadot").

        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform the coordinates into this reference frame.

        """

        rectangular: Tuple[str] = ("x", "y", "z", "vx", "vy", "vz", "date")
        spherical: Tuple[str] = (
            "ra",
            "dec",
            "Delta",
            "RA*cos(Dec)_rate",
            "Dec_rate",
            "deltadot",
            "date",
        )

        if all([x in eph for x in rectangular]):
            r: u.Quantity = (
                u.Quantity([eph["x"], eph["y"], eph["z"]]).reshape((3, len(eph))).T
            )
            v: u.Quantity = (
                u.Quantity([eph["vx"], eph["vy"], eph["vz"]]).reshape((3, len(eph))).T
            )
            return cls(r, v, eph["date"], frame=frame)
        elif all([x in eph for x in spherical]):
            c: SphericalRepresentation = SphericalRepresentation(
                eph["ra"], eph["dec"], eph["Delta"]
            )
            d: SphericalDifferential = SphericalDifferential(
                eph["RA*cos(Dec)_rate"],
                eph["Dec_rate"],
                eph["deltadot"],
            ).to_cartesian(base=c)
            c = c.to_cartesian()

            r: u.Quantity = u.Quantity([c.x, c.y, c.z]).reshape((3, len(c))).T
            v: u.Quantity = u.Quantity([d.x, d.y, d.z]).reshape((3, len(c))).T
            return cls(r, v, eph["date"], frame=frame)

        raise ValueError(
            "`Ephem` does not have the required time, position, and/or"
            " velocity fields."
        )


class DynamicalModel(abc.ABC):
    """Super-class for dynamical models.

    Parameters
    ----------
    **kwargs
        Arguments passed on to `~scipy.integrate.solve_ivp`.  Units are seconds,
        km, and km/s, e.g., ``max_step`` is a float value in units of seconds.
        For relative and absolute tolerance keywords, ``rtol`` and ``atol``,
        6-element arrays may be used, where the first three elements are for
        position, and the last three are for velocity.

    """

    @requires("scipy")
    def __init__(self, **kwargs):
        self.solver_kwargs: dict = dict(
            rtol=1e-8,
            atol=[1e-4, 1e-4, 1e-4, 1e-10, 1e-10, 1e-10],
            jac=self.df_drv,
            method="LSODA",
        )
        self.solver_kwargs.update(kwargs)

    @abc.abstractclassmethod
    def dx_dt(cls, t: float, rv: np.ndarray, *args) -> np.ndarray:
        """Derivative of position and velocity.


        Parameters
        ----------
        t : float
            Time, s.  Not used.

        rv : ndarray
            First three elements are the position vector at time ``t``, km. The
            next three elements are the velocity vector at time ``t``, km/s.

        *args :
            Additional parameters.


        Returns
        -------
        dx_dt : `numpy.ndarray`
            First three elements for :math:`dr/dt`, next three for :math:`dv/dt`.

        """

    @abc.abstractclassmethod
    def df_drv(cls, t: float, rv: np.ndarray, *args) -> np.ndarray:
        """Jacobian matrix, :math:`df/drv`.


        Parameters
        ----------
        t : float
            Time, s.  Not used.

        rv : ndarray
            First three elements are the position vector at time ``t``, km. The
            next three elements are the velocity vector at time ``t``, km/s.

        *args :
            Additional parameters.


        Returns
        -------
        df_drv : `numpy.ndarray`
            First three elements for :math:`df/dr`, next three for
            :math:`df/dv`.

        """

    def solve(
        self,
        initial: State,
        t_f: Time,
        *args,
    ) -> State:
        """Solve the equations of motion for a single particle.

        The solution is calculated with `scipy.integrate.solve_ivp`.


        Parameters
        ----------
        initial : State
            Initial state (position and velocity at time) of the particle.

        t_f : Time
            Time at which the solution is desired.

        *args :
            Additional arguments passed to `dx_dt` and `df_drv`.


        Returns
        -------
        final : State

        """

        final = State([0, 0, 0], [0, 0, 0], t_f, frame=initial.frame)

        result = solve_ivp(
            self.dx_dt,
            (initial.t.et[0], final.t.et[0]),
            initial.rv,
            args=args,
            **self.solver_kwargs,
        )

        if not result.success:
            raise SolverFailed(result.message)

        final.r = result.y[:3, -1] * u.km
        final.v = result.y[3:, -1] * u.km / u.s
        return final


class FreeExpansion(DynamicalModel):
    """Equation of motion solver for particle motion in free space.


    Parameters
    ----------
    **kwargs
        Arguments passed on to `~scipy.integrate.solve_ivp`.  Units are seconds,
        km, and km/s, e.g., ``max_step`` is a float value in units of seconds.
        For relative and absolute tolerance keywords, ``rtol`` and ``atol``,
        6-element arrays may be used, where the first three elements are for
        position, and the last three are for velocity.

    """

    @classmethod
    def dx_dt(cls, t: float, rv: np.ndarray, *args) -> np.ndarray:
        dx_dt = np.empty(6)
        dx_dt[:3] = rv[3:]
        dx_dt[3:] = 0

        return dx_dt

    @classmethod
    def df_drv(cls, t: float, rv: np.ndarray, *args) -> np.ndarray:
        # df_drv[i, j] = df_i/drv_j
        df_drv = np.zeros((6, 6))

        df_drv[0, 3] = 1
        df_drv[1, 4] = 1
        df_drv[2, 5] = 1

        return df_drv


class SolarGravity(DynamicalModel):
    """Equation of motion solver for a particle orbiting the Sun.


    Parameters
    ----------
    **kwargs
        Arguments passed on to `~scipy.integrate.solve_ivp`.  Units are seconds,
        km, and km/s, e.g., ``max_step`` is a float value in units of seconds.
        For relative and absolute tolerance keywords, ``rtol`` and ``atol``,
        6-element arrays may be used, where the first three elements are for
        position, and the last three are for velocity.

    """

    _GM: float = (const.G * const.M_sun).to_value("km3/s2")

    @property
    def GM(self):
        """Gravitational constant times mass."""
        return u.Quantity(self._GM, "km3/s2")

    @classmethod
    def dx_dt(cls, t: float, rv: np.ndarray, *args) -> np.ndarray:
        r = rv[:3]
        v = rv[3:]

        r2 = (r**2).sum()
        r1 = np.sqrt(r2)
        r3 = r2 * r1
        GM_r3 = cls._GM / r3

        dx_dt = np.empty(6)
        dx_dt[:3] = v
        dx_dt[3:] = -r * GM_r3

        return dx_dt

    @classmethod
    def df_drv(cls, t: float, rv: np.ndarray, *args) -> np.ndarray:
        r = rv[:3]
        r2 = (r**2).sum()
        r1 = np.sqrt(r2)
        GM_r5 = GM_r3 = cls._GM / (r2 * r2 * r1)

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


class SolarGravityAndRadiationPressure(DynamicalModel):
    """Equation of motion solver for a particle orbiting the Sun, including radiation force.


    Dust is parameterized with ``beta``, the ratio of the force from solar
    radiation pressure (:math:`F_r`) to that from solar gravity (:math:`F_g`):

    .. math::
        \\beta = \\frac{{F_r}}{{F_g}}

    For spherical dust grains, ``beta`` reduces to:

    .. math::
        \\beta = \\frac{{0.57 Q_{{pr}}}}{{\\rho a}}

    where :math:`Q_{{pr}}` is the radiation pressure efficiency averaged over
    the solar spectrum, :math:`\\rho` is the mass density of the grain (g/cm3),
    and :math:`a` is the grain radius (μm) (Burns et al. 1979).

    Only Newtonian gravity and radiation pressure are considered.
    Poynting-Roberston drag and general relativity are not included.


    Parameters
    ----------
    **kwargs
        Arguments passed on to `~scipy.integrate.solve_ivp`.  Units are seconds,
        km, and km/s, e.g., ``max_step`` is a float value in units of seconds.
        For relative and absolute tolerance keywords, ``rtol`` and ``atol``,
        6-element arrays may be used, where the first three elements are for
        position, and the last three are for velocity.

    """

    # For quick reference
    _GM: float = (const.G * const.M_sun).to_value("km3/s2")

    @property
    def GM(self):
        """Gravitational constant times mass."""
        return u.Quantity(self._GM, "km3/s2")

    @classmethod
    def dx_dt(cls, t: float, rv: np.ndarray, beta: float, *args) -> np.ndarray:
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
    def df_drv(cls, t: float, rv: np.ndarray, beta: float, *args) -> np.ndarray:
        r = rv[:3]
        r2 = (r**2).sum()
        r1 = np.sqrt(r2)
        r3 = r1 * r2
        GM_r5 = cls._GM * (1 - beta) / (r2 * r3)

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
