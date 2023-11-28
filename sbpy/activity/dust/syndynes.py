# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy activity.dust.syndynes
===========================

Generate cometary dust syndynes and synchrones.

"""

__all__ = [
    "Syndynes",
]

import time
import logging
from typing import Iterable, List, Tuple, Union, Optional

import numpy as np

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    BaseCoordinateFrame,
    SkyCoord,
    get_body_barycentric_posvel,
)

from .dynamics import State, SolarGravity, SolarGravityAndRadiationPressure


class Syndynes:
    """Syndyne / synchrone generator for cometary dust.


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


    Parameters
    ----------
    source : State
        State vector (i.e., position and velocity at time) of the object
        producing dust at the time of the observation.

    betas : ~numpy.ndarray, optional
        Array of beta-parameters to be simulated (dimensionless).

    ages : ~astropy.units.Quantity, optional
        Array of particle ages (time).

    observer : State, optional
        State vector of the observer in the same reference frame as ``source``.
        Default is the Earth obtained via ``astropy.coordinates.get_body``.

    """

    def __init__(
        self,
        source: State,
        betas: Union[Iterable, u.Quantity[u.dimensionless_unscaled]],
        ages: u.Quantity[u.s],
        observer: Optional[State] = None,
    ) -> None:
        if len(source) != 1:
            raise ValueError("Only one source state vector allowed.")

        self.source: State = source
        self.betas: u.Quantity[u.dimensionless_unscaled] = u.Quantity(
            betas, ""
        ).reshape((-1,))
        self.ages: u.Quantity[u.s] = u.Quantity(ages, "s").reshape((-1,))

        self.observer: State
        if observer is None:
            # use the Earth
            r_e: SkyCoord
            v_e: SkyCoord
            t: Time = source.t.reshape(())
            r_e, v_e = get_body_barycentric_posvel("earth", t)
            self.observer = State.from_skycoord(
                SkyCoord(
                    x=r_e.x,
                    y=r_e.y,
                    z=r_e.z,
                    v_x=v_e.x,
                    v_y=v_e.y,
                    v_z=v_e.z,
                    obstime=t,
                    frame="icrs",
                    representation_type="cartesian",
                )
            )
            # self.observer = State.from_skycoord(
            #     get_body("earth", source.t).transform_to(source.frame)
            # )
        elif observer.frame != source.frame:
            raise ValueError("source and observer frames are not equal.")
        else:
            self.observer = observer

        self.solve()

    def __repr__(self) -> str:
        return f"<Syndynes: {len(self.betas)} beta values, {len(self.ages)} time steps>"

    def _initialize_states(self) -> None:
        """Generate the initial particle states."""

        # integrate from observation time, t_f, back to t_i
        t_f = self.source.t.et

        states: List[State] = []
        for i, age in enumerate(self.ages):
            t_i: Time = self.source.t - age
            state = SolarGravity.solve(self.source, t_i)
            states.append(state)

        self.initial_states = State.from_states(states)

        logger: logging.Logger = logging.getLogger()
        logger.info("Initialized %d time steps.", self.ages.size)

    def solve(self) -> None:
        """Generate syndynes by solving the equations of motion."""

        logger: logging.Logger = logging.getLogger()

        self._initialize_states()

        particles: List[State] = []
        t0: float = time.monotonic()
        for i in range(self.betas.size):
            for j in range(self.ages.size):
                particles.append(
                    SolarGravityAndRadiationPressure.solve(
                        self.initial_states[j], self.source.t, self.betas[i]
                    )
                )
        t1: float = time.monotonic()
        self.particles = State.from_states(particles)

        logger.info(
            "Solved for %d syndynes, %d time steps each.",
            self.betas.size,
            self.ages.size,
        )
        logger.info(f"{(t1 - t0) / self.betas.size / self.ages.size} s/particle.")

    @property
    def syndynes(self):
        """Iterator for each syndyne."""

        n: int = self.ages.size
        i: int
        beta: float
        for i, beta in enumerate(self.betas):
            syn = self.particles[i * n : i + n]
            coords = syn.observe(self.observer)
            yield beta, syn, coords

    @property
    def synchrones(self):
        """Iterator for each synchrone."""

        n: int = self.betas.size
        i: int
        age: u.Quantity[u.s]
        for i, age in enumerate(self.ages):
            syn = self.particles[i::n]
            coords = syn.observe(self.observer)
            yield age, syn, coords

    def get_syndyne(self, beta: float) -> np.ndarray:
        """Get the positions of a single syndyne.


        Parameters
        ----------
        beta : float
            beta-value of the syndyne to get.


        Returns
        -------
        r : ndarray
            Array of position vectors of shape (N, 3), where N is the number of
            time steps (ages).

        """

        try:
            i = np.flatnonzero(self.betas.to_value("") == beta)[0]
        except IndexError:
            raise ValueError(f"beta-value not found: {beta}")

        return self.r[i]

    def get_synchrone(self, age: u.Quantity[u.s]) -> np.ndarray:
        """Get the positions of a single synchrone.


        Parameters
        ----------
        age : float
            Age of the synchrone to get.


        Returns
        -------
        r : ndarray
            Array of position vectors of shape (N, 3), where N is the number of
            beta-values.

        """

        try:
            i = np.flatnonzero(self.ages == age)[0]
        except IndexError:
            raise ValueError(f"Age not found: {age}")

        return self.r[:, i]

    def get_orbit(self, ages: u.Quantity[u.s]) -> Tuple[State, SkyCoord]:
        pass
