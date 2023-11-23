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
from typing import List, Union, Optional

import numpy as np

import astropy.units as u
from astropy.time import Time

from .dynamics import State, SolarGravity, SolarGravityAndRadiation


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

    """

    def __init__(
        self,
        source: State,
        betas: Optional[Union[np.ndarray, u.Quantity]],
        ages: Optional[u.Quantity],
    ) -> None:
        if len(source) != 1:
            raise ValueError("Only one source state vector allowed.")

        self.source: State = source
        self.betas: u.Quantity[""] = u.Quantity(betas, "").reshape((-1,))
        self.ages: u.Quantity["s"] = u.Quantity(ages, "s").reshape((-1,))

        self.solve()

    def __repr__(self) -> str:
        return f"<Syndynes: {len(self.betas)} beta values, {len(self.ages)} time steps>"

    def initialize_states(self) -> None:
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

        self.initialize_states()

        self.r: np.ndarray = np.zeros((self.betas.size, self.ages.size, 3))
        t0: float = time.monotonic()
        for i in range(self.betas.size):
            for j in range(self.ages.size):
                state: State = SolarGravityAndRadiation.solve(
                    self.initial_states[j], self.source.t, self.betas[i]
                )
                self.r[i, j] = state.r
        t1: float = time.monotonic()

        logger.info(
            "Solved for %d syndynes, %d time steps each.",
            self.betas.size,
            self.ages.size,
        )
        logger.info(f"{(t1 - t0) / self.betas.size / self.ages.size} s/particle.")

    @property
    def syndynes(self):
        """Iterator for each syndyne."""

        for i in range(self.betas.size):
            yield self.r[i]

    @property
    def synchrones(self):
        """Iterator for each synchrone."""

        for i in range(self.ages.size):
            yield self.r[:, i]

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
