# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""
sbpy activity.dust.syndynes
===========================

Generate cometary dust syndynes and synchrones.

"""

__all__ = [
    "Syndynes",
]

from typing import List, Union, Optional
import logging

import numpy as np

import astropy.units as u
from astropy.time import Time

from .dynamics import State, SolarGravityAndRadiation


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

    def initialize_states(self):
        """Generate the initial particle states."""

        # integrate from observation time, t_f, back to t_i
        t_f = self.source.t.et

        self.initial_states: List[State] = []
        for i, age in enumerate(self.ages):
            t_i: Time = self.source.t - age
            state = SolarGravityAndRadiation.solve(self.source, t_i, 0)
            self.initial_states.append(state)

        logging.info("Initialized %d time steps.", self.ages.size)

    def solve(self):
        """Generate syndynes by solving the equations of motion."""

        states: np.ndarray[State] = np.zeros((self.ages.size, self.betas.size), State)
        for i in range(self.ages.size):
            for j in range(self.betas.size):
                states[i, j] = SolarGravityAndRadiation.solve(
                    self.initial_states[i], self.source.t, self.betas[j]
                )

        self.syndynes: np.ndarray[State] = states
        logging.info(
            "Solved for %d syndynes, %d time steps each.",
            self.betas.size,
            self.ages.size,
        )
