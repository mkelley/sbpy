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

import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord

from .dynamics import FrameType, State, DynamicalModel, SolarGravityAndRadiationPressure


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
        producing dust at the time of the observation.  Must be with respect to
        the central mass (e.g., the Sun).

    betas : ~numpy.ndarray, optional
        Array of beta-parameters to be simulated (dimensionless).

    ages : ~astropy.units.Quantity, optional
        Array of particle ages (time).

    observer : State, optional
        State vector of the observer in the same reference frame as ``source``.

    solver : `~sbpy.activity.dust.dynamics.DynamicalModel`, optional
        Solve the equations of motion with this object.

    """

    def __init__(
        self,
        source: State,
        betas: Union[Iterable, u.Quantity],
        ages: u.Quantity,
        observer: Optional[State] = None,
        solver: Optional[DynamicalModel] = SolarGravityAndRadiationPressure(),
    ) -> None:
        if len(source) != 1:
            raise ValueError("Only one source state vector allowed.")

        self.source: State = source
        self.betas: u.Quantity = u.Quantity(betas, "").reshape((-1,))
        self.ages: u.Quantity = u.Quantity(ages, "s").reshape((-1,))

        self.observer: State
        if observer is None:
            self.observer = None
        else:
            if observer.frame != source.frame:
                raise ValueError("source and observer frames are not equal.")
            self.observer = observer

        self.solver = solver

        self.initialize_states()

    def __repr__(self) -> str:
        return f"<Syndynes:\n betas\n    {self.betas}\n ages\n    {self.ages}>"

    def initialize_states(self) -> None:
        """Generate the initial particle states.

        This method is automatically run on initialization.

        """

        states: List[State] = []
        for age in self.ages:
            t_i: Time = self.source.t - age
            state = self.solver.solve(self.source, t_i, 0)
            states.append(state)

        self.initial_states = State.from_states(states)

        logger: logging.Logger = logging.getLogger()
        logger.info("Initialized %d time steps.", self.ages.size)

    def solve(self) -> None:
        """Generate test particle positions by solving the equations of motion."""

        logger: logging.Logger = logging.getLogger()

        particles: List[State] = []
        t0: float = time.monotonic()
        for i in range(self.betas.size):
            for j in range(self.ages.size):
                particles.append(
                    self.solver.solve(
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

    def get_syndyne(
        self,
        i: int,
        frame: Optional[FrameType] = None,
    ) -> Tuple[float, State, SkyCoord]:
        """Get a single syndyne.


        Parameters
        ----------
        i : int
            Index of the syndyne (same index as the `betas` array).

        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform observer coordinates into this reference frame.


        Returns
        -------
        beta : float
            The syndyne's beta value.

        syn : State
            The particle states.

        coords : SkyCoord, optional
            The observed coordinates.  Only returned when ``.observer`` is
            defined.

        """

        n: int = self.ages.size
        syn: State = self.particles[i * n : (i + 1) * n]

        if self.observer is None:
            return float(self.betas[i]), syn

        coords: SkyCoord = self.observer.observe(syn, frame)
        return float(self.betas[i]), syn, coords

    def get_synchrone(
        self,
        i: int,
        frame: Optional[FrameType] = None,
    ) -> Tuple[u.Quantity, State, SkyCoord]:
        """Get a single synchrone.


        Parameters
        ----------
        i : int
            Index of the synchrone (same index as the `ages` array).

        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform observer coordinates into this reference frame.


        Returns
        -------
        age : `astropy.units.Quantity`
            The syndyne's age.

        syn : State
            The particle states.

        coords : SkyCoord, optional
            The observed coordinates.  Only returned when ``.observer`` is
            defined.

        """

        n: int = self.ages.size
        syn: State = self.particles[i::n]

        if self.observer is None:
            return self.ages[i], syn

        coords: SkyCoord = self.observer.observe(syn, frame)
        return self.ages[i], syn, coords

    def syndynes(
        self, frame: Optional[FrameType] = None
    ) -> Tuple[float, State, SkyCoord]:
        """Iterator for each syndyne from `get_syndyne`.


        Parameters
        ----------
        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform observer coordinates into this reference frame.


        Returns
        -------
        iterator

        """
        for i in range(len(self.betas)):
            yield self.get_syndyne(i, frame)

    def synchrones(
        self, frame: Optional[FrameType] = None
    ) -> Tuple[u.Quantity, State, SkyCoord]:
        """Iterator for each synchrone from `get_synchrone`.


        Parameters
        ----------
        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform observer coordinates into this reference frame.


        Returns
        -------
        iterator

        """

        for i in range(len(self.ages)):
            yield self.get_synchrone(i, frame)

    def get_orbit(
        self, dt: u.Quantity, frame: Optional[FrameType] = None
    ) -> Union[State, Tuple[State, SkyCoord]]:
        """Calculate and observe the orbit of the dust source.


        Parameters
        ----------
        dt : `astropy.units.Quantity`
            The times at which to calculate the orbit, relative to the
            observation time.

        frame : string or `~astropy.coordinates.BaseCoordinateFrame`, optional
            Transform observer coordinates into this reference frame.


        Returns
        -------
        orbit : State
            The orbital states.

        coords : SkyCoord, optional
            The observed coordinates.  Only returned when ``.observer`` is
            defined.

        """

        states: List[State] = []
        for i in range(len(dt)):
            t: Time = self.source.t + dt[i]
            states.append(self.solver.solve(self.source, t, 0))
        states: State = State.from_states(states)

        if self.observer is None:
            return states

        coords: SkyCoord = self.observer.observe(states, frame)
        return states, coords
