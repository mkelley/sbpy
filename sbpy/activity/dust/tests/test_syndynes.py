# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord, HeliocentricEclipticIAU76
from astropy.time import Time
from ....data import Ephem
from ..syndynes import Syndynes, State
from ..dynamics import SolarGravity, SolarGravityAndRadiationPressure


pytest.importorskip("scipy")


@pytest.fixture
def example_syndynes():
    comet = State(
        [1, 0, 0] * u.au,
        [0, 30, 0] * u.km / u.s,
        Time("2023-12-07"),
        frame="heliocentriceclipticiau76",
    )
    betas = [0.1]
    ages = [1, 10, 100] * u.d
    observer = State(
        [0, 1, 0] * u.au, [30, 0, 0] * u.km / u.s, comet.t, frame=comet.frame
    )
    syn = Syndynes(comet, betas, ages, observer=observer)
    syn.solve()
    return comet, betas, ages, syn, observer


class TestSyndynes:
    def test_init(self, example_syndynes):
        comet = State([1, 0, 0] * u.au, [0, 30, 0] * u.km / u.s, Time("2023-12-07"))
        betas = [0.1]
        ages = [1, 10, 100] * u.d
        syn = Syndynes(comet, betas, ages)

        assert syn.observer is None

        observer = State([0, 1, 0] * u.au, [30, 0, 0] * u.km / u.s, comet.t)
        syn = Syndynes(comet, betas, ages, observer=observer)

        # observer frame not the same
        comet.frame = "heliocentriceclipticiau76"
        with pytest.raises(ValueError):
            Syndynes(comet, betas, ages, observer=observer)

        # fix observer frame
        observer.frame = comet.frame
        Syndynes(comet, betas, ages, observer=observer)

        # try with BaseCoordinateFrame instances
        comet.frame = HeliocentricEclipticIAU76()
        observer.frame = None
        with pytest.raises(ValueError):
            Syndynes(comet, betas, ages, observer=observer)

        observer.frame = comet.frame
        Syndynes(comet, betas, ages, observer=observer)

        # only one length 1 states allowed
        with pytest.raises(ValueError):
            Syndynes(State.from_states([comet, comet]), betas, ages)

    def test_repr(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes
        assert (
            repr(syn)
            == """<Syndynes:
 betas
    [0.1]
 ages
    [  86400.  864000. 8640000.] s>"""
        )

    def test_initialize_states(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        solver = SolarGravity()
        for i, age in enumerate(ages):
            initial = syn.initial_states[i]
            expected = solver.solve(comet, comet.t - age)
            assert u.allclose(initial.r, expected.r)
            assert u.allclose(initial.v, expected.v)
            assert u.isclose((initial.t - expected.t).to("s"), 0 * u.s)

    def test_solve(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        solver = SolarGravityAndRadiationPressure()
        for i, beta in enumerate(betas):
            syn = syn.get_syndyne(i)[1]
            for j, age in enumerate(ages):
                initial = solver.solve(comet, comet.t - age, 0)
                expected = solver.solve(initial, comet.t, beta)
                assert u.allclose(syn[j].r, expected.r)
                assert u.allclose(syn[j].v, expected.v)
                assert all((syn.t - expected.t) == 0 * u.s)

    def test_syndynes(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        for beta, states, coords in syn.syndynes():
            assert beta == betas[0]
            assert np.allclose((states.t - comet.t).jd, 0)
            assert np.allclose((coords.obstime - comet.t).jd, 0)

            # State.observe is already tested, so just to a generous test here
            assert np.allclose(coords.lon.deg, 315, atol=2)
            assert np.allclose(coords.lat.deg, 0)

        # remove the observer and verify outputs
        syn.observer = None
        for beta, states in syn.syndynes():
            assert beta == betas[0]
            assert np.allclose((states.t - comet.t).jd, 0)
            assert np.allclose((coords.obstime - comet.t).jd, 0)

    def test_synchrones(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        for i, (age, states, coords) in enumerate(syn.synchrones()):
            assert age == ages[i]
            assert np.allclose((states.t - comet.t).jd, 0)
            assert np.allclose((coords.obstime - comet.t).jd, 0)

            # State.observe is already tested, so a generous test here
            assert np.allclose(coords.lon.deg, 315, atol=2)
            assert np.allclose(coords.lat.deg, 0)

        # remove the observer and verify outputs
        syn.observer = None
        for i, (age, states) in enumerate(syn.synchrones()):
            assert age == ages[i]
            assert np.allclose((states.t - comet.t).jd, 0)
            assert np.allclose((coords.obstime - comet.t).jd, 0)

    def test_get_orbit(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        dt = [-1, 0, 1] * u.d
        orbit, coords = syn.get_orbit(dt)
        assert np.allclose((orbit.t - comet.t).jd, dt.value)

        solver = SolarGravity()
        for i in range(len(dt)):
            expected = solver.solve(comet, comet.t + dt[i])
            assert u.allclose(orbit[i].r, expected.r)
            assert u.allclose(orbit[i].v, expected.v)
