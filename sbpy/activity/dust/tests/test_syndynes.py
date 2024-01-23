# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import astropy.units as u
from astropy.coordinates.errors import ConvertError
from astropy.time import Time
from ..syndynes import Syndynes, State
from ....dynamics.models import SolarGravity, SolarGravityAndRadiationPressure


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

        # no observer
        syn = Syndynes(comet, betas, ages)
        assert syn.observer is None

        # observer and comet are both in the "Arbitrary" frame
        observer = State([0, 1, 0] * u.au, [30, 0, 0] * u.km / u.s, comet.t)
        syn = Syndynes(comet, betas, ages, observer=observer)
        syn.solve()
        syn.get_syndyne(0)

        # cannot convert comet frame to observer frame:
        comet = State(comet.r, comet.v, comet.t, frame="heliocentriceclipticiau76")
        syn = Syndynes(comet, betas, ages, observer=observer)
        syn.solve()
        with pytest.raises(
            ConvertError,
            match="Cannot transform from.*HeliocentricEclipticIAU76.*to.*ArbitraryFrame",
        ):
            syn.get_syndyne(0)

        # fix observer frame
        observer = State(observer.r, observer.v, observer.t, comet.frame)
        syn = Syndynes(comet, betas, ages, observer=observer)
        syn.solve()
        syn.get_syndyne(0)

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
            assert u.allclose(initial.r, expected.r, atol=1 * u.cm, rtol=1e-11)
            assert u.allclose(initial.v, expected.v, atol=1 * u.um / u.s, rtol=1e-11)
            assert u.isclose(
                (initial.t - expected.t).to("s"), 0 * u.s, atol=1 * u.ns, rtol=1e-11
            )

    def test_solve(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        solver = SolarGravityAndRadiationPressure()
        for i, beta in enumerate(betas):
            syn = syn.get_syndyne(i)[1]
            for j, age in enumerate(ages):
                initial = solver.solve(comet, comet.t - age, 0)
                expected = solver.solve(initial, comet.t, beta)
                assert u.allclose(syn[j].r, expected.r, atol=1 * u.cm, rtol=1e-11)
                assert u.allclose(syn[j].v, expected.v, atol=1 * u.um / u.s, rtol=1e-11)
                assert np.allclose(syn.t.et, expected.t.et, rtol=1e-11)

    def test_syndynes(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        for beta, states, coords in syn.syndynes():
            assert beta == betas[0]
            assert np.allclose(states.t.et, comet.t.et, rtol=1e-11)
            assert np.allclose(coords.obstime.et, comet.t.et, rtol=1e-11)

            # State.observe is already tested, so being generous here
            assert u.allclose(coords.lon, [315, 315, 314] * u.deg, rtol=1e-3)
            assert u.allclose(coords.lat, 0 * u.deg)

        # remove the observer and verify outputs
        syn.observer = None
        for beta, states in syn.syndynes():
            assert beta == betas[0]
            assert np.allclose(states.t.et, comet.t.et, rtol=1e-11)
            assert np.allclose(coords.obstime.et, comet.t.et, rtol=1e-11)

    def test_synchrones(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        expected_lon = [315, 315, 314] * u.deg
        for i, (age, states, coords) in enumerate(syn.synchrones()):
            assert age == ages[i]
            assert np.allclose(states.t.et, comet.t.et, rtol=1e-11)
            assert np.allclose(coords.obstime.et, comet.t.et, rtol=1e-11)

            # State.observe is already tested, so a generous test here
            assert u.allclose(coords.lon, expected_lon[i], rtol=1e-3)
            assert u.allclose(coords.lat, 0 * u.deg)

        # remove the observer and verify outputs
        syn.observer = None
        for i, (age, states) in enumerate(syn.synchrones()):
            assert age == ages[i]
            assert np.allclose(states.t.et, comet.t.et, rtol=1e-11)
            assert np.allclose(coords.obstime.et, comet.t.et, rtol=1e-11)

    def test_get_orbit(self, example_syndynes):
        comet, betas, ages, syn, observer = example_syndynes

        dt = [-1, 0, 1] * u.d
        orbit, coords = syn.get_orbit(dt)
        assert u.allclose((orbit.t - comet.t).to("s"), dt)

        solver = SolarGravity()
        for i in range(len(dt)):
            expected = solver.solve(comet, comet.t + dt[i])
            assert u.allclose(orbit[i].r, expected.r, rtol=1e-11, atol=1 * u.cm)
            assert u.allclose(orbit[i].v, expected.v, rtol=1e-11, atol=1 * u.um / u.s)
