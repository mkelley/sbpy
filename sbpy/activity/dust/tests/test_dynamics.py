# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

import numpy as np
import astropy.units as u
from astropy.time import Time

from .... import time
from ..dynamics import (
    State,
    FreeExpansion,
    SolarGravity,
    SolarGravityAndRadiation,
    SolverFailed,
)


pytest.importorskip("scipy")


def test_spice_prop2b():
    """Test case from SPICE NAIF toolkit prop2b, v2.2.0

    State at t0:
    R   (km):          0.00000   70710678.11865   70710678.11865
    V (km/s):          0.00000         -0.04464          0.04464

    State at tau/2:
    R   (km):         -0.00000  -70710678.11865  -70710678.11865
    V (km/s):          0.00000          0.04464         -0.04464

    """

    class EarthGravity(SolarGravity):
        _GM = 3.9860043543609598e5

    r1 = 1e8
    s = np.sqrt(EarthGravity._GM / r1)
    half_period = np.pi * r1 / s

    r = [0, r1 / np.sqrt(2), r1 / np.sqrt(2)]
    v = [0, -s / np.sqrt(2), s / np.sqrt(2)]

    initial = State(r, v, Time("2023-01-01"))
    t_f = initial.t + half_period * u.s
    final = EarthGravity.solve(initial, t_f)

    assert np.allclose(initial.r.value, [0, 70710678.11865, 70710678.11865])
    assert np.allclose(initial.v.value, [0, -0.04464, 0.04464], atol=0.00001)
    assert np.allclose(final.r.value, [0, -70710678.11865, -70710678.11865])
    assert np.allclose(final.v.value, [0, 0.04464, -0.04464], atol=0.00001)

    t_f = initial.t + 5 * half_period * u.s
    final = EarthGravity.solve(initial, t_f, max_step=1e8 * u.s)

    assert np.allclose(initial.r.value, [0, 70710678.11865, 70710678.11865])
    assert np.allclose(initial.v.value, [0, -0.04464, 0.04464], atol=0.00001)
    assert np.allclose(final.r.value, [0, -70710678.11865, -70710678.11865])
    assert np.allclose(final.v.value, [0, 0.04464, -0.04464], atol=0.00001)


class TestFreeExpansion:
    def test(self):
        r = [0, 1e6, 0]
        v = [0, -1, 1]

        initial = State(r, v, Time("2023-01-01"))
        t_f = initial.t + 1e6 * u.s
        final = FreeExpansion.solve(initial, t_f)

        assert np.allclose(final.r.value, [0, 0, 1e6])
        assert np.allclose(final.v.value, [0, -1, 1])
