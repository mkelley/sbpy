# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import SkyCoord
from sbpy.data import Ephem

from .... import time
from ..dynamics import (
    State,
    FreeExpansion,
    SolarGravity,
    SolarGravityAndRadiationPressure,
    SolverFailed,
)


pytest.importorskip("scipy")


class TestState:
    def test_init(self):
        t = Time("2022-08-02")
        state = State(
            [1, 1, 0] * u.au,
            [30, 0, 0] * u.km / u.s,
            t,
            frame="heliocentriceclipticiau76",
        )

        # test properties
        assert u.allclose(state.r, [1, 1, 0] * u.au)
        assert u.allclose(state.v, [30000, 0, 0] * u.m / u.s)
        assert np.isclose((state.t - t).jd, 0)

        # test internal state values
        assert np.allclose(state._r, ([1, 1, 0] * u.au).to_value("km"))
        assert np.allclose(state._v, ([30, 0, 0] * u.km / u.s).to_value("km / s"))
        assert np.isclose(state._t, t.tdb.to_value("et"))

    def test_init_radec(self):
        """Test RA, Dec, distance to verify that internal conversions work.

        Coordinates are 2P/Encke at 2022-08-02 UTC

        Once for RA, Dec, then again for vectors, but adjusted for light travel
        time.

        RA, Dec is defined in the ICRF, so get barycentric coordinates from
        Horizons.

        >>> from astropy.time import Time
        >>> from sbpy.data import Ephem
        >>>
        >>> t = Time('2022-08-02')
        >>> eph = Ephem.from_horizons('2P', id_type='designation',
        ...                           epochs=t,
        ...                           closest_apparition=True,
        ...                           location='@0')
        ... # doctest: +REMOTE_DATA
        >>> [eph[k] for k in ('ra', 'dec', 'Delta', 'RA*cos(Dec)_rate',
        ...  'Dec_rate', 'delta_rate')]  # doctest: +REMOTE_DATA
        [<MaskedQuantity [348.37706] deg>,
         <MaskedQuantity [-1.86304] deg>,
         <MaskedQuantity [3.90413464] AU>,
         <MaskedQuantity [6.308283] arcsec / h>,
         <MaskedQuantity [4.270114] arcsec / h>,
         <MaskedQuantity [-4.19816] km / s>]

        >>> from astroquery.jplhorizons import Horizons
        >>> import astropy.units as u
        >>> from astropy.constants import c
        >>>
        >>> t_delayed = t - eph['delta'] / c  # doctest: +REMOTE_DATA
        >>> vectors = Horizons('2P', id_type='designation', epochs=t_delayed.tdb.jd,
        ...                    location='@10').vectors(closest_apparition=True,
        ...                    refplane='ecliptic')  # doctest: +REMOTE_DATA
        >>> [vectors[k].quantity for k in ('x', 'y', 'z', 'vx', 'vy', 'vz')]
        ... # doctest: +REMOTE_DATA
        [<Quantity [3.83111326] AU>,
         <Quantity [-0.77324033] AU>,
         <Quantity [0.19606241] AU>,
         <Quantity [-0.00173363] AU / d>,
         <Quantity [0.00382319] AU / d>,
         <Quantity [0.00054533] AU / d>]

        """

        # barycentric coordinates
        coords = SkyCoord(
            ra=348.37706 * u.deg,
            dec=-1.86304 * u.deg,
            distance=3.90413464 * u.au,
            pm_ra_cosdec=6.308283 * u.arcsec / u.hr,
            pm_dec=4.270114 * u.arcsec / u.hr,
            radial_velocity=-4.19816 * u.km / u.s,
            obstime=Time("2022-08-02"),
        )
        state = State.from_skycoord(coords)

        # heliocentric vectors
        r = [3.83111326, -0.77324033, 0.19606241] * u.au
        v = [-0.00173363, 0.00382319, 0.00054533] * u.au / u.day
        assert u.allclose(state.r, r)
        assert u.allclose(state.v, v)

    def test_coords_property(self):
        """Test conversion of coords to internal state and back to coords."""
        coords = SkyCoord(
            ra=348.37706 * u.deg,
            dec=-1.86304 * u.deg,
            distance=3.90413464 * u.au,
            pm_ra_cosdec=6.308283 * u.arcsec / u.hr,
            pm_dec=4.270114 * u.arcsec / u.hr,
            radial_velocity=-4.19816 * u.km / u.s,
            obstime=Time("2022-08-02"),
        )
        state = State.from_skycoord(coords)
        new_coords = state.coords.transform_to(coords.frame)
        assert u.isclose(new_coords.ra, coords.ra)
        assert u.isclose(new_coords.dec, coords.dec)
        assert u.isclose(new_coords.distance, coords.distance)
        assert u.isclose(new_coords.pm_ra_cosdec, coords.pm_ra_cosdec)
        assert u.isclose(new_coords.pm_dec, coords.pm_dec)
        assert u.isclose(new_coords.radial_velocity, coords.radial_velocity)
        assert np.isclose((coords.obstime - new_coords.obstime).jd, 0)

    # def test_from_ephem(self):
    #     r = [3.83111326, -0.77324033, 0.19606241] * u.au
    #     v = [-0.00173363, 0.00382319, 0.00054533] * u.au / u.day
    #     t = Time("2022-08-02")
    #     eph = Ephem.from_dict(
    #         {
    #             "x": r[0],
    #             "y": r[1],
    #             "z": r[2],
    #             "vy": v[0],
    #             "vz": v[1],
    #             "vx": v[2],
    #             "date": t,
    #         },
    #         frame="heliocentriceclipticiau76",
    #     )
    #     state = State.from_ephem(eph)
    #     assert u.allclose(state.r, r)
    #     assert u.allclose(state.v, v)
    #     assert np.isclose((state.t - t).jd, 0)

    #     eph = Ephem.from_dict(
    #         {
    #             "ra": 348.37706 * u.deg,
    #             "dec": -1.86304 * u.deg,
    #             "delta": 3.90413464 * u.au,
    #             "RA*cos(Dec)_rate": 6.308283 * u.arcsec / u.hr,
    #             "Dec_rate": 4.270114 * u.arcsec / u.hr,
    #             "delta_rate": -4.19816 * u.km / u.s,
    #             "date": t,
    #         },
    #         frame="icrf",
    #     )
    #     state = State.from_ephem(eph)
    #     assert u.allclose(state.r, r)
    #     assert u.allclose(state.v, v)
    #     assert np.isclose((state.t - t).jd, 0)


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
    final = EarthGravity.solve(initial, t_f)

    assert np.allclose(initial.r.value, [0, 70710678.11865, 70710678.11865])
    assert np.allclose(initial.v.value, [0, -0.04464, 0.04464], atol=0.00001)
    assert np.allclose(final.r.value, [0, -70710678.11865, -70710678.11865])
    assert np.allclose(final.v.value, [0, 0.04464, -0.04464], atol=0.00001)


class TestSolarGravity:
    @pytest.mark.parametrize("r1_au", ([0.3, 1, 3, 10, 30]))
    def test_circular_orbit(self, r1_au):
        r1 = r1_au * u.au.to("km")
        s = np.sqrt(SolarGravity._GM / r1)
        half_period = np.pi * r1 / s

        r = [0, r1 / np.sqrt(2), r1 / np.sqrt(2)]
        v = [0, -s / np.sqrt(2), s / np.sqrt(2)]

        initial = State(r, v, Time("2023-01-01"))
        t_f = initial.t + half_period * u.s
        final = SolarGravity.solve(initial, t_f)

        assert np.allclose(final.r.value, -initial.r.value)
        assert np.allclose(final.v.value, -initial.v.value)

        t_f = initial.t + 2 * half_period * u.s
        final = SolarGravity.solve(initial, t_f)

        assert np.allclose(final.r.value, initial.r.value)
        assert np.allclose(final.v.value, initial.v.value)


class TestFreeExpansion:
    def test(self):
        r = [0, 1e6, 0]
        v = [0, -1, 1]

        initial = State(r, v, Time("2023-01-01"))
        t_f = initial.t + 1e6 * u.s
        final = FreeExpansion.solve(initial, t_f)

        assert np.allclose(final.r.value, [0, 0, 1e6], atol=2e-7)
        assert np.allclose(final.v.value, [0, -1, 1])


class TestSolarGravityAndRadiation:
    def test_reduced_gravity(self):
        """Radiation pressure is essentially a reduced gravity problem."""

        r1 = 1e8
        s = np.sqrt(SolarGravityAndRadiationPressure._GM / r1)

        initial = State([0, 0, r1], [0, s, 0], Time("2023-01-01"))
        t_f = initial.t + 1e6 * u.s
        beta = 0.1
        final1 = SolarGravityAndRadiationPressure.solve(initial, t_f, beta)

        class ReducedGravity(SolarGravity):
            _GM = (1 - beta) * SolarGravity._GM

        final2 = ReducedGravity.solve(initial, t_f)

        assert u.allclose(final1.r, final2.r)
        assert u.allclose(final1.v, final2.v)
