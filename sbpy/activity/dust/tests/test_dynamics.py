# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest

import numpy as np
import astropy.units as u
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    HeliocentricEclipticIAU76,
)

from .... import time  # for ephemeris time
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

    def test_init_shape_mismatch(self):
        with pytest.raises(ValueError):
            State(
                [1, 1, 0] * u.au,
                [[0, 0, 30], [30, 0, 0]] * u.km / u.s,
                Time("2022-08-02"),
                frame="heliocentriceclipticiau76",
            )

        with pytest.raises(ValueError):
            State(
                [1, 1, 0] * u.au,
                [30, 0, 0] * u.km / u.s,
                Time(["2022-08-02", "2023-08-02"]),
                frame="heliocentriceclipticiau76",
            )

    def test_rv_shapes(self):
        good = [1, 2, 3]
        bad = [1, 2, 3, 4]
        t = Time("2023-12-03")

        State(good * u.km, good * u.km / u.s, t)

        with pytest.raises(ValueError):
            State(good * u.km, bad * u.km / u.s, t)

        with pytest.raises(ValueError):
            State(bad * u.km, good * u.km / u.s, t)

        bad = [[[1, 2, 3], [1, 2, 3]]]

        with pytest.raises(ValueError):
            State(good * u.km, bad * u.km / u.s, t)

        with pytest.raises(ValueError):
            State(bad * u.km, good * u.km / u.s, t)

    def test_repr(self):
        state = State(
            [1, 1, 0] * u.au,
            [30, 0, 0] * u.km / u.s,
            Time("2022-08-02"),
            frame="heliocentriceclipticiau76",
        )
        assert (
            repr(state)
            == """<State (heliocentriceclipticiau76):
 r
    [1.49597871e+08 1.49597871e+08 0.00000000e+00] km
 v
    [30.  0.  0.] km / s
 t
    [7.12670469e+08]>"""
        )

    def test_len(self):
        state = State(
            [1, 1, 0] * u.au,
            [30, 0, 0] * u.km / u.s,
            Time("2022-08-02"),
            frame="heliocentriceclipticiau76",
        )
        assert len(state) == 1

        state = State(
            [[1, 1, 0], [1, 1, 0]] * u.au,
            [[30, 0, 0], [30, 0, 0]] * u.km / u.s,
            Time(["2022-08-02", "2023-08-02"]),
            frame="heliocentriceclipticiau76",
        )
        assert len(state) == 2

    def test_getitem(self):
        state = State(
            [[1, 1, 0], [1, 1, 1]] * u.au,
            [[30, 0, 0], [0, 30, 0]] * u.km / u.s,
            Time(["2022-08-02", "2023-08-02"]),
            frame="heliocentriceclipticiau76",
        )
        assert u.allclose(state[0].r, [1, 1, 0] * u.au)
        assert u.allclose(state[0].v, [30, 0, 0] * u.km / u.s)
        assert np.isclose((state.t[0] - Time("2022-08-02")).jd, 0)

        assert u.allclose(state[1].r, [1, 1, 1] * u.au)
        assert u.allclose(state[1].v, [0, 30, 0] * u.km / u.s)
        assert np.isclose((state.t[1] - Time("2023-08-02")).jd, 0)

    def test_operators(self):
        t = Time("2022-08-02")
        a = State([1, 2, 3] * u.au, [10, 20, -30] * u.km / u.s, t)
        b = State([4, 5, 6] * u.au, [100, 200, -300] * u.km / u.s, t)

        x = a + b
        assert u.allclose(x.r, [5, 7, 9] * u.au)
        assert u.allclose(x.v, [110, 220, -330] * u.km / u.s)

        x = -a
        assert u.allclose(x.r, [-1, -2, -3] * u.au)
        assert u.allclose(x.v, [-10, -20, 30] * u.km / u.s)

        x = a - b
        assert u.allclose(x.r, [-3, -3, -3] * u.au)
        assert u.allclose(x.v, [-90, -180, 270] * u.km / u.s)

        c = State([4, 5, 6] * u.au, [100, 200, -300] * u.km / u.s, t + 1 * u.s)
        with pytest.raises(ValueError):
            a + c

    def test_abs(self):
        t = Time("2022-08-02")
        a = State([1, 2, 3] * u.au, [10, 20, -30] * u.km / u.s, t)
        b = State.from_states(
            [
                a,
                State([4, 5, 6] * u.au, [100, 200, -300] * u.km / u.s, t),
            ]
        )

        def length(*x):
            return np.sqrt(np.sum(np.array(x) ** 2))

        r, v = abs(a)
        assert u.isclose(r, length(1, 2, 3) * u.au, atol=1 * u.um, rtol=1e-10)
        assert u.isclose(
            v, length(10, 20, 30) * u.km / u.s, atol=1 * u.um / u.s, rtol=1e-10
        )

        r, v = abs(b)
        assert u.allclose(
            r, [length(1, 2, 3), length(4, 5, 6)] * u.au, atol=1 * u.um, rtol=1e-10
        )
        assert u.allclose(
            v,
            [length(10, 20, 30), length(100, 200, 300)] * u.km / u.s,
            atol=1 * u.um / u.s,
            rtol=1e-10,
        )

    def test_vector_properties(self):
        state = State(
            [1, 2, 4] * u.au,
            [30, -10, 5] * u.km / u.s,
            Time("2022-08-02"),
            frame="heliocentriceclipticiau76",
        )
        assert u.isclose(state.x, 1 * u.au)
        assert u.isclose(state.y, 2 * u.au)
        assert u.isclose(state.z, 4 * u.au)
        assert u.isclose(state.v_x, 30 * u.km / u.s)
        assert u.isclose(state.v_y, -10 * u.km / u.s)
        assert u.isclose(state.v_z, 5 * u.km / u.s)
        assert np.allclose(
            state.rv, [1.49597871e8, 2 * 1.49597871e8, 4 * 1.49597871e8, 30, -10, 5]
        )

        state = State(
            [[1, 2, 4], [7, 8, 9]] * u.au,
            [[30, -10, 5], [5, 4, 6]] * u.km / u.s,
            Time("2022-08-02"),
            frame="heliocentriceclipticiau76",
        )
        assert u.allclose(state.x, [1, 7] * u.au)
        assert u.allclose(state.y, [2, 8] * u.au)
        assert u.allclose(state.z, [4, 9] * u.au)
        assert u.allclose(state.v_x, [30, 5] * u.km / u.s)
        assert u.allclose(state.v_y, [-10, 4] * u.km / u.s)
        assert u.allclose(state.v_z, [5, 6] * u.km / u.s)
        assert np.allclose(
            state.rv,
            [
                [1.49597871e8, 2 * 1.49597871e8, 4 * 1.49597871e8, 30, -10, 5],
                [7 * 1.49597871e8, 8 * 1.49597871e8, 9 * 1.49597871e8, 5, 4, 6],
            ],
        )

    def test_from_skycoord(self):
        """Test initialization from RA, Dec, distance.

        Coordinates are 2P/Encke at 2022-08-02 UTC

        Initialize using RA, Dec, and distance from Horizons.  These are
        light travel time corrected quantities.

        Compare to vectors from Horizons, explicitly adjusted for light travel
        time.
        RA, Dec is defined in the ICRF, so get barycentric coordinates from
        Horizons.

            from astropy.time import Time
            from sbpy.data import Ephem

            t = Time("2022-08-02")  # UTC
            eph = Ephem.from_horizons("2P", id_type="designation",
                                      epochs=t,
                                      closest_apparition=True,
                                      location="@0",
                                      extra_precision=True)
            [eph[k] for k in ("ra", "dec", "Delta", "RA*cos(Dec)_rate",
            "Dec_rate", "delta_rate")]

        [<MaskedQuantity [348.37706151] deg>,
         <MaskedQuantity [-1.8630357] deg>,
         <MaskedQuantity [3.90413272] AU>,
         <MaskedQuantity [6.308291] arcsec / h>,
         <MaskedQuantity [4.270122] arcsec / h>,
         <MaskedQuantity [-4.198173] km / s>]

            from astroquery.jplhorizons import Horizons
            import astropy.units as u
            from astropy.constants import c

            sun_vectors = Horizons("Sun", epochs=t.tdb.jd, location="@0").vectors(
                closest_apparition=True, refplane="ecliptic")

            ltt = eph["lighttime"]
            for i in range(3):
                t_delayed = t - ltt
                comet_vectors = Horizons("2P", id_type="designation", epochs=t_delayed.tdb.jd,
                    location="@0").vectors(closest_apparition=True, refplane="ecliptic")
                ltt = np.sqrt(np.sum([float(comet_vectors[x] - sun_vectors[x])**2 for x in "xyz"])) * u.au / c

            print([float(comet_vectors[x] - sun_vectors[x]) for x in "xyz"])
            print([float(comet_vectors[f"v{x}"] - sun_vectors[f"v{x}"]) for x in "xyz"])

        [3.831111550969744, -0.773239683630105, 0.19606224529093624]
        [-0.0017336347284832058, 0.003823198388051765, 0.0005453275902763597]

        """

        # barycentric coordinates
        coords = SkyCoord(
            ra=348.37706151 * u.deg,
            dec=-1.8630357 * u.deg,
            distance=3.90413272 * u.au,
            pm_ra_cosdec=6.308291 * u.arcsec / u.hr,
            pm_dec=4.270122 * u.arcsec / u.hr,
            radial_velocity=-4.198173 * u.km / u.s,
            obstime=Time("2022-08-02"),
        ).transform_to("heliocentriceclipticiau76")
        state = State.from_skycoord(coords)

        # heliocentric vectors, light travel time corrected
        r = [3.831111550969744, -0.773239683630105, 0.19606224529093624] * u.au
        v = (
            [-0.0017336347284832058, 0.003823198388051765, 0.0005453275902763597]
            * u.au
            / u.day
        )

        # if DE440s is used for the SkyCoord frame transformation, then the
        # agreement is better than 40 km
        assert u.allclose(state.r, r, atol=130 * u.km, rtol=1e-9)
        assert u.allclose(state.v, v, atol=2 * u.mm / u.s, rtol=1e-9)

    def test_skycoord_property(self):
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

        new_coords = State.from_skycoord(coords).to_skycoord()
        new_coords.representation_type = "spherical"

        assert u.isclose(new_coords.ra, coords.ra)
        assert u.isclose(new_coords.dec, coords.dec)
        assert u.isclose(new_coords.distance, coords.distance)
        assert u.isclose(new_coords.pm_ra * np.cos(new_coords.dec), coords.pm_ra_cosdec)
        assert u.isclose(new_coords.pm_dec, coords.pm_dec)
        assert u.isclose(new_coords.radial_velocity, coords.radial_velocity)
        assert np.isclose((coords.obstime - new_coords.obstime).jd, 0)

        # test skycoords representation using string
        new_coords = State(
            [1, 2, 3] * u.km,
            [4, 5, 6] * u.km / u.s,
            coords.obstime,
            frame="heliocentriceclipticiau76",
        ).to_skycoord()
        assert isinstance(new_coords.frame, HeliocentricEclipticIAU76)

    def test_observe(self):
        """Observe comet Encke from the Earth.

        Get heliocentric vectors for both:

            from astropy.time import Time
            import astropy.constants as const
            from astroquery.jplhorizons import Horizons

            def print_rect(data):
                print([float(data[x]) for x in "xyz"])
                print([float(data[f"v{x}"]) for x in "xyz"])

            # Horizons.vectors uses TDB
            t = Time("2022-08-02", scale="tdb")
            q = Horizons(399,
                        epochs=t.tdb.jd,
                        location="@10")
            earth_data = q.vectors(refplane="ecliptic", aberrations="geometric")
            print_rect(earth_data)

        [0.644307773374595, -0.7841972063903224, 3.391591147538755e-05]
        [0.01301350774835054, 0.01086343081039913, -2.045610960977488e-07]

            q = Horizons("2P",
                         id_type="designation",
                         epochs=t.tdb.jd,
                         location="@10")
            comet_data = q.vectors(refplane="ecliptic", closest_apparition=True, aberrations="geometric")
            print_rect(comet_data)

        [0.644307773374595, -0.7841972063903224, 3.391591147538755e-05]
        [0.01301350774835054, 0.01086343081039913, -2.045610960977488e-07]

        Compare to Horizons's ICRS coordinates, adjusted for light travel time

            # Horizons.ephemerides uses UTC
            delta = np.sqrt(np.sum([(earth_data[x] - comet_data[x])**2 for x in "xyz"])) * u.au
            q = Horizons("2P",
                         id_type="designation",
                         epochs=t.utc.jd,
                         location="500")
            comet_eph = q.ephemerides(closest_apparition=True, extra_precision=True)
            print(comet_eph["RA", "DEC", "delta"])

             RA        DEC         delta
            deg        deg           AU
        ----------- --------- ----------------
        358.7792014 3.3076422 3.19284032416352

        Compare to spiceypy

            import spiceypy
            import sbpy.time

            heclip = [
                3.831072342963699 - 0.6443181936605892,
                -0.7731534793031922 - -0.7841885076117138,
                0.1960745721596477 - 3.39157477079006e-05
            ]
            t = Time("2022-08-02", scale="tdb")
            xmat = spiceypy.pxform("ECLIPJ2000", "J2000", t.et)
            icrf = spiceypy.mxv(xmat, heclip)
            delta, ra, dec = spiceypy.recrad(icrf)
            delta
            Out[30]: 3.1927974753993995
            np.degrees((ra, dec))
            Out[125]: array([358.78017634,   3.3083223 ])

        """

        frame = "heliocentriceclipticiau76"
        t = Time("2022-08-02", scale="tdb")
        r = [0.644307773374595, -0.7841972063903224, 3.391591147538755e-05]
        v = [0.01301350774835054, 0.01086343081039913, -2.045610960977488e-07]
        earth = State(r * u.au, v * u.au / u.day, t, frame=frame)

        r = [3.831073731476054, -0.7731565407334685, 0.196074135515735]
        v = [-0.001734044117773847, 0.003823283252059133, 0.0005453056645337847]
        comet = State(r * u.au, v * u.au / u.day, t, frame=frame)

        angular_tol = 1 * u.arcsec
        linear_tol = 1e5 * u.km

        # astropy and spiceypy agree within an arcsec
        coords = earth.observe(comet, "icrs")
        assert u.isclose(coords.ra, 358.78017634 * u.deg, atol=angular_tol, rtol=1e-8)
        assert u.isclose(coords.dec, 3.3083223 * u.deg, atol=angular_tol, rtol=1e-8)
        assert u.isclose(
            coords.distance, 3.1927974753994 * u.au, atol=linear_tol, rtol=1e-8
        )

        # but astropy and Horizons disagree a little bit more, maybe due to the
        # light travel time treatment?
        angular_tol = 4 * u.arcsec
        assert u.isclose(coords.ra, 358.7792014 * u.deg, atol=angular_tol, rtol=1e-8)
        assert u.isclose(coords.dec, 3.3076422 * u.deg, atol=angular_tol, rtol=1e-8)
        assert u.isclose(
            coords.distance, 3.19284032416352 * u.au, atol=linear_tol, rtol=1e-8
        )

        comet.t = Time("2022-08-02", scale="utc")
        with pytest.raises(ValueError):
            earth.observe(comet)

    def test_from_states(self):
        t = Time("2022-08-02")
        a = State([1, 2, 3], [4, 5, 6], t, frame="icrs")
        b = State([1, 2, 3], [4, 5, 6], t, frame="heliocentriceclipticiau76")

        with pytest.raises(ValueError):
            State.from_states([a, b])

        # execute without error
        a.frame = "heliocentriceclipticiau76"
        c = State.from_states([a, b])

        # execute without error
        a.frame = HeliocentricEclipticIAU76(obstime=t)
        b.frame = HeliocentricEclipticIAU76(obstime=t)
        c = State.from_states([a, b])

        assert u.allclose(c[0].r, a.r)
        assert u.allclose(c[0].v, a.v)
        assert np.isclose((c[0].t - a.t).jd, 0)
        assert u.allclose(c[1].r, b.r)
        assert u.allclose(c[1].v, b.v)
        assert np.isclose((c[1].t - b.t).jd, 0)
        assert str(c.frame).lower() == str(a.frame).lower()

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

    solver = EarthGravity()

    r1 = 1e8
    s = np.sqrt(solver.GM.to_value("km3/s2") / r1)
    half_period = np.pi * r1 / s

    r = [0, r1 / np.sqrt(2), r1 / np.sqrt(2)]
    v = [0, -s / np.sqrt(2), s / np.sqrt(2)]

    initial = State(r, v, Time("2023-01-01"))
    t_f = initial.t + half_period * u.s
    final = solver.solve(initial, t_f)

    assert np.allclose(initial.r.value, [0, 70710678.11865, 70710678.11865])
    assert np.allclose(initial.v.value, [0, -0.04464, 0.04464], atol=0.00001)
    assert np.allclose(final.r.value, [0, -70710678.11865, -70710678.11865])
    assert np.allclose(final.v.value, [0, 0.04464, -0.04464], atol=0.00001)

    t_f = initial.t + 5 * half_period * u.s
    final = solver.solve(initial, t_f)

    assert np.allclose(initial.r.value, [0, 70710678.11865, 70710678.11865])
    assert np.allclose(initial.v.value, [0, -0.04464, 0.04464], atol=0.00001)
    assert np.allclose(final.r.value, [0, -70710678.11865, -70710678.11865])
    assert np.allclose(final.v.value, [0, 0.04464, -0.04464], atol=0.00001)


class TestSolarGravity:
    @pytest.mark.parametrize("r1_au", ([0.3, 1, 3, 10, 30]))
    def test_circular_orbit(self, r1_au):
        solver = SolarGravity()

        r1 = r1_au * u.au.to("km")
        s = np.sqrt(solver.GM.to_value("km3/s2") / r1)
        half_period = np.pi * r1 / s

        r = [0, r1 / np.sqrt(2), r1 / np.sqrt(2)]
        v = [0, -s / np.sqrt(2), s / np.sqrt(2)]

        initial = State(r, v, Time("2023-01-01"))
        t_f = initial.t + half_period * u.s
        final = solver.solve(initial, t_f)

        assert np.allclose(final.r.value, -initial.r.value)
        assert np.allclose(final.v.value, -initial.v.value)

        t_f = initial.t + 2 * half_period * u.s
        final = solver.solve(initial, t_f)

        assert np.allclose(final.r.value, initial.r.value)
        assert np.allclose(final.v.value, initial.v.value)


class TestFreeExpansion:
    def test(self):
        r = [0, 1e6, 0]
        v = [0, -1, 1]

        solver = FreeExpansion()

        initial = State(r, v, Time("2023-01-01"))
        t_f = initial.t + 1e6 * u.s
        final = solver.solve(initial, t_f)

        assert np.allclose(final.r.value, [0, 0, 1e6], atol=2e-7)
        assert np.allclose(final.v.value, [0, -1, 1])


class TestSolarGravityAndRadiationPressure:
    def test_reduced_gravity(self):
        """Radiation pressure is essentially a reduced gravity problem."""
        solver = SolarGravityAndRadiationPressure()

        r1 = 1e8
        s = np.sqrt(solver.GM.to_value("km3/s2") / r1)

        initial = State([0, 0, r1], [0, s, 0], Time("2023-01-01"))
        t_f = initial.t + 1e6 * u.s
        beta = 0.1
        final1 = solver.solve(initial, t_f, beta)

        class ReducedGravity(SolarGravity):
            _GM = (1 - beta) * SolarGravity._GM

        solver = ReducedGravity()
        final2 = solver.solve(initial, t_f)

        assert u.allclose(final1.r, final2.r)
        assert u.allclose(final1.v, final2.v)
