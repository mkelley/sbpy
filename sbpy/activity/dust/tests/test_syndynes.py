# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import astropy.units as u
import astropy.constants as const
from astropy.coordinates import SkyCoord
from astropy.time import Time
from ....data import Ephem
from ..syndynes import State, Syndynes, SpiceEphemerisTime


def test_spice_ephemeris_time():
    """Compare to SPICE result.

    spiceypy.utc2et('2022-08-01')
    712584069.1832777

    """

    t = Time('2022-08-01', scale='utc')
    assert np.isclose(t.et, 712584069.1832777)


class TestState:
    def test_init(self):
        t = Time('2022-08-02')
        coords = SkyCoord(x=1 * u.au, y=1 * u.au, z=0 * u.au,
                          v_x=30 * u.km / u.s, v_y=0 * u.km / u.s,
                          v_z=0 * u.km / u.s, obstime=t,
                          frame='heliocentriceclipticiau76',
                          representation_type='cartesian')
        state = State(coords)

        # test properties
        assert u.allclose(state.r, [1, 1, 0] * u.au)
        assert u.allclose(state.v, [30000, 0, 0] * u.m / u.s)
        assert np.isclose((state.t - t).jd, 0)

        # test internal state values
        assert np.allclose(state._r, ([1, 1, 0] * u.au).to_value('km'))
        assert np.allclose(
            state._v, ([30, 0, 0] * u.km / u.s).to_value('km / s'))
        assert np.isclose(state._t, t.tdb.to_value('et'))

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

        coords = SkyCoord(ra=348.37706 * u.deg,
                          dec=-1.86304 * u.deg,
                          distance=3.90413464 * u.au,
                          pm_ra_cosdec=6.308283 * u.arcsec / u.hr,
                          pm_dec=4.270114 * u.arcsec / u.hr,
                          radial_velocity=-4.19816 * u.km / u.s,
                          obstime=Time('2022-08-02'))
        state = State(coords)

        r = [3.83111326, -0.77324033, 0.19606241] * u.au
        v = [-0.00173363, 0.00382319, 0.00054533] * u.au / u.day
        assert u.allclose(state.r, r)
        assert u.allclose(state.v, v)

    def test_from_vectors(self):
        r = u.Quantity([1, 1, 0], 'au')
        v = u.Quantity([30, 0, 0], 'km/s')
        t = Time('2022-08-02')
        state = State.from_vectors(r, v, t)

        # test properties
        assert u.allclose(state.r, [1, 1, 0] * u.au)
        assert u.allclose(state.v, [30000, 0, 0] * u.m / u.s)
        assert np.isclose((state.t - t).jd, 0)

        # test internal state values
        assert np.allclose(state._r, r.to_value('km'))
        assert np.allclose(state._v, v.to_value('km / s'))
        assert np.isclose(state._t, t.tdb.to_value('et'))

    def test_coords_property(self):
        """Test conversion of coords to internal state and back to coords."""
        coords = SkyCoord(ra=348.37706 * u.deg,
                          dec=-1.86304 * u.deg,
                          distance=3.90413464 * u.au,
                          pm_ra_cosdec=6.308283 * u.arcsec / u.hr,
                          pm_dec=4.270114 * u.arcsec / u.hr,
                          radial_velocity=-4.19816 * u.km / u.s,
                          obstime=Time('2022-08-02'))
        state = State(coords)
        new_coords = state.coords.transform_to(coords.frame)
        assert u.isclose(new_coords.ra, coords.ra)
        assert u.isclose(new_coords.dec, coords.dec)
        assert u.isclose(new_coords.distance, coords.distance)
        assert u.isclose(new_coords.pm_ra_cosdec, coords.pm_ra_cosdec)
        assert u.isclose(new_coords.pm_dec, coords.pm_dec)
        assert u.isclose(new_coords.radial_velocity, coords.radial_velocity)
        assert np.isclose((coords.obstime - new_coords.obstime).jd, 0)

    def test_from_ephem(self):
        r = [3.83111326, -0.77324033, 0.19606241] * u.au
        v = [-0.00173363, 0.00382319, 0.00054533] * u.au / u.day
        t = Time('2022-08-02')
        eph = Ephem.from_dict({'x': r[0],
                               'y': r[1],
                               'z': r[2],
                               'vy': v[0],
                               'vz': v[1],
                               'vx': v[2],
                               'date': t},
                              frame='heliocentriceclipticiau76')
        state = State.from_ephem(eph)
        assert u.allclose(state.r, r)
        assert u.allclose(state.v, v)
        assert np.isclose((state.t - t).jd, 0)

        eph = Ephem.from_dict({'ra': 348.37706 * u.deg,
                               'dec': -1.86304 * u.deg,
                               'delta': 3.90413464 * u.au,
                               'RA*cos(Dec)_rate': 6.308283 * u.arcsec / u.hr,
                               'Dec_rate': 4.270114 * u.arcsec / u.hr,
                               'delta_rate': -4.19816 * u.km / u.s,
                               'date': t},
                              frame='icrf')
        state = State.from_ephem(eph)
        assert u.allclose(state.r, r)
        assert u.allclose(state.v, v)
        assert np.isclose((state.t - t).jd, 0)
