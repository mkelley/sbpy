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
