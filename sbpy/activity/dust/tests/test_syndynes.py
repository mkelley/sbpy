# Licensed under a 3-clause BSD style license - see LICENSE.rst

import numpy as np
from astropy.time import Time
from ..syndynes import State, Syndynes, SpiceEphemerisTime


def test_spice_ephemeris_time():
    """Compare to SPICE result.

    spiceypy.utc2et('2022-08-01')
    712584069.1832777

    """

    t = Time('2022-08-01', scale='utc')
    assert np.isclose(t.et, 712584069.1832777)
