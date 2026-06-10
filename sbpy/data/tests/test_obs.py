# Licensed under a 3-clause BSD style license - see LICENSE.rst
import os
import pytest

from astropy.io import ascii

from .. import obs
from ..obs import Obs
from ..core import QueryError

pytest.importorskip("astroquery")


def _data_path(filename):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    return os.path.join(data_dir, filename)


def mpc_get_observations_from_cache(targetid, id_type=None, **kwargs):
    """Read and return cached table.

    To generate the local data:

    from astroquery.mpc import MPC

    objects = ["9530", "2024 YR4", "100P"]
    for i, obj in enumerate(objects):
        obs = MPC.get_observations(obj)
        obs = obs[:10 + i]  # only save 10 + i observations
        obs.write(f"TestObs-from_mpc-{obj}.ecsv")

    """

    fn = _data_path(f"TestObs-from_mpc-{targetid}.ecsv")
    if not os.path.exists(fn):
        raise ValueError

    return ascii.read(fn)


@pytest.fixture(autouse=True)
def patch_astroquery(monkeypatch):
    """Replace astroquery MPC.get_observations"""

    monkeypatch.setattr(obs.MPC, "get_observations", mpc_get_observations_from_cache)


class TestObs:
    def test_from_mpc_asteroid_numbered(self):
        tab = Obs.from_mpc("9530")
        assert len(tab) == 10

    def test_from_mpc_asteroid_designation(self):
        tab = Obs.from_mpc("2024 YR4")
        assert len(tab) == 11

    def test_from_mpc_comet(self):
        tab = Obs.from_mpc("100P")
        assert len(tab) == 12

    def test_from_mpc_error(self):
        with pytest.raises(QueryError):
            Obs.from_mpc("C/2222 X2")
