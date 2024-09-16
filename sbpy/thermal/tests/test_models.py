# Licensed under a 3-clause BSD style license - see LICENSE.rst

import astropy.units as u
from ..models import StandardThermalModel


class TestStandardThermalModel:
    def test_lebofsky86_ceres():
        """1983 September 16 data from Table Va."""

        # filters = "KMNQ"
        wave = [2.22, 4.8, 10.6, 20.2] * u.um
        fluxd = [1.56, 3.03, 419, 1147] * u.Jy

        phys = {
            "D": 917.0 * u.km,
            "Ap": 0.0987,
            "beta": 0.74,
        }

        # convert geometric albedo to bolometric albedo using phase integral for
        # Ceres from Table I
        phys["A"] = phys["Ap"] * 0.366

        eph = {
            "rh": 2.982 * u.au,
            "delta": 2.141 * u.au,
            "phase": 12.5 * u.deg,
        }

        STM = StandardThermalModel(phys)

        result = STM.total_flux_density(wave, eph, unit=fluxd.unit)

        assert u.isclose(result, fluxd)
