# Licensed under a 3-clause BSD style license - see LICENSE.rst

import pytest
import numpy as np
import astropy.units as u
from ..surface import Surface, LambertianSurface

from numpy import pi


@pytest.mark.parametrize(
    "a, expected",
    (
        [0 * u.deg, 0 * u.deg],
        [90 * u.deg, 90 * u.deg],
        [180 * u.deg, 180 * u.deg],
        [181 * u.deg, 179 * u.deg],
        [0 * u.rad, 0 * u.rad],
        [pi / 2 * u.rad, pi / 2 * u.rad],
        [pi * u.rad, pi * u.rad],
        [(pi + 0.1) * u.rad, (pi - 0.1) * u.rad],
    ),
)
def test_validate_normal_angle(a, expected):
    assert Surface._validate_normal_angle(a) == expected
    assert Surface._validate_normal_angle(-a) == expected


class TestLambertianSurface:
    @pytest.mark.parametrize(
        "i, e, expected",
        (
            [0 * u.deg, 0 * u.deg, 0.1 * u.dimensionless_unscaled],
            [0 * u.deg, 60 * u.deg, 0.05 * u.dimensionless_unscaled],
            [60 * u.deg, 60 * u.deg, 0.025 * u.dimensionless_unscaled],
            [0 * u.deg, 90 * u.deg, 0 * u.dimensionless_unscaled],
            [0 * u.deg, 100 * u.deg, 0 * u.dimensionless_unscaled],
            [90 * u.deg, 0 * u.deg, 0 * u.dimensionless_unscaled],
            [100 * u.deg, 0 * u.deg, 0 * u.dimensionless_unscaled],
        ),
    )
    def test_reflectance(self, i, e, expected):
        surface = LambertianSurface({"A": 0.1})
        result = surface.reflectance(i, e)
        assert u.isclose(result, expected)
        assert result.unit == expected.unit

    @pytest.mark.parametrize(
        "i, expected",
        (
            [0 * u.deg, 0.9 * u.dimensionless_unscaled],
            [60 * u.deg, 0.45 * u.dimensionless_unscaled],
            [90 * u.deg, 0 * u.dimensionless_unscaled],
            [100 * u.deg, 0 * u.dimensionless_unscaled],
        ),
    )
    def test_absorptance(self, i, expected):
        surface = LambertianSurface({"A": 0.1})
        result = surface.absorptance(i)
        assert u.isclose(result, expected)
        assert result.unit == expected.unit


# @pytest.mark.parametrize(
#     "theta_i, expected",
#     (
#         [0 * u.deg, 1 * u.Jy / u.sr],
#         [60 * u.deg, 0.5 * u.Jy / u.sr],
#         [90 * u.deg, 0 * u.Jy / u.sr],
#         [np.pi / 2 * u.rad, 0 * u.Jy / u.sr],
#         [100 * u.deg, 0 * u.Jy / u.sr],
#     ),
# )
# def test_surface_spectral_flux_density(theta_i, expected):
#     surface = Surface({"A": 1})
#     result = surface.surface_radiance(theta_i, 1 * u.Jy)
#     assert u.isclose(result, expected)
#     assert result.unit == expected.unit
