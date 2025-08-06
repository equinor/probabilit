from probabilit.distributions import (
    _fit_triangular_distribution,
    _pert_to_beta,
)
import pytest
from scipy.stats import triang
import numpy as np


class TestTriangular:
    @pytest.mark.parametrize("c", [0.1, 0.5, 0.7])
    @pytest.mark.parametrize("loc", [-1, 0, 1])
    @pytest.mark.parametrize("scale", [1, 10, 25])
    @pytest.mark.parametrize("low_perc", [0.01, 0.05, 0.1, 0.2])
    def test_triangular_roundstrips(self, c, loc, scale, low_perc):
        # Test round-trips
        mode = loc + c * scale
        high_perc = 0.8

        # Get parameters to optimize toward
        distr = triang(loc=loc, scale=scale, c=c)
        target_low, target_high = distr.ppf([low_perc, high_perc])

        # Found parameters
        loc_f, scale_f, c_f = _fit_triangular_distribution(
            mode=mode,
            low=target_low,
            high=target_high,
            low_perc=low_perc,
            high_perc=high_perc,
        )

        np.testing.assert_allclose([loc_f, scale_f, c_f], [loc, scale, c], atol=1e-8)

    @pytest.mark.parametrize("delta", [0.001, 0.01, 0.1, 0.2, 0.3, 0.4])
    def test_triangular_roundstrips_squeeze(self, delta):
        loc = 0
        scale = 10
        c = 0.8
        mode = loc + c * scale

        # Get parameters to optimize toward
        distr = triang(loc=loc, scale=scale, c=c)
        target_low, target_high = distr.ppf([delta, 1 - delta])

        # Found parameters
        loc_f, scale_f, c_f = _fit_triangular_distribution(
            mode=mode,
            low=target_low,
            high=target_high,
            low_perc=delta,
            high_perc=1 - delta,
        )

        np.testing.assert_allclose([loc_f, scale_f, c_f], [loc, scale, c], atol=1e-8)


class TestPERT:
    @pytest.mark.parametrize("gamma", [1, 3, 4, 7])
    @pytest.mark.parametrize("maximum", [10, 12, 14])
    def test_pert_properties(self, gamma, maximum):
        # Convert from PERT parameters to beta
        a, b, loc, scale = _pert_to_beta(
            minimum=1, mode=4, maximum=maximum, gamma=gamma
        )

        # The mode of the beta distribution (from Wikipedia)
        mode = (a - 1) / (a + b - 2)
        # The mode should be located in the correct positoin on [0, 1]
        np.testing.assert_allclose(mode, (4 - 1) / (maximum - 1))

        # Desired mean of PERT matches actual mean of beta
        mean = (1 + gamma * 4 + maximum) / (gamma + 2)
        np.testing.assert_allclose(mean, (a / (a + b)) * scale + loc)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
