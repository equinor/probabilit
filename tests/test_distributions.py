from probabilit.distributions import (
    _triang_params_from_perc,
    _triang_extract,
    _pert_to_beta,
)
import pytest
from scipy.stats import triang
import numpy as np


class TestTriangular:
    @pytest.mark.parametrize("c", np.linspace(0.5, 0.95, num=7))
    @pytest.mark.parametrize("scale", [1, 10, 100, 1000])
    @pytest.mark.parametrize("high_perc", [0.7, 0.8, 0.9])
    def test_triang_params_from_perc(self, c, scale, high_perc):
        # Test round-trips
        loc = 0
        initial = np.array([loc, scale, c])
        dist = triang(loc=loc, scale=scale, c=c)
        p10, mode, high = _triang_extract(dist, high_perc=high_perc)
        if (p10 < mode - 0.01) and (high > mode + 0.01):
            loc, scale, c = _triang_params_from_perc(
                p10, mode, high, high_perc=high_perc
            )
            final = np.array([loc, scale, c])
            np.testing.assert_allclose(final, initial, atol=1e-3)


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
