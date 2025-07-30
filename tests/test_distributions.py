from probabilit.distributions import _triang_params_from_perc, _triang_extract
import pytest
from scipy.stats import triang
import numpy as np


class TestTriangular:
    @pytest.mark.parametrize("c", np.linspace(0.5, 0.95, num=7))
    @pytest.mark.parametrize("scale", [1, 10, 100, 1000])
    def test_triang_params_from_perc(self, c, scale):
        # Test round-trips
        loc = 0
        initial = np.array([loc, scale, c])
        dist = triang(loc=loc, scale=scale, c=c)
        p10, mode, p90 = _triang_extract(dist)
        if (p10 < mode - 0.01) and (p90 > mode + 0.01):
            loc, scale, c = _triang_params_from_perc(p10, mode, p90)
            final = np.array([loc, scale, c])
            np.testing.assert_allclose(final, initial, atol=1e-3)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
