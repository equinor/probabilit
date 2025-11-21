from probabilit.distributions import (
    _fit_triangular_distribution,
    _pert_to_beta,
    _pert_fit_min_max_from_percentiles,
    Lognormal,
    Uniform,
)
import pytest
from scipy.stats import triang, beta
import numpy as np


class TestTriangular:
    @pytest.mark.parametrize("c", [0.1, 0.2, 0.5, 0.7])
    @pytest.mark.parametrize("loc", [0, 10**6])
    @pytest.mark.parametrize("scale", [1, 10**6])
    @pytest.mark.parametrize("low_perc", [0, 0.01, 0.05, 0.1, 0.2, 0.3])
    @pytest.mark.parametrize("high_perc", [1.0, 0.99, 0.95, 0.9, 0.8, 0.7])
    def test_triangular_roundtrips(self, c, loc, scale, low_perc, high_perc):
        # Test round-trips
        mode = loc + c * scale

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

        # Check if distributions are close using pointwise RMSE
        x = np.linspace(loc, loc + scale, num=2**8)
        func_true = distr.pdf(x)
        func_est = triang(loc=loc_f, scale=scale_f, c=c_f).pdf(x)
        rmse = np.mean((func_true - func_est) ** 2) ** 0.5
        assert rmse < 1e-7

    @pytest.mark.parametrize("delta", [0.001, 0.01, 0.1, 0.2, 0.3, 0.4])
    def test_triangular_roundtrips_squeeze(self, delta):
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

        x = np.linspace(loc, loc + scale, num=2**8)
        func_true = distr.pdf(x)
        func_est = triang(loc=loc_f, scale=scale_f, c=c_f).pdf(x)
        rmse = np.mean((func_true - func_est) ** 2) ** 0.5
        assert rmse < 1e-7


class TestLognormal:
    @pytest.mark.parametrize(
        "mean,std",
        [
            (1.0, 0.5),
            (10.0, 1.0),
            (100.0, 20.0),
        ],
    )
    def test_lognormal_moments(self, mean, std):
        rng = np.random.default_rng(42)
        dist = Lognormal(mean, std)
        samples = dist.sample(10000, random_state=rng)

        np.testing.assert_allclose(np.mean(samples), mean, rtol=0.05)
        np.testing.assert_allclose(np.std(samples), std, rtol=0.05)

    @pytest.mark.parametrize(
        "mu,sigma",
        [
            (0.0, 0.5),
            (1.0, 1.0),
            (-0.5, 0.3),
        ],
    )
    def test_lognormal_from_log_params_moments(self, mu, sigma):
        rng = np.random.default_rng(42)
        dist = Lognormal.from_log_params(mu, sigma)
        samples = dist.sample(10000, random_state=rng)

        # Calculate expected moments from log-space parameters
        expected_mean = np.exp(mu + sigma**2 / 2)
        expected_variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
        expected_std = np.sqrt(expected_variance)

        np.testing.assert_allclose(np.mean(samples), expected_mean, rtol=0.05)
        np.testing.assert_allclose(np.std(samples), expected_std, rtol=0.05)


class TestPERT:
    @pytest.mark.parametrize("min", [-(10**6), 0, 2, 4])
    @pytest.mark.parametrize("max", [10, 12, 14, 10**6])
    @pytest.mark.parametrize("low_perc", [0.01, 0.1, 0.2])
    @pytest.mark.parametrize("high_perc", [0.99, 0.9, 0.7])
    def test_pert_roundtrips(self, min, max, low_perc, high_perc):
        mode = 5
        gamma = 4
        a, b, loc, scale = _pert_to_beta(
            minimum=min, mode=mode, maximum=max, gamma=gamma
        )

        distr = beta(a, b, loc, scale)
        low, high = distr.ppf([low_perc, high_perc])

        min_f, max_f = _pert_fit_min_max_from_percentiles(
            low, mode, high, low_perc=low_perc, high_perc=high_perc, gamma=gamma
        )

        # Check if distributions are close using pointwise RMSE
        x = np.linspace(loc, loc + scale, num=2**8)
        func_true = distr.pdf(x)

        # Calulate the estimated beta-function
        a_f, b_f, loc_f, scale_f = _pert_to_beta(
            minimum=min_f, mode=mode, maximum=max_f, gamma=gamma
        )
        func_est = beta(a_f, b_f, loc=loc_f, scale=scale_f).pdf(x)

        # Do the comparison
        rmse = np.mean((func_true - func_est) ** 2) ** 0.5
        assert rmse < 1e-5

    @pytest.mark.parametrize("gamma", [1, 3, 4, 7])
    @pytest.mark.parametrize("maximum", [10, 12, 14])
    def test_pert_to_beta(self, gamma, maximum):
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


class TestUniform:
    @pytest.mark.parametrize(
        "min_val,max_val", [(-5, 5), (0, 15), (10, 100), (-10, -2), (1, 2)]
    )
    def test_uniform_properties(self, min_val, max_val):
        rng = np.random.default_rng(42)
        dist = Uniform(min_val, max_val)
        samples = dist.sample(10000, random_state=rng)

        # Test that all samples are within bounds
        assert np.all(samples >= min_val)
        assert np.all(samples <= max_val)

        expected_mean = (min_val + max_val) / 2
        np.testing.assert_allclose(np.mean(samples), expected_mean, rtol=0.1, atol=0.1)

        expected_variance = (max_val - min_val) ** 2 / 12
        expected_std = np.sqrt(expected_variance)
        np.testing.assert_allclose(np.std(samples), expected_std, rtol=0.05)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
