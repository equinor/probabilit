from probabilit.sampling import sample
from probabilit.modeling import EmpiricalDistribution, Constant
from probabilit.math import Floor, Equal, All, Min, Max, Exp, Log, IF
from probabilit.distributions import Distribution, Triangular
import numpy as np


class TestModelingExamples:
    def test_die_problem(self):
        """If we throw 2 die, what is the probability that each one ends up
        with the same number?"""

        die1 = Floor(1 + Distribution("uniform") * 6)
        die2 = Floor(1 + Distribution("uniform") * 6)
        equal = Equal(die1, die2)

        samples = sample(equal, 999, random_state=42)
        assert len(samples) == 999

        np.testing.assert_allclose(samples.mean(), 1 / 6, atol=0.01)

    def test_estimating_pi(self):
        """Consider the unit square [0, 1]^2. The area of the square is 1.
        The area of a quarter circle is pi * r^2 / 4 = pi / 4.
        So the fraction (quarter circle area) / (total area) = pi / 4.

        Use this to estimate pi.
        """

        x = Distribution("uniform")
        y = Distribution("uniform")
        inside = x**2 + y**2 < 1
        pi_estimate = 4 * inside

        samples = sample(pi_estimate, 9999, random_state=37)
        np.testing.assert_allclose(samples.mean(), np.pi, atol=0.01)

    def test_broken_stick_problem(self):
        """Consider a stick of length 1. Pick two points uniformly at random on
        the stick, and break the stick at those points. What is the probability
        that the three segments obtained in this way form a triangle?

        Of course this is the probability that no one of the short sticks is
        longer than 1/2. This probability turns out to be 1/4.

        https://sites.math.duke.edu/education/webfeatsII/gdrive/Team%20D/project/brokenstick.htm
        https://mathoverflow.net/questions/2014/if-you-break-a-stick-at-two-points-chosen-uniformly-the-probability-the-three-r
        """

        # Cuts along the stick
        cut1 = Distribution("uniform", loc=0, scale=1)
        cut2 = Distribution("uniform", loc=0, scale=1)

        # The lengths
        length1 = Min(cut1, cut2)
        length2 = Max(cut1, cut2) - Min(cut1, cut2)
        length3 = 1 - Max(cut1, cut2)

        # No one of the short sticks is longer than 1/2 <=> all are shorter
        prob = All(length1 < 1 / 2, length2 < 1 / 2, length3 < 1 / 2)

        samples = sample(prob, 9999, random_state=42)
        np.testing.assert_allclose(samples.mean(), 1 / 4, atol=0.01)

    def test_mutual_fund_problem(self):
        """Suppose you save 1200 units of money per year and that the yearly
        interest rate has a   distribution `N(1.11, 0.15)`.
        How much money will you have over a 20 year horizon?

        From: https://curvo.eu/backtest/en/market-index/sp-500?currency=eur
        In the last 33 years, the S&P 500 index (in EUR) had a compound annual
        growth rate of 10.83%, a standard deviation of 15.32%, and a Sharpe ratio of 0.66.
        """

        saved_per_year = 1200
        returns = 0
        for year in range(20):
            interest = Distribution("norm", loc=1.11, scale=0.15)
            returns = returns * interest + saved_per_year
        samples = sample(returns, 999, random_state=42)

        # Regression test essentially
        print(samples.mean(), samples.std())
        np.testing.assert_allclose(samples.mean(), 77085.8183243356)
        np.testing.assert_allclose(samples.std(), 33664.80313953588)

    def test_total_person_hours(self):
        """Based on Example 19.2 from Risk Analysis: A Quantitative Guide, 3rd Edition by David Vose.

        Estimate the number of person-hours requires to rivet 562 plates of a ship's hull.
        The quickest anyone has ever riveted a single plate is 3h 45min, while the worst time recorded is 5h 30min.
        Most likely value is estimated to be 4h 15min.
        What is the total person-hours?

        Naively, we could model the problem as:
        total_person_hours = 562 * Triangular(3.75, 4.25, 5.5),
        but note that the triangular distribution here models the uncertainty of an individual plate,
        but we are using it as if it were the distribution of the average time for 562 plates.

        A straight forward approach that gives the correct answer is to add 562 triangular distributions.
        """

        rng = np.random.default_rng(42)
        num_rivets = 562
        total_person_hours = 0

        for i in range(num_rivets):
            total_person_hours += Triangular(
                low=3.75, mode=4.25, high=5.5, low_perc=0, high_perc=1.0
            )

        num_samples = 1000
        res_total_person_hours = sample(
            total_person_hours, num_samples, random_state=rng
        )

        # The mean and standard deviation of a Triangular(3.75, 4.25, 5.5) are 4.5 and 0.368,
        # so by the Central Limit Theoreom we have that
        # total_person_hours = Normal(4.5 * 562, 0.368 * sqrt(562)) = Normal(2529, 8.724)
        expected_mean = 4.5 * num_rivets
        expected_std = 0.368 * np.sqrt(num_rivets)

        sample_mean = np.mean(res_total_person_hours)
        sample_std = np.std(res_total_person_hours, ddof=1)

        # Within 2% of theoretical values
        np.testing.assert_allclose(sample_mean, expected_mean, rtol=0.02)
        np.testing.assert_allclose(sample_std, expected_std, rtol=0.02)

    def test_conditional_if_statement(self):
        """Suppose mens height has distribution N(176, 7.1).
        What is the distribution of the difference between height of two men?
        Caveat: there is a 10% chance that the two men are identical twins,
        and in that case their height should be perfectly equal.
        """

        # Height of two random men
        height1 = Distribution("norm", loc=176, scale=7.1)
        height2 = Distribution("norm", loc=176, scale=7.1)

        # If they are twins, their height should be perfectly correlated
        is_twin = Distribution("bernoulli", p=0.1)

        height2 = IF(is_twin, height1, height2)

        # This is the answer to the question
        (
            abs_diff_samples,
            height1_samples,
            height2_samples,
        ) = sample([abs(height2 - height1), height1, height2], 999, random_state=42)

        # At least one of the realizations should be identical
        assert np.any(np.isclose(height1_samples, height2_samples))
        np.testing.assert_allclose(
            np.abs(height1_samples - height2_samples), abs_diff_samples
        )

    def test_fault_controlled_owc_correlation(self):
        """
        Test that oil-water contact (OWC) correlation between segments
        depends on fault state in geological modeling.

        When fault is open (leaking): Seg2 should have same contact as Seg1
        When fault is closed: Seg2 should follow independent distribution (1950-2000m)
        """
        # Setup
        rng = np.random.default_rng(42)
        n_samples = 100

        # Seg1: OWC = 2000 +/- 5 m (observed segment)
        owc1 = Distribution("uniform", loc=1995, scale=10)

        # Fault state: 30% probability of being open (leaking)
        fault_is_open = Distribution("bernoulli", p=0.3)

        # Seg2: Conditional OWC based on fault state
        # If fault open: same as Seg1
        # If fault closed: independent uniform distribution 1950-2000m
        owc2 = fault_is_open * owc1 + (1 - fault_is_open) * Distribution(
            "uniform", loc=1950, scale=50
        )

        # Generate samples
        (owc1_samples, fault_samples, owc2_samples) = sample(
            [owc1, fault_is_open, owc2], n_samples, random_state=rng
        )

        # Verify fault-controlled correlation
        for i in range(n_samples):
            if fault_samples[i]:  # Fault is open (leaking)
                assert np.isclose(owc2_samples[i], owc1_samples[i], rtol=1e-10), (
                    f"Sample {i}: When fault is open, Seg2 OWC ({owc2_samples[i]:.2f}) "
                    f"should equal Seg1 OWC ({owc1_samples[i]:.2f})"
                )
            else:  # Fault is closed
                assert 1950 <= owc2_samples[i] <= 2000, (
                    f"Sample {i}: When fault is closed, Seg2 OWC ({owc2_samples[i]:.2f}) "
                    f"should be in independent range [1950-2000m]"
                )

        # Additional statistical checks
        open_fault_count = np.sum(fault_samples)
        closed_fault_count = n_samples - open_fault_count

        # Verify we have reasonable sample sizes for both scenarios
        assert open_fault_count > 0, "Should have some samples with open fault"
        assert closed_fault_count > 0, "Should have some samples with closed fault"

    def test_stopping_distance_problem(self):
        """A car is driving at a speed of 100 km / hour. Suddenly it has to slam
        the breaks. What is the breaking distance of the car?

        The main uncertainty is the speed and the coefficient of friction.
        For fun we can also account for some variation in the gravitational
        constant g, since it depends on altitude and where on earth we are."""

        # The kinetic energy is E = 1/2 m v^2
        # The breaking for has to stop this energy
        # Energy = force x distance
        # 1/2 m v^2 = (mu g m) x distance
        # distance = v^2 / (2 mu g)

        # Around 3% relative error, but on multiplicative scale
        velocity_kmh = Exp(Distribution("norm", 0, 0.03)) * 100
        velocity_ms = velocity_kmh / 3.6  # Convert to meters per second
        g = Distribution("norm", loc=9.8220, scale=0.0020)

        # On dry concrete/asphalt, around 0.7 is reasonable
        # https://en.wikibooks.org/wiki/Physics_Study_Guide/Frictional_coefficients
        mu_dry = Distribution("norm", loc=0.7, scale=0.02)
        distance_dry = velocity_ms**2 / (2 * mu_dry * g)

        # Sample and create a simple regression/snapshot test
        samples = sample(distance_dry, 999, random_state=42, method="lhs")
        np.testing.assert_allclose(np.mean(samples), 55.98118937315399)
        np.testing.assert_allclose(np.std(samples), 3.810850016422358)

        # On wet concrete mu=0.58, but on wet asphalt mu = 0.53.
        # Here we model a 50/50 chance of being on either (mixture distribution)
        # and a sigma of 0.02 either way.
        is_concrete = Distribution("bernoulli", p=0.5)
        mu_wet = Distribution("norm", loc=0.53, scale=0.02) + 0.05 * is_concrete
        distance_wet = velocity_ms**2 / (2 * mu_wet * g)

        # Sample and create a simple regression/snapshot test
        samples = sample(distance_wet, 999, random_state=42, method="lhs")
        print(samples.mean(), samples.std())
        np.testing.assert_allclose(np.mean(samples), 71.32497704280316)
        np.testing.assert_allclose(np.std(samples), 6.025907656905284)


def test_constant_arithmetic():
    # Test that converstion with int works
    two = Constant(2)
    result = two + 2
    np.testing.assert_allclose(sample(result), 4)

    # Test that subtraction works both ways
    two = Constant(2)
    five = Constant(5)
    result1 = five - two
    result2 = 5 - two
    result3 = five - two
    np.testing.assert_allclose(sample(result1), sample(result2))
    np.testing.assert_allclose(sample(result1), sample(result3))
    np.testing.assert_allclose(sample(result1), 5 - 2)

    # Test that divison works both ways
    two = Constant(2)
    five = Constant(5)
    result1 = five / two
    result2 = 5 / two
    result3 = five / two
    np.testing.assert_allclose(sample(result1), sample(result2))
    np.testing.assert_allclose(sample(result1), sample(result3))
    np.testing.assert_allclose(sample(result1), 5 / 2)

    # Test absolute value and negation
    result = abs(-two)
    np.testing.assert_allclose(sample(result), 2)

    # Test powers
    result = five**two
    np.testing.assert_allclose(sample(result), 5**2)


def test_constant_expressions():
    # Test a few longer expressions
    two = Constant(2)
    five = Constant(5)
    result = two + two - five**2 + abs(-five)
    np.testing.assert_allclose(sample(result), 2 + 2 - 5**2 + abs(-5))

    result = two / five - two**3 + Exp(5)
    np.testing.assert_allclose(sample(result), 2 / 5 - 2**3 + np.exp(5))

    result = 1 / five - (Log(5) + Exp(Log(10)))
    np.testing.assert_allclose(sample(result), 1 / 5 - (np.log(5) + 10))


def test_single_expression():
    # A graph with a single node is an edge-case
    samples = sample(Constant(2))
    np.testing.assert_allclose(samples, 2)


def test_constant_idempotent():
    for a in [-1, 0.0, 1.3, 3]:
        assert Constant(Constant(a)).value == Constant(a).value


def test_that_an_empirical_distribution_can_be_a_parameter():
    location = EmpiricalDistribution(data=[1, 2, 3, 3, 3, 3])
    result = Distribution("norm", loc=location, scale=1)
    sample(result**2, 99, random_state=42)


def test_that_distribution_params_with_transforms():
    # Plain old numbers work as arguments without raising any errors
    loc = 2
    samples1 = sample(Distribution("norm", loc=loc), 99, random_state=0)

    # The same number wrapped in constant
    loc = Constant(2)
    samples2 = sample(Distribution("norm", loc=loc), 99, random_state=0)

    # A more complex expression: loc = 0 + sqrt(9) - Log(2) = 0 + 3 - 1 = 2
    loc = Constant(0) + (Constant(9) ** 0.5) - Log(2.718281828459045)
    samples3 = sample(Distribution("norm", loc=loc), 99, random_state=0)

    np.testing.assert_allclose(samples1, samples2)
    np.testing.assert_allclose(samples1, samples3)


def test_correlations():
    # Test that inducing correlations leads to samples with approximately the
    # same observed correlations as the desired correlations
    a = Distribution("norm", loc=0, scale=1)
    b = Distribution("norm", loc=0, scale=1)
    c = Distribution("norm", loc=0, scale=1)
    d = Distribution("norm", loc=0, scale=1)
    expression = a + b + c + d

    # This is not a valid correlation matrix (not pos.def) - but probabilit will fix that
    corr_mat = np.array([[1.0, 0.8, 0.5], [0.8, 1.0, 0.8], [0.5, 0.8, 1.0]])
    expression.correlate(a, b, c, corr_mat=corr_mat)

    samples = sample(expression, 999, random_state=42)

    observed_corr_mat = np.corrcoef(
        np.vstack([a.samples_, b.samples_, c.samples_, d.samples_])
    )
    desired_corr_mat = np.eye(4)
    desired_corr_mat[:3, :3] = corr_mat

    np.testing.assert_allclose(observed_corr_mat, desired_corr_mat, atol=0.075)


def test_all_correlations_at_unity():
    # 1. Test that a correlation matrices with all entries at 1.0 works
    # 2. Test that the resulting observed matrix is very close

    a = Distribution("norm", loc=0, scale=1)
    b = Distribution("norm", loc=0, scale=1)
    c = Distribution("norm", loc=0, scale=1)
    expression = a + b + c

    corr_mat = np.ones((3, 3))
    expression.correlate(a, b, c, corr_mat=corr_mat)
    samples = sample(expression, 999, random_state=42, method="lhs")

    obs_corr = np.corrcoef(np.array([a.samples_, b.samples_, c.samples_]))
    assert np.linalg.norm(obs_corr - corr_mat) <= 0.00015


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
