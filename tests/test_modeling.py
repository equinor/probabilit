from probabilit.modeling import (
    EmpiricalDistribution,
    Constant,
    Log,
    Exp,
    Distribution,
    Floor,
    Equal,
    All,
    Min,
    Max,
)
from probabilit.distributions import Triangular
import numpy as np


class TestModelingExamples:
    def test_die_problem(self):
        """If we throw 2 die, what is the probability that each one ends up
        with the same number?"""

        die1 = Floor(1 + Distribution("uniform") * 6)
        die2 = Floor(1 + Distribution("uniform") * 6)
        equal = Equal(die1, die2)

        samples = equal.sample(999, random_state=42)

        np.testing.assert_allclose(samples.mean(), 1 / 6, atol=0.001)

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

        samples = pi_estimate.sample(9999, random_state=42)
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

        samples = prob.sample(9999, random_state=42)
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
        samples = returns.sample(999, random_state=42)

        # Regression test essentially
        np.testing.assert_allclose(samples.mean(), 76583.58738496085)
        np.testing.assert_allclose(samples.std(), 33483.2245611436)

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
        res_total_person_hours = total_person_hours.sample(num_samples, rng)

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

        # height2 = IF(is_twin, height1, height2)
        height2 = is_twin * height1 + (1 - is_twin) * height2

        # This is the answer to the question
        (abs(height2 - height1)).sample(999, random_state=42)

        # At least one of the realizations should be identical
        assert np.any(np.isclose(height1.samples_, height2.samples_))

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
        owc2_samples = owc2.sample(n_samples, rng)

        # Get individual component samples for verification
        owc1_samples = owc1.samples_
        fault_samples = fault_is_open.samples_.astype(bool)
        owc2_samples = owc2.samples_

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


def test_copying():
    # Create a graph
    mu = Distribution("norm", loc=0, scale=1)
    a = Distribution("norm", loc=mu, scale=Constant(0.5))

    # Create a copy
    a2 = a.copy()

    # The copy is not the same object
    assert a2 is not a

    # However, the IDs match and they are equal
    assert a2 == a and (a2._id == a._id)

    # The same holds for parents - they are copied
    assert a2.kwargs["loc"] is not a.kwargs["loc"]

    a.sample()
    assert hasattr(a, "samples_")
    assert not hasattr(a2, "samples_")

    # Now create a copy and ensure samples are copied too
    a3 = a.copy()
    assert hasattr(a3, "samples_")
    assert a3.samples_ is not a.samples_


def test_constant_arithmetic():
    # Test that converstion with int works
    two = Constant(2)
    result = two + 2
    np.testing.assert_allclose(result.sample(), 4)

    # Test that subtraction works both ways
    two = Constant(2)
    five = Constant(5)
    result1 = five - two
    result2 = 5 - two
    result3 = five - two
    np.testing.assert_allclose(result1.sample(), result2.sample())
    np.testing.assert_allclose(result1.sample(), result2.sample())
    np.testing.assert_allclose(result1.sample(), result3.sample())
    np.testing.assert_allclose(result1.sample(), 5 - 2)

    # Test that divison works both ways
    two = Constant(2)
    five = Constant(5)
    result1 = five / two
    result2 = 5 / two
    result3 = five / two
    np.testing.assert_allclose(result1.sample(), result2.sample())
    np.testing.assert_allclose(result1.sample(), result2.sample())
    np.testing.assert_allclose(result1.sample(), result3.sample())
    np.testing.assert_allclose(result1.sample(), 5 / 2)

    # Test absolute value and negation
    result = abs(-two)
    np.testing.assert_allclose(result.sample(), 2)

    # Test powers
    result = five**two
    np.testing.assert_allclose(result.sample(), 5**2)


def test_constant_expressions():
    # Test a few longer expressions
    two = Constant(2)
    five = Constant(5)
    result = two + two - five**2 + abs(-five)
    np.testing.assert_allclose(result.sample(), 2 + 2 - 5**2 + abs(-5))

    result = two / five - two**3 + Exp(5)
    np.testing.assert_allclose(result.sample(), 2 / 5 - 2**3 + np.exp(5))

    result = 1 / five - (Log(5) + Exp(Log(10)))
    np.testing.assert_allclose(result.sample(), 1 / 5 - (np.log(5) + 10))


def test_single_expression():
    # A graph with a single node is an edge-case
    samples = Constant(2).sample()
    np.testing.assert_allclose(samples, 2)


def test_constant_idempotent():
    for a in [-1, 0.0, 1.3, 3]:
        assert Constant(Constant(a)).value == Constant(a).value


def test_that_an_empirical_distribution_can_be_a_parameter():
    location = EmpiricalDistribution(data=[1, 2, 3, 3, 3, 3])
    result = Distribution("norm", loc=location, scale=1)
    (result**2).sample(99, random_state=42)


def test_that_distribution_params_with_transforms():
    # Plain old numbers work as arguments without raising any errors
    loc = 2
    samples1 = Distribution("norm", loc=loc).sample(99, random_state=0)

    # The same number wrapped in constant
    loc = Constant(2)
    samples2 = Distribution("norm", loc=loc).sample(99, random_state=0)

    # A more complex expression: loc = 0 + sqrt(9) - Log(2) = 0 + 3 - 1 = 2
    loc = Constant(0) + (Constant(9) ** 0.5) - Log(2.718281828459045)
    samples3 = Distribution("norm", loc=loc).sample(99, random_state=0)

    np.testing.assert_allclose(samples1, samples2)
    np.testing.assert_allclose(samples1, samples3)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
