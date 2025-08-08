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
                low=3.75, mode=4.25, high=5.5, low_perc=0.00001, high_perc=0.99999
            )

        num_samples = 10000
        res_total_person_hours = total_person_hours.sample(num_samples, rng)

        # The mean and standard deviation of a Triangular(3.75, 4.25, 5.5) are 4.5 and 0.368,
        # so by the Central Limit Theoreom we have that
        # total_person_hours = Normal(4.5 * 562, 0.368 * sqrt(562)) = Normal(2529, 8.724)
        expected_mean = 4.5 * num_rivets
        expected_std = 0.368 * np.sqrt(num_rivets)

        sample_mean = np.mean(res_total_person_hours)
        sample_std = np.std(res_total_person_hours)

        assert abs(sample_mean - expected_mean) < 0.3
        assert abs(sample_std - expected_std) < 0.1


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


def test_empirical_distribution():
    # Test that an empirical distribution can be a parameter
    location = EmpiricalDistribution(data=[1, 2, 3, 3, 3, 3])
    result = Distribution("norm", loc=location, scale=1)
    (result**2).sample(99, random_state=42)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
