from probabilit.modeling import Constant, Log, Exp, Distribution, Floor, Equal
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


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
