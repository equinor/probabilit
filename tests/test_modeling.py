from probabilit.modeling import Constant, Log, Exp, Distribution
import numpy as np


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
