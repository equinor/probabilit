# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 10:57:25 2025

@author: TODL
"""

from probabilit.modeling import Distribution, Constant, Log


def test_garbage_collector():
    loc = Constant(0)
    scale = Constant(1)
    a = Distribution("norm", loc=loc, scale=scale)
    b = Distribution("norm", loc=loc, scale=scale)
    the_sum = a + b
    the_power = the_sum**2
    the_result = Log(1 + the_power)

    # No garbage collection => all nodes have samples_
    the_result.sample(99, random_state=42, gc_strategy=None)
    assert hasattr(loc, "samples_")
    assert hasattr(scale, "samples_")
    assert hasattr(a, "samples_")
    assert hasattr(b, "samples_")
    assert hasattr(the_sum, "samples_")
    assert hasattr(the_power, "samples_")
    assert hasattr(the_result, "samples_")

    # Full garbage collection => only the result has samples_
    the_result.sample(99, random_state=42, gc_strategy=[])
    assert not hasattr(loc, "samples_")
    assert not hasattr(scale, "samples_")
    assert not hasattr(a, "samples_")
    assert not hasattr(b, "samples_")
    assert not hasattr(the_sum, "samples_")
    assert not hasattr(the_power, "samples_")
    assert hasattr(the_result, "samples_")

    # Partial garbage collection => selected nodes have samples_
    the_result.sample(99, random_state=42, gc_strategy=[loc, scale])
    assert hasattr(loc, "samples_")
    assert hasattr(scale, "samples_")
    assert not hasattr(a, "samples_")
    assert not hasattr(b, "samples_")
    assert not hasattr(the_sum, "samples_")
    assert not hasattr(the_power, "samples_")
    assert hasattr(the_result, "samples_")


def test_garbage_collector_large_graph():
    sampling_nodes = []

    result = 0
    for year in range(99):
        addition = Distribution("norm", loc=100, scale=10)
        interest_rate = Distribution("norm", loc=1.05, scale=0.05)
        result = result * interest_rate + addition

        # Store nodes to check them later
        sampling_nodes.append(addition)
        sampling_nodes.append(interest_rate)

    result.sample(99, random_state=42, gc_strategy=[])

    # Verify that GC worked
    assert hasattr(result, "samples_")
    assert not any(hasattr(n, "samples_") for n in sampling_nodes)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "-v", "--capture=sys"])
