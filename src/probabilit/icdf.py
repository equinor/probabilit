from collections.abc import Buffer
from functools import singledispatch
from pytensor.tensor.random.basic import NormalRV, BernoulliRV, UniformRV, TriangularRV
import pytensor.tensor as pt


@singledispatch
def icdf(rv_op, draws, *dist_params):
    raise NotImplementedError(f"icdf not implemented for {rv_op}")


@icdf.register(NormalRV)
def normal_icdf(rv_op, draws, loc, scale):
    return loc + scale * pt.sqrt(2) * pt.erfinv(2 * draws - 1)


@icdf.register(UniformRV)
def uniform_icdf(rv_op, draws, lower, upper):
    return lower + (upper - lower) * draws


@icdf.register(BernoulliRV)
def bernoulli_icdf(rv_op, draws, p):
    return pt.switch(draws > p, 1, 0)


@icdf.register(TriangularRV)
def triangular_icdf(rv_op, draws, lower, upper, c):
    tri_range = upper - lower
    breakpoint = (c - lower) / tri_range
    return pt.switch(
        draws > breakpoint,
        lower + pt.sqrt(draws * range * (c - lower)),
        upper - pt.sqrt((1 - draws) * range * (upper - c)),
    )
