from functools import singledispatch

import pytensor.tensor as pt
from pytensor.tensor.random.basic import BernoulliRV, NormalRV, TriangularRV, UniformRV


@singledispatch
def quantile(rv_op, draws, *dist_params):
    raise NotImplementedError(f"quantile not implemented for {rv_op}")


@quantile.register(NormalRV)
def normal_quantile(rv_op, draws, loc, scale):
    return loc + scale * pt.sqrt(2) * pt.erfinv(2 * draws - 1)


@quantile.register(UniformRV)
def uniform_quantile(rv_op, draws, lower, upper):
    return lower + (upper - lower) * draws


@quantile.register(BernoulliRV)
def bernoulli_quantile(rv_op, draws, p):
    return pt.switch(draws > p, 1, 0)


@quantile.register(TriangularRV)
def triangular_quantile(rv_op, draws, lower, upper, c):
    tri_range = upper - lower
    breakpoint = (c - lower) / tri_range
    return pt.switch(
        draws > breakpoint,
        lower + pt.sqrt(draws * range * (c - lower)),
        upper - pt.sqrt((1 - draws) * range * (upper - c)),
    )
