"""
Probabilit uses distributions implemented in SciPy by default, e.g.:

>>> normal = Distribution("norm", loc=0, scale=1)
>>> gamma = Distribution("gamma", a=1)
>>> generalized_pareto = Distribution("genpareto", c=2)

For a full list, see:

  - https://docs.scipy.org/doc/scipy/reference/stats.html#continuous-distributions

In some cases it might make sense to implement our own custom distributions.
CUSTOM DISTRIBUTIONS SHOULD BE IMPLEMENTED SPARINGLY. WE DO NOT WANT TO GO DOWN
THE PATH OF RE-IMPLEMENTING SCIPY. For better or worse, scipy is a de-facto
standard and many are used to it. Using scipy also delegates documentation burden.

Some reasons to to this:

  1. Better naming and easier API, e.g. `Normal(...)` vs `Distribution("norm", ...)`
  2. Alternative parametrizations (typical example is the LogNorm)
  3. Distributions not found in SciPy

Most of the custom distributions are syntactic sugar:

>>> Distribution("uniform", loc=-1, scale=2)
Distribution("uniform", loc=-1, scale=2)
>>> Uniform(min=-1, max=1)
Distribution("uniform", loc=-1, scale=2)

"""

import warnings

import numpy as np
import pytensor.tensor as pt
import scipy as sp

from probabilit.math import Exp, Log, Sign


def Distribution(name, *args, **kwargs):
    match name.lower():
        case "uniform":
            # Scipy uses loc, scale, but pytensor uses min/max
            def _scipy_uniform(loc=0, scale=1, **kwargs):
                return pt.random.uniform(loc, loc + scale, **kwargs)

            return _scipy_uniform(*args, **kwargs)
        case "normal" | "norm":
            return pt.random.normal(*args, **kwargs)
        case "lognormal":
            return pt.random.lognormal(*args, **kwargs)
        case "exponential":
            return pt.random.exponential(*args, **kwargs)
        case "gamma":
            return pt.random.gamma(*args, **kwargs)
        case "bernoulli":
            return pt.random.bernoulli(*args, **kwargs)
        case "triangular":
            return pt.random.triangular(*args, **kwargs)
        case _:
            raise ValueError(f"Unknown distribution {name}")


def Uniform(min=0, max=1):
    """Uniform distribution on [min, max)."""
    return Distribution("uniform", loc=min, scale=max - min)


def Normal(loc, scale):
    """Normal distribution parametrized by mean (loc) and std (scale)."""
    return Distribution("norm", loc=loc, scale=scale)


def TruncatedNormal(loc, scale, low=-np.inf, high=np.inf):
    """A truncated Normal distribution parametrized by mean (loc) and
    std (scale) defined on [low, high).

    Examples
    --------
    >>> distr = TruncatedNormal(loc=0, scale=1, low=3, high=3.3)
    >>> distr.sample(7, random_state=0).round(3)
    array([3.13 , 3.182, 3.146, 3.129, 3.095, 3.159, 3.099])
    """
    # (a, b) are defined in terms of loc and scale, so transform them
    a, b = (low - loc) / scale, (high - loc) / scale
    return Distribution("truncnorm", a=a, b=b, loc=loc, scale=scale)


class Lognormal:
    def __init__(self, mean, std):
        """
        A Lognormal distribution with mean and std corresponding directly
        to the expected value and standard deviation of the resulting lognormal.

        Examples
        --------
        >>> samples = Lognormal(mean=2, std=1).sample(999, random_state=0)
        >>> float(np.mean(samples))
        2.00173...
        >>> float(np.std(samples))
        1.02675...

        Composite distributions work too:

        >>> mean = Distribution("expon", scale=1)
        >>> Lognormal(mean=mean, std=1).sample(5, random_state=0)
        array([0.86196529, 0.69165866, 0.41782557, 1.23340656, 2.90778578])
        """
        # Transform parameters (they can be numbers, distributions, etc)
        variance = Sign(std) * std**2  # Square it but keep the sign (so negative fails)
        sigma_squared = Log(1 + variance / (mean**2))
        sigma = (sigma_squared) ** (1 / 2)
        mu = Log(mean) - sigma_squared / 2

        # Call the parent class
        Distribution("lognorm", s=sigma, scale=Exp(mu))

    @classmethod
    def from_log_params(cls, mu, sigma):
        """
        Create a lognormal distribution from log-space parameters.
        Parameters correspond to the mean and standard deviation of the
        underlying normal distribution (i.e., the parameters of log(X) where
        X is the lognormal random variable).

        Examples
        --------
        >>> mu = Distribution("norm")
        >>> Lognormal.from_log_params(mu=mu, sigma=1).sample(5, random_state=0)
        array([1.99625633, 1.45244764, 1.19926216, 2.94150961, 4.47459182])
        """
        return Distribution("lognorm", s=sigma, scale=Exp(mu))


def PERT(minimum, mode, maximum, gamma=4.0):
    """Returns a Beta distribution, parameterized by the PERT parameters.

    A high gamma value means a more concentrated distribution.

    Examples
    --------
    >>> PERT(0, 6, 10)
    Distribution("beta", a=3.4, b=2.6, loc=0, scale=10)
    >>> PERT(0, 6, 10, gamma=10)
    Distribution("beta", a=7.0, b=5.0, loc=0, scale=10)
    """
    # Based on Wikipedia and another implementation:
    # https://en.wikipedia.org/wiki/PERT_distribution
    # https://github.com/Calvinxc1/PertDist/blob/6577394265f57153441b5908147d94115b9edeed/pert/pert.py#L80
    a, b, loc, scale = _pert_to_beta(minimum, mode, maximum, gamma=gamma)
    return Distribution("beta", a=a, b=b, loc=loc, scale=scale)


def Triangular(low, mode, high, low_perc=0.1, high_perc=0.9):
    """Find optimal scipy parametrization given (low, mode, high) and
    return Distribution("triang", loc=..., scale=..., c=...).

    This distribution does *not* work with composite distributions.
    The arguments must be numbers, they cannot be other distributions.

    Examples
    --------
    >>> Triangular(low=1, mode=5, high=9)
    Distribution("triang", loc=-2.2360679774997894, scale=14.472135954999578, c=0.5000000000000001)
    >>> Triangular(low=1, mode=5, high=9, low_perc=0.25, high_perc=0.75)
    Distribution("triang", loc=-8.656854249492383, scale=27.313708498984766, c=0.5)
    >>> Triangular(low=1, mode=5, high=9, low_perc=0, high_perc=1)
    Distribution("triang", loc=1, scale=8, c=0.5)
    """
    # A few comments on fitting can be found here:
    # https://docs.analytica.com/index.php/Triangular10_50_90

    if not ((0 <= low_perc <= 1.0) and (0 <= high_perc <= 1.0)):
        raise ValueError("Percentiles must be between 0 and 1.")

    # No need to optimize if low and high are boundaries of distribution support
    if np.isclose(low_perc, 0.0) and np.isclose(high_perc, 1.0):
        loc, scale, c = low, high - low, (mode - low) / (high - low)

    else:
        # Optimize parameters
        loc, scale, c = _fit_triangular_distribution(
            low=low,
            mode=mode,
            high=high,
            low_perc=low_perc,
            high_perc=high_perc,
        )
    return Distribution("triang", loc=loc, scale=scale, c=c)


def _fit_triangular_distribution(low, mode, high, low_perc=0.10, high_perc=0.90):
    """Returns a tuple (loc, scale, c) to be used with scipy.

    Examples
    --------
    >>> _fit_triangular_distribution(3, 8, 10, low_perc=0.10, high_perc=0.90)
    (-0.207..., 12.53..., 0.65...)
    >>> _fit_triangular_distribution(3, 8, 10, low_perc=0.4, high_perc=0.6)
    (-27.63..., 65.82..., 0.54...)
    >>> _fit_triangular_distribution(3, 8, 10, low_perc=0, high_perc=1.0)
    (3.00..., 6.99..., 0.71...)
    """

    def triangular_cdf(x, a, b, mode):
        """Calculate CDF of triangular distribution at point x"""
        if x <= a:
            return x * 0
        if x >= b:
            return x * 0 + 1.0
        if x <= mode:
            return ((x - a) ** 2) / ((b - a) * (mode - a))
        else:
            return 1 - ((b - x) ** 2) / ((b - a) * (b - mode))

    def equations(params):
        """System of equations to solve for a and b"""
        a, b = params

        # Calculate CDFs at the given percentile values
        cdf_low = triangular_cdf(low, a, b, mode)
        cdf_high = triangular_cdf(high, a, b, mode)

        # Return the difference from target percentiles
        return (cdf_low - low_perc, cdf_high - high_perc)

    # Initial guesses for a and b, the lower and upper bounds for support
    a0 = low - abs(mode - low)
    b0 = high + abs(high - mode)

    # Solve the system of equations
    a, b = sp.optimize.fsolve(equations, (a0, b0))
    rmse = np.sqrt(np.sum(np.array(equations([a, b])) ** 2))
    if rmse > 1e-6:
        warnings.warn(f"Optimization of Triangular params has {rmse=}")

    # Calculate the relative position of the mode, return (loc, scale, c)
    c = (mode - a) / (b - a)
    return float(a), float(b - a), float(c)


def _pert_to_beta(minimum, mode, maximum, gamma=4.0):
    """Convert the PERT parametrization to a beta distribution.

    Returns (a, b, loc, scale).

    Examples
    --------
    >>> _pert_to_beta(0, 3/4, 1)
    (4.0, 2.0, 0, 1)
    >>> _pert_to_beta(0, 30/4, 10)
    (4.0, 2.0, 0, 10)
    >>> _pert_to_beta(0, 9, 10, gamma=6)
    (6.4, 1.6, 0, 10)
    """
    # https://en.wikipedia.org/wiki/PERT_distribution
    if not (minimum < mode < maximum):
        raise ValueError(f"Must have {minimum=} < {mode=} < {maximum=}")
    if gamma <= 0:
        raise ValueError(f"Gamma must be positive, got {gamma=}")

    # Determine location and scale
    loc = minimum
    scale = maximum - minimum

    # Determine a and b
    a = 1 + gamma * (mode - minimum) / scale
    b = 1 + gamma * (maximum - mode) / scale

    return (a, b, loc, scale)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
