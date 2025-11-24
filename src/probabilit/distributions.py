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
>>> Uniform(minimum=-1, maximum=1)
Distribution("uniform", loc=-1, scale=2)

Some functions are included here to let us define a distribution using
e.g. P10 and P90 instead of minimum and maximum. This is useful when an
expert produces a tuple (low, mode, high) and we wish to fit to it:

>>> price = PERT(low=100, mode=150, high=200, low_perc=0.1, high_perc=0.9)

The PERT function will optimize for the minimum and maximum of the support of
the PERT distribution, then convert it to a Beta parametrization for scipy:

>>> loc, scale = price.to_scipy().kwds["loc"], price.to_scipy().kwds["scale"]
>>> (loc, loc + scale) #  Minimum and maximum of distribution
(51.32..., 248.67...)
"""

import numpy as np
import warnings
import scipy as sp
from probabilit.modeling import Distribution, Log, Exp, Sign


def Uniform(minimum=0, maximum=1):
    """Uniform distribution on [minimum, maximum)."""
    return Distribution("uniform", loc=minimum, scale=maximum - minimum)


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


class Lognormal(Distribution):
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
        super().__init__(distr="lognorm", s=sigma, scale=Exp(mu))

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


def PERT(low, mode, high, *, low_perc=0.0, high_perc=1.0, gamma=4.0):
    """Returns a Beta distribution, parameterized by the PERT parameters.
    Finds an optimal parametrization given (low, mode, high) and
    returns Distribution("beta", a=..., b=..., loc=..., scale=...).
    A high gamma value means a more concentrated distribution.

    Examples
    --------
    >>> PERT(2,5,7,low_perc=0.1, high_perc=0.9)
    Distribution("beta", a=3.50..., b=2.49..., loc=-1.29..., scale=10.04...)
    >>> PERT(0, 6, 10)
    Distribution("beta", a=3.4, b=2.6, loc=0, scale=10)
    """
    # Based on Wikipedia and another implementation:
    # https://en.wikipedia.org/wiki/PERT_distribution
    if not ((0 <= low_perc <= 1.0) and (0 <= high_perc <= 1.0)):
        raise ValueError("Percentiles must be between 0 and 1.")

    if high_perc <= low_perc:
        raise ValueError(f"Must have {high_perc=} > {low_perc=}")

    if np.isclose(low_perc, 0.0) and np.isclose(high_perc, 1.0):
        minimum, maximum = low, high
    else:
        # Estimate min and max from low and high
        minimum, maximum = _fit_pert_distribution(
            low,
            mode,
            high,
            low_perc=low_perc,
            high_perc=high_perc,
            gamma=gamma,
        )
    a, b, loc, scale = _pert_to_beta(
        minimum=minimum, mode=mode, maximum=maximum, gamma=gamma
    )
    return Distribution("beta", a=a, b=b, loc=loc, scale=scale)


def Triangular(low, mode, high, *, low_perc=0.0, high_perc=1.0):
    """Find optimal scipy parametrization given (low, mode, high) and
    return Distribution("triang", loc=..., scale=..., c=...).

    This distribution does *not* work with composite distributions.
    The arguments must be numbers, they cannot be other distributions.

    Examples
    --------
    >>> Triangular(low=1, mode=5, high=9)
    Distribution("triang", loc=1, scale=8, c=0.5)
    >>> Triangular(low=1, mode=5, high=9, low_perc=0.25, high_perc=0.75)
    Distribution("triang", loc=-8.65..., scale=27.31..., c=0.50...)
    """
    # A few comments on fitting can be found here:
    # https://docs.analytica.com/index.php/Triangular10_50_90

    if not ((0 <= low_perc <= 1.0) and (0 <= high_perc <= 1.0)):
        raise ValueError("Percentiles must be between 0 and 1.")

    if high_perc <= low_perc:
        raise ValueError("Must have {high_perc=} > {low_perc=}")

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


def _fit_triangular_distribution(low, mode, high, *, low_perc=0.10, high_perc=0.90):
    """Returns a tuple (loc, scale, c) to be used with scipy.

    Description
    -----------

    Suppose we have a triangular distribution defined on [minimum, maximum]
    with a mode. If we want to compute a percentile, such as P90, we can do it:

    >>> sp.stats.triang(loc=0, scale=10, c=0.5).ppf(0.9)
    np.float64(7.76393202250021)

    This function answers the reverse question. Suppose we are given minimum=0,
    mode=5 and we know that P90=7.763932. What is the maximum value?

    >>> loc, scale, c = _fit_triangular_distribution(low=0, mode=5, high=7.763932,
    ...                              low_perc=0, high_perc=0.90)
    >>> maximum = loc + scale
    >>> maximum
    9.999999...

    In general the user may ask for minimum and maximum given any PXX and PYY.

    Examples
    --------
    >>> _fit_triangular_distribution(3, 8, 10, low_perc=0.10, high_perc=0.90)
    (-0.207..., 12.53..., 0.65...)
    >>> _fit_triangular_distribution(3, 8, 10, low_perc=0.4, high_perc=0.6)
    (-27.63..., 65.82..., 0.54...)
    >>> _fit_triangular_distribution(3, 8, 10, low_perc=0, high_perc=1.0)
    (3.00..., 6.99..., 0.71...)
    """

    # Scale the problem with f(x) = x * a + b, to (-1, 1).
    # This makes all following optimization scale and shift invariant
    a = 2 / (high - low)
    b = 1 - (2 * high) / (high - low)

    def scaler(x):
        return a * x + b

    def inv_scaler(y):
        return (y - b) / a

    low, mode, high = scaler(low), scaler(mode), scaler(high)

    def rmse_minimum_maximum(parameters):
        """Given (minimum, maximum) of a distribution, create the distribution,
        evaluate the inverse-CDF (PPF) and see how close (low, high) is to the
        desired values of (low, high).

        We parametrize this function by (under_mode, over_mode) because the
        constraints under_mode > 0 and over_mode > 0 implies that
          minimum < mode < maximum
        those two box constraints (non-negativity on each variable) are easier
        to deal with for most optimizers, compared to (minimum < maximum).
        """

        # Parameterize as differences relative to the mode, so we obey
        # the constraint: minimum < mode < maximum
        under_mode, over_mode = parameters
        assert under_mode > 0 and over_mode > 0

        # Convert to minimum and maximum
        minimum, maximum = mode - under_mode, mode + over_mode

        # Create distribution
        loc = minimum
        scale = maximum - minimum
        c = (mode - minimum) / scale
        distr = sp.stats.triang(loc=loc, scale=scale, c=c)

        # Check how close we are to the desired low and high
        est_low, est_high = distr.ppf([low_perc, high_perc])
        residuals = np.array([low - est_low, high - est_high])
        return np.sqrt(np.mean(residuals**2))

    # Initial guesses for a and b, the lower and upper bounds for support
    # The shift and scale are empirical: use what makes tests pass.
    # We also ensure that initial guesses are positive
    under_mode0 = max((mode - low) * 0.5, 0) + 0.01
    over_mode0 = max((high - mode) * 0.5, 0) + 0.01

    # Small number close to zero for optimization bounds
    epsilon = np.finfo(float).eps ** 0.5

    result = sp.optimize.minimize(
        rmse_minimum_maximum,
        x0=[under_mode0, over_mode0],
        bounds=[(epsilon, np.inf), (epsilon, np.inf)],
        method="L-BFGS-B",  # Empirical: we chose this method since tests pass
    )

    if result.fun > 1e-6:
        warnings.warn(f"Optimization of Triangular params has {result.fun=}")

    # Extract the minimum and maximum of the distribution
    under_mode, over_mode = result.x
    minimum, maximum = mode - under_mode, mode + over_mode

    # We scale to (-1, 1) in the beginning, and now we must scale back
    minimum, mode, maximum = inv_scaler(minimum), inv_scaler(mode), inv_scaler(maximum)

    # Back to scipy parametrization
    loc = minimum
    scale = maximum - minimum
    c = (mode - minimum) / scale
    return float((loc)), float((scale)), float(c)


def _pert_to_beta(minimum, mode, maximum, *, gamma=4.0):
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
    # https://github.com/Calvinxc1/PertDist/blob/6577394265f57153441b5908147d94115b9edeed/pert/pert.py#L80
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

    return a, b, loc, scale


def _fit_pert_distribution(low, mode, high, *, low_perc=0.10, high_perc=0.90, gamma=4):
    """
    Returns the maximum and the minimum of a PERT distribution with
    percentiles corresponding to the inputs.

    Examples
    --------
    >>> _fit_pert_distribution(1, 5, 7, low_perc=0, high_perc=1)
    (1.00..., 6.99...)
    >>> _fit_pert_distribution(2, 5, 7, low_perc=0.1, high_perc=0.9)
    (-1.29..., 8.74...)
    """
    # Scale the problem with f(x) = x * a + b, to (-1, 1).
    a = 2 / (high - low)
    b = 1 - (2 * high) / (high - low)

    def scaler(x):
        return a * x + b

    def inv_scaler(y):
        return (y - b) / a

    low, mode, high = scaler(low), scaler(mode), scaler(high)

    def rmse_minimum_maximum(parameters):
        """Given (minimum, maximum) of a distribution, create the distribution,
        evaluate the inverse-CDF (PPF) and see how close (low, high) is to the
        desired values of (low, high).

        We parametrize this function by (under_mode, over_mode) because the
        constraints under_mode > 0 and over_mode > 0 implies that
          minimum < mode < maximum
        those two box constraints (non-negativity on each variable) are easier
        to deal with for most optimizers, compared to (minimum < maximum).
        """

        # Parameterize as differences relative to the mode, so we obey
        # the constraint: minimum < mode < maximum
        under_mode, over_mode = parameters

        # Convert to minimum and maximum
        minimum, maximum = mode - under_mode, mode + over_mode

        # Create corresponding beta distribution
        a, b, loc, scale = _pert_to_beta(
            minimum=minimum, mode=mode, maximum=maximum, gamma=gamma
        )
        distr = sp.stats.beta(a, b, loc=loc, scale=scale)

        # Check how close we are to the desired low and high
        est_low, est_high = distr.ppf([low_perc, high_perc])
        residuals = np.array([low - est_low, high - est_high])
        return np.sqrt(np.mean(residuals**2))

    # Initial guesses for a and b, the lower and upper bounds for support
    # The shift and scale are empirical: use what makes tests pass.
    # We also ensure that initial guesses are positive
    under_mode0 = max((mode - low) * 0.5, 0) + 0.01
    over_mode0 = max((high - mode) * 0.5, 0) + 0.01

    # Small number close to zero for optimization bounds
    epsilon = np.finfo(float).eps ** 0.5
    result = sp.optimize.minimize(
        rmse_minimum_maximum,
        x0=[under_mode0, over_mode0],
        bounds=[(epsilon, np.inf), (epsilon, np.inf)],
        method="L-BFGS-B",  # Empirical: we chose this method since tests pass
    )

    if result.fun > 1e-6:
        warnings.warn(f"Optimization of PERT params has {result.fun=}")

    # Extract the minimum and maximum of the distribution
    under_mode, over_mode = result.x
    minimum, maximum = mode - under_mode, mode + over_mode

    # We scale to (-1, 1) in the beginning, and now we must scale back
    minimum, maximum = inv_scaler(minimum), inv_scaler(maximum)
    return float(minimum), float(maximum)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
