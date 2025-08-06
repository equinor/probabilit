import scipy as sp
import numpy as np
import warnings
from probabilit.modeling import Distribution


def Normal(loc, scale):
    return Distribution("norm", loc=loc, scale=scale)


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
    """Find optimal scipy parametrization given (p10, mode, p90) and
    return Distribution("triang", loc=..., scale=..., c=...).

    This distribution does *not* work with composite distributions.
    The arguments must be numbers, they cannot be other distributions.

    Examples
    --------
    >>> Triangular(low=1, mode=5, high=9)
    Distribution("triang", loc=-2.236068061140598, scale=14.472136057963969, c=0.5000000024295282)
    """
    # A few comments on fitting can be found here:
    # https://docs.analytica.com/index.php/Triangular10_50_90

    if not (low < mode < high):
        raise ValueError(f"Must have {low=} < {mode=} < {high=}")
    if not ((0 <= low_perc <= 1.0) and (0 <= high_perc <= 1.0)):
        raise ValueError("Percentiles must be between 0 and 1.")

    # Optimize parameters
    loc, scale, c = _triang_params_from_perc(low, mode, high, low_perc, high_perc)
    return Distribution("triang", loc=loc, scale=scale, c=c)


def _triang_params_from_perc(low, mode, high, low_perc=0.1, high_perc=0.9):
    """Given (p10, mode, p90), finds (shift, scale, c).

    Examples
    --------
    >>> from scipy.stats import triang
    >>> import math
    >>> dist = triang(loc=-5, scale=13, c=0.85)
    >>> loc, scale, c = _triang_params_from_perc(*_triang_extract(dist))
    >>> math.isclose(loc, -5, rel_tol=0.001)
    True
    >>> math.isclose(scale, 13, rel_tol=0.001)
    True
    >>> math.isclose(c, 0.85, rel_tol=0.001)
    True
    """
    assert low < mode < high

    # Shift and scale inputs before solving optimization problem
    spread = high - low
    center = (high + low) / 2
    low = (low - center) / spread
    mode = (mode - center) / spread
    high = (high - center) / spread

    # Given (p10, mode, p90) we need to find a scipy parametrization
    # in terms of (loc, scale, c). This cannot be solved analytically.
    desired = np.array([low, mode, high])

    # Initial guess
    loc_initial = low
    scale_initial = np.log(high - low)
    c_initial = sp.special.logit((mode - low) / (high - low))
    x0 = np.array([loc_initial, scale_initial, c_initial])

    # Optimize
    result = sp.optimize.minimize(
        _triang_objective, x0=x0, args=(desired,), method="BFGS"
    )

    assert result.fun < 1e-2
    # Issues can arise. Determining this beforehand
    # is hard, so we simply try to optimize and see if we get close.
    if result.fun > 1e-4:
        warnings.warn(f"Optimization of triangular params did not converge:\n{result}")

    # Extract parameters
    loc_opt = result.x[0]
    scale_opt = np.exp(result.x[1])
    c_opt = sp.special.expit(result.x[2])

    # Shift and scale problem back
    loc_opt = loc_opt * spread + center
    scale_opt = scale_opt * spread

    return float(loc_opt), float(scale_opt), float(c_opt)


def _triang_extract(triangular, low_perc=0.1, high_perc=0.9):
    """Given a triangular distribution, extract (p10, mode, p90).

    Examples
    --------
    >>> from scipy.stats import triang
    >>> dist = triang(loc=-5, scale=13, c=0.6)
    >>> p10, mode, p90 = _triang_extract(dist)
    >>> mode
    2.8
    >>> p90
    5.4
    """
    p10, p90 = triangular.ppf([low_perc, high_perc])
    loc = triangular.kwds.get("loc", 0)
    scale = triangular.kwds.get("scale", 1)
    c = triangular.kwds.get("c", 0.5)
    mode = loc + scale * c

    return float(p10), float(mode), float(p90)


def _triang_objective(parameters, desired):
    """Pass parameters (loc, log(scale), logit(c)) into sp.stats.triang
    and return the RMSE between actual and desired (p10, mode, p90)."""

    loc, scale, c = parameters
    scale = np.exp(scale)  # Scale must be positive
    c = np.clip(sp.special.expit(c), 0, 1)  # C must be between 0 and 1

    # Create distribution
    triangular = sp.stats.triang(loc=loc, scale=scale, c=c)

    # Extract information
    p10, mode, p90 = _triang_extract(triangular)
    actual = np.array([p10, mode, p90])

    if not np.isfinite(actual).all():
        return 1e3

    # RMSE
    return np.sqrt(np.sum((desired - actual) ** 2)) / scale


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
