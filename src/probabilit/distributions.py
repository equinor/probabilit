import scipy as sp
import numpy as np
import warnings


def Triangular(p10, mode, p90):
    """Find optimal scipy parametrization given (p10, mode, p90) and
    return Distribution("triang", loc=..., scale=..., c=...)."""
    from probabilit.modeling import Distribution  # Avoid circular imports

    # if not (p10 < mode < p90):
    #    raise ValueError(f"Must have {p10=} < {mode=} < {p90=}")

    # Optimize parameters
    loc, scale, c = _triang_params_from_perc(p10, mode, p90)
    return Distribution("triang", loc=loc, scale=scale, c=c)


def _triang_params_from_perc(p10, mode, p90):
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

    # Shift and scale inputs before solving optimization problem
    spread = p90 - p10
    center = (p90 + p10) / 2
    p10 = (p10 - center) / spread
    mode = (mode - center) / spread
    p90 = (p90 - center) / spread

    # Given (p10, mode, p90) we need to find a scipy parametrization
    # in terms of (loc, scale, c). This cannot be solved analytically.
    desired = np.array([p10, mode, p90])

    # Initial guesses for optimization
    loc_initial = p10
    scale_initial = np.log(p90 - p10)
    c_initial = sp.special.logit((mode - p10) / (p90 - p10))
    x0 = np.array([loc_initial, scale_initial, c_initial])

    # Optimize
    result = sp.optimize.minimize(
        _triang_objective, x0=x0, args=(desired,), method="BFGS"
    )

    # assert result.success
    # Issues can arise if e.g. (p10=-2, mode=2, p90=2), since there is no
    # triangular distributions that match these criteria. In general the mode
    # must be sufficiently between p10 and p90. Determining this beforehand
    # is hard, so we simply try to optimize and see if we get close.
    if result.fun > 1e-2:
        warnings.warn(f"Optimization of triangular params did not converge:\n{result}")

    # Extract parameters
    loc_opt = result.x[0]
    scale_opt = np.exp(result.x[1])
    c_opt = sp.special.expit(result.x[2])

    # Shift and scale problem back
    loc_opt = loc_opt * spread + center
    scale_opt = scale_opt * spread

    return float(loc_opt), float(scale_opt), float(c_opt)


def _triang_extract(triangular):
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
    p10, p90 = triangular.ppf([0.1, 0.9])
    loc = triangular.kwds.get("loc", 0)
    scale = triangular.kwds.get("scale", 1)
    c = triangular.kwds.get("c", 0.5)
    mode = loc + scale * c

    return float(p10), float(mode), float(p90)


def _triang_objective(parameters, desired):
    """Pass parameters (loc, log(scale), logit(scale)) into sp.stats.triang
    and return the RMSE between actual and desired (p10, mode, p90)."""

    loc, scale, c = parameters
    scale = np.exp(scale)  # Scale must be positive
    c = sp.special.expit(c)  # C must be between 0 and 1

    # Create distribution
    triangular = sp.stats.triang(loc=loc, scale=scale, c=c)

    # Extract information
    p10, mode, p90 = _triang_extract(triangular)
    actual = np.array([p10, mode, p90])

    if not np.isfinite(actual).any():
        return 1e3

    # RMSE
    return np.sqrt(np.sum((desired - actual) ** 2))


# ========================================================
if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])

    # 0.3977688370411745 0.12620833534058773 0.8084622980757837
    # 0.6246170152336684 0.7165879070002568 0.9288370744653666

    for _ in range(99):
        print("---------------------")
        loc = np.random.rand()
        scale = np.random.rand()
        c = np.random.rand() * 0.5 + 0.25

        print(loc, scale, c)

        from scipy.stats import triang

        dist = triang(loc=loc, scale=scale, c=c)
        print(_triang_extract(dist))
        opt_loc, opt_scale, opt_c = _triang_params_from_perc(*_triang_extract(dist))

        assert abs(loc - opt_loc) <= 0.01
        assert abs(scale - opt_scale) <= 0.01
        assert abs(c - opt_c) <= 0.01

        print(loc, opt_loc)
        print(scale, opt_scale)
        print(c, opt_c)
