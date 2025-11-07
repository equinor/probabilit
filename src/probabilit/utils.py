import itertools
from collections.abc import Iterable, Iterator, Mapping
from typing import Any

import numpy as np
import numpy.typing as npt
import scipy as sp

from probabilit.types import Array2D


def adjust_minmax_quantiles(
    quantiles: Iterable[float],
    cumulatives: Iterable[float],
    expected: float,
) -> Array2D:
    """Adjust minimum and maximum in `quantiles` so we hit expected value.

    Examples
    --------
    Data for a distribution with mean (2.5 + 5.5) / 2 = 4 leads to no change:
    >>> adjust_minmax_quantiles([0, 0.5, 1], [0, 5, 6], expected=4.0)
    array([0., 5., 6.])

    Increasing the expected value to 5:
    >>> adjust_minmax_quantiles([0, 0.5, 1], [0, 5, 6], expected=5)
    array([2.00014464, 5.        , 7.99985536])

    >>> adjust_minmax_quantiles([0, 0.1, 0.3, 1], [0, 1, 1.5, 2], 1.6)
    array([0.03013972, 1.        , 1.5       , 2.20998004])
    """
    quantiles = np.array(quantiles, dtype=float)
    cumulatives = np.array(cumulatives, dtype=float)
    assert np.all(np.diff(quantiles) > 0)
    assert np.all(np.diff(cumulatives) > 0)
    assert np.isclose(np.min(quantiles), 0)
    assert np.isclose(np.max(quantiles), 1)

    def empirical_mean(
        quantiles: npt.NDArray[Any],
        cumulatives: npt.NDArray[Any],
    ) -> float:
        """Compute the expected value of a histogram."""
        return sp.stats.rv_histogram(
            (np.diff(quantiles), cumulatives), density=False
        ).mean()

    def transform(
        low_scale: float,
        high_scale: float,
        cumulatives: npt.NDArray[Any],
    ) -> tuple[float, float]:
        """Return new low and high values in the cumulatives."""
        cumulatives = cumulatives.copy()
        q1, q2 = cumulatives[:2]
        qn1, qn = cumulatives[-2:]
        high = max(qn1 + np.exp(high_scale) * (qn - qn1), qn1 + 1e-6)
        low = min(q2 - np.exp(low_scale) * (q2 - q1), q2 - 1e-6)
        return (low, high)

    def objective(
        params: tuple[float, float],
        quantiles: npt.NDArray[Any],
        cumulatives: npt.NDArray[Any],
        expected: float,
    ) -> float:
        """Objective function to minimize."""
        low_scale, high_scale = params

        # Transform the endpoints of the cumulatives
        (low, high) = transform(low_scale, high_scale, cumulatives)
        cumulatives_copy = np.array(cumulatives)
        cumulatives_copy[0], cumulatives_copy[-1] = low, high

        observed_mean = empirical_mean(quantiles, cumulatives_copy)

        # Create objective and return
        main_obj = np.abs(observed_mean - expected)
        bi_obj = (low - cumulatives[0]) ** 2 + (high - cumulatives[-1]) ** 2
        return main_obj + 1e-2 * bi_obj

    result = sp.optimize.minimize(
        fun=objective,
        args=(quantiles, cumulatives, expected),
        x0=[0, 0],
        method="nelder-mead",
    )
    low_scale, high_scale = result.x
    (low, high) = transform(low_scale, high_scale, cumulatives)
    cumulatives[0], cumulatives[-1] = low, high

    return cumulatives


def zip_args(
    args: Iterable[Any],
    kwargs: Mapping[str, Any],
) -> Iterator[tuple[Any, dict[str, Any]]]:
    """Zip argument and keyword arguments for repeated function calls.

    Examples
    --------
    >>> args = ((1, 2, 3), itertools.repeat(None))
    >>> kwargs = {"a": (5, 6, 7), "b": itertools.repeat(9)}
    >>> for args_i, kwargs_i in zip_args(args, kwargs):
    ...     print(args_i, kwargs_i)
    (1, None) {'a': 5, 'b': 9}
    (2, None) {'a': 6, 'b': 9}
    (3, None) {'a': 7, 'b': 9}
    """
    zipped_args = zip(*args) if args else itertools.repeat(args)
    zipped_kwargs = zip(*kwargs.values()) if kwargs else itertools.repeat(kwargs)

    for args_i, kwargs_i in zip(zipped_args, zipped_kwargs):
        yield args_i, dict(zip(kwargs.keys(), kwargs_i))


def build_corrmat(correlations: Iterable[tuple[tuple[int, ...], Array2D]]) -> Array2D:
    """Given a list of [(indices1, corrmat1), (indices2, corrmat2), ...],
    create a big correlation matrix.

    Examples
    --------
    >>> correlations = [((0, 2), np.array([[1, 0.5], [0.5, 1]]))]
    >>> build_corrmat(correlations)
    array([[1. , 0. , 0.5],
           [0. , 1. , 0. ],
           [0.5, 0. , 1. ]])
    """
    # TODO: If no correlation is given, we implicitly assume zero.
    # For instance, if no correlation between indices (0, 3) is given
    # in the input data, then C[0, 3] = C[3, 0] = 0.0, which is strictly
    # speaking not the same (no preference vs. preference for 0 corr)
    n = max(max(idx) for (idx, _) in correlations)
    C = np.eye(n + 1, dtype=float)

    for idx_i, corrmat_i in correlations:
        C[np.ix_(idx_i, idx_i)] = corrmat_i

    return C


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
