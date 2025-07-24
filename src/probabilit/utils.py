import itertools
import numpy as np


def zip_args(args, kwargs):
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


def build_corrmat(correlations):
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
