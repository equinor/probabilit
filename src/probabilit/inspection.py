"""
Inspection
----------

Inspection of results, plotting, tables, exporting, etc.
"""

from __future__ import annotations

from collections.abc import Callable
from numbers import Number
from typing import cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import seaborn as sns

from probabilit.modeling import (
    Node,
    NoOp,
    Transform,
    _Node,
    _SampleKwargs,
    _Samples,
)


def plot(
    *variables: complex | np.number | Number | Node[_Samples],
    corr: npt.ArrayLike | None = None,
    sample_kwargs: _SampleKwargs | None = None,
    **kwargs: object,
) -> sns.PairGrid:
    """Utility function for quick plotting of one or several variables.

    A scalar `corr`, including a zero-dimensional array, requires exactly two
    variables. Otherwise, supply a square array-like correlation matrix.
    Matrix rows and columns must follow the order of `variables`.

    Examples
    --------
    >>> from probabilit import Distribution
    >>> a = Distribution("uniform", loc=0, scale=1)
    >>> b = Distribution("uniform", loc=0, scale=1)
    >>> c = Distribution("uniform", loc=0, scale=1)

    >>> pairgrid = plot(a)
    >>> pairgrid = plot(a, b)
    >>> pairgrid = plot(a, b, corr=0.5)
    >>> pairgrid = plot(a, b, corr=[[1.0, 0.5], [0.5, 1.0]])

    >>> corr = np.eye(3) / 2 + np.ones((3, 3)) / 2
    >>> pairgrid = plot(a, b, c, corr=corr)

    >>> pairgrid = plot(a, sample_kwargs={'size':99})
    """
    # Create an NoOp node, then copy the NoOp (which copies all parents too)
    # This prevents us from mutating the input arguments
    no_operation = NoOp(*variables).copy()
    nodes = no_operation.parents

    corr_mat = None
    if corr is not None:
        corr_mat = np.asarray(corr)
        if corr_mat.ndim == 0:
            if len(nodes) != 2:
                raise ValueError("Scalar `corr` requires exactly two variables.")
            rho = corr_mat.item()
            corr_mat = np.array([[1.0, rho], [rho, 1.0]])

    # Check if variables are already sampled
    sampled = [hasattr(v, "samples_") for v in nodes]

    if any(sampled) and not all(sampled):
        raise ValueError("Either all variables must be sampled, or none.")

    # Sample if not sampled, or any  keyword args are specified
    if not any(sampled) or (corr is not None) or (sample_kwargs is not None):
        # Apply defaults first
        sampling_options: _SampleKwargs = {"size": 999, "random_state": 0}
        if sample_kwargs:
            sampling_options.update(sample_kwargs)

        # Correlate if a correlation is given
        if corr_mat is not None:
            no_operation.correlate(*nodes, corr_mat=corr_mat)

        no_operation.sample(**sampling_options)

    # Transform to dataframe and return plot
    df = pd.DataFrame({f"var_{i}": var.samples_ for (i, var) in enumerate(nodes, 1)})
    # Seaborn owns the forwarded keyword argument contract.
    return cast(Callable[..., sns.PairGrid], sns.pairplot)(df, **kwargs)


def treeprint(node: _Node) -> None:
    """Print a computational graph in a tree-like fashion.

    Examples
    --------
    >>> from probabilit import Distribution
    >>> scale = Distribution("expon")
    >>> a = Distribution("norm", loc=1, scale=scale)
    >>> treeprint(a + scale - scale**2)
    Subtract
       ├──Add
       │  ├──Distribution("norm", loc=1, scale=Distribution("expon"))
       │  │  └──Distribution("expon")
       │  └──Distribution("expon")
       └──Power
          ├──Distribution("expon")
          └──Constant(2)

    """
    elbow, pipe, tee, blank = "└──", "│  ", "├──", "   "

    def _treeprint(
        node: _Node, last: bool = True, header: str = "", root: bool = False
    ) -> None:
        # Recursive version
        output = type(node).__name__ if isinstance(node, Transform) else str(node)
        print(header + ("" if root else (elbow if last else tee)) + output)

        if parents := list(node.get_parents()):
            for i, parent in enumerate(parents):
                _treeprint(
                    parent,
                    header=header + (blank if last else pipe),
                    last=i == len(parents) - 1,
                )

    _treeprint(node, last=True, header="", root=True)
