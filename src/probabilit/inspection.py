"""
Inspection
----------

Inspection of results, plotting, tables, exporting, etc.
"""

from numbers import Number
from typing import Any

import numpy as np
import pandas as pd
import seaborn
from seaborn.axisgrid import PairGrid

from probabilit.modeling import Distribution, Node, NoOp, Transform
from probabilit.types import Array2D


def plot(
    *variables: Distribution,
    corr: Array2D | None = None,
    sample_kwargs: Any = None,
    **kwargs: Any,
) -> PairGrid:
    """Utility function for quick plotting of one or several variables.

    Examples
    --------
    >>> a = Distribution("uniform", loc=0, scale=1)
    >>> b = Distribution("uniform", loc=0, scale=1)
    >>> c = Distribution("uniform", loc=0, scale=1)

    >>> pairgrid = plot(a)
    >>> pairgrid = plot(a, b)
    >>> pairgrid = plot(a, b, corr=0.5)

    >>> corr = np.eye(3) / 2 + np.ones((3, 3)) / 2
    >>> pairgrid = plot(a, b, c, corr=corr)

    >>> pairgrid = plot(a, sample_kwargs={'size':99})
    """
    # Create an NoOp node, then copy the NoOp (which copies all parents too)
    # This prevents us from mutating the input arguments
    no_operation = NoOp(*variables).copy()
    variables = no_operation.parents

    if len(variables) == 2 and isinstance(corr, Number):
        corr = np.array([[1.0, corr], [corr, 1.0]])

    # Check if variables are already sampled
    sampled = [hasattr(v, "samples_") for v in variables]

    if any(sampled) and not all(sampled):
        raise ValueError("Either all variables must be sampled, or none.")

    # Sample if not sampled, or any  keyword args are specified
    if not any(sampled) or (corr is not None) or (sample_kwargs is not None):
        # Apply defaults first
        sample_kwargs = {"size": 999, "random_state": 0} | (sample_kwargs or {})

        # Correlate if a correlation is given
        if corr is not None:
            no_operation.correlate(*variables, corr_mat=corr)

        no_operation.sample(**sample_kwargs)

    # Transform to dataframe and return plot
    df = pd.DataFrame(
        {f"var_{i}": var.samples_ for (i, var) in enumerate(variables, 1)}
    )
    return seaborn.pairplot(df, **kwargs)


def treeprint(node: Node) -> None:
    """Print a computational graph in a tree-like fashion.

    Examples
    --------
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
        node: Node,
        last: bool = True,
        header: str = "",
        root: bool = False,
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

    return _treeprint(node, last=True, header="", root=True)


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    # mu = Constant(1)
    a = Distribution("norm", loc=0, scale=1)
    b = Distribution("norm", loc=a, scale=0.5)

    plot(a, b)
    grid = plot(a)

    from probabilit.modeling import MultivariateDistribution

    cov = np.array([[1, 0.9], [0.9, 1]])
    n1, n2 = MultivariateDistribution("multivariate_normal", mean=[1, 2], cov=cov)
    from probabilit.inspection import plot

    plot(n1, n2)
