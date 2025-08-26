"""
Inspection
----------

Inspection of results, plotting, tables, exporting, etc.
"""

import seaborn
import pandas as pd
from probabilit.modeling import NoOp, Distribution
import numpy as np
from numbers import Number


def plot(*variables, corr=None, sample_kwargs=None, **kwargs):
    """Utility function for quick plotting of one or several variables."""
    if len(variables) == 2 and isinstance(corr, Number):
        corr = np.array([[1.0, corr], [corr, 1.0]])

    if sample_kwargs is None:
        sample_kwargs = dict()

    # Apply defaults first
    sample_kwargs = {"size": 999, "random_state": 0} | sample_kwargs

    # Create an NoOp node, then copy the NoOp (which copies all parents too)
    # This prevents us from mutating the input arguments
    no_operation = NoOp(*variables).copy()
    variables = no_operation.parents  # Get reference back to the variables

    # Correlate if a correlation is given
    if corr is not None:
        no_operation.correlate(*variables, corr_mat=corr)

    no_operation.sample(**sample_kwargs)

    # Transform to dataframe and return plot
    df = pd.DataFrame(
        {f"var_{i}": var.samples_ for (i, var) in enumerate(variables, 1)}
    )
    return seaborn.pairplot(df, **kwargs)


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
