"""
Inspection
----------

Inspection of results, plotting, tables, exporting, etc.
"""

import seaborn
import pandas as pd
from probabilit.modeling import NoOp, Distribution
import numpy as np


def plot(*variables, corr=None, **kwargs):
    """Utility function for quick plotting of one or several variables."""
    has_samples = [hasattr(var, "samples_") for var in variables]
    if not (all(has_samples) or not any(has_samples)):
        raise ValueError("Either all variables must be sampled or none must be.")

    if not any(has_samples):
        # Create an NoOp node, then copy the NoOp (which copies all parents too)
        # This prevents us from mutating the input arguments
        no_operation = NoOp(*variables).copy()
        variables = no_operation.parents  # Get reference back to the variables
        no_operation.sample(size=999, random_state=42)  # Sample to populate _samples

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
