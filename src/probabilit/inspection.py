"""
Inspection
----------

Inspection of results, plotting, tables, exporting, etc.
"""

import seaborn
import pandas as pd
from probabilit.modeling import Add, Distribution
import numpy as np


def plot(*variables, corr=None, **kwargs):
    """Utility function for quick plotting of one or several variables."""
    # TODO: This function can be improved or customized. For now we use seaborn

    # Create an adder node, then copy the adder (which copies all parents too)
    # This prevents us from mutating the input arguments
    adder = Add(*variables).copy()
    variables = adder.parents  # Get reference back to the variables
    adder.sample(size=999, random_state=42)  # Sample to populate _samples

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
