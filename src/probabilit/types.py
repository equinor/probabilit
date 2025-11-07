from typing import Literal

import numpy as np

Array1D = np.ndarray[tuple[int], np.dtype[np.float64]]
Array2D = np.ndarray[tuple[int, int], np.dtype[np.float64]]
CorrelationType = Literal["pearson", "spearman"]
CorrelatorType = Literal["cholesky", "imanconover", "permutation", "composite"]
SamplingMethodType = Literal["lhs", "halton", "sobol"]
DistributionType = Literal[
    "uniform",
    "norm",
    "truncnorm",
    "lognorm",
    "beta",
    "triang",
    "dirichlet",
]
