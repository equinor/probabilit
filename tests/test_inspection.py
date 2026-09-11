import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from probabilit import Normal, plot


@pytest.mark.parametrize(
    "corr",
    [
        pytest.param(0.5, id="scalar"),
        pytest.param(np.float32(0.5), id="numpy-scalar"),
        pytest.param(np.array(0.5), id="zero-dimensional-array"),
        pytest.param(np.array([[1.0, 0.5], [0.5, 1.0]]), id="array"),
        pytest.param([[1.0, 0.5], [0.5, 1.0]], id="list"),
        pytest.param(pd.DataFrame([[1.0, 0.5], [0.5, 1.0]]), id="dataframe"),
    ],
)
def test_plot_accepts_array_like_correlations(corr):
    try:
        grid = plot(
            Normal(),
            Normal(),
            corr=corr,
            sample_kwargs={"size": 99, "correlator": "cholesky"},
            diag_kind="hist",
        )
        np.testing.assert_allclose(grid.data.corr().iloc[0, 1], 0.5)
    finally:
        plt.close("all")


@pytest.mark.parametrize("num_variables", [1, 3])
def test_scalar_correlation_requires_two_variables(num_variables):
    variables = [Normal() for _ in range(num_variables)]

    with pytest.raises(ValueError, match="exactly two variables"):
        plot(*variables, corr=0.5)
