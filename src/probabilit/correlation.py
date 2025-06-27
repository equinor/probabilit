"""
Correlation
-----------

Inducing correlations, working with correlation matrices, etc.
"""

import contextlib
import os

# CVXPY prints error messages about incompatible ortools version during import.
# Since we use the SCS solver and not GLOP/PDLP (which need ortools), these errors
# are irrelevant and would only confuse users. We suppress them by redirecting
# stdout/stderr during import.
# https://github.com/cvxpy/cvxpy/issues/2470
with (
    open(os.devnull, "w") as devnull,
    contextlib.redirect_stdout(devnull),
    contextlib.redirect_stderr(devnull),
):
    import cvxpy as cp

import numpy as np


def nearest_correlation_matrix(matrix, *, weights=None, eps=1e-6, verbose=False):
    """Returns the correlation matrix nearest to `matrix`, weighted elementwise
    by `weights`.

    Parameters
    ----------
    matrix : np.ndarray
        The matrix that we want to find the nearest correlation matrix to.
        A square 2-dimensional NumPy ndarray.
    weights : np.ndarray or None, optional
        An elementwise weighting matrix. A square 2-dimensional NumPy ndarray
        that must have the same shape as `matrix`. The default is None.
    eps : float, optional
        Tolerance for the optimization solver. The default is 1e-6.
    verbose : bool, optional
        Whether to print information from the solver. The default is False.

    Returns
    -------
    np.ndarray
        The correlation matrix that is nearest to the input matrix.

    Notes
    -----
    This function implements equation (3) in the paper "An Augmented Lagrangian
    Dual Approach for the H-Weighted Nearest Correlation Matrix Problem" by
    Houduo Qi and Defeng Sun.
        http://www.personal.soton.ac.uk/hdqi/REPORTS/Cor_matrix_H.pdf
    Another useful link is:
        https://nhigham.com/2020/04/14/what-is-a-correlation-matrix/

    Examples
    --------
    >>> X = np.array([[1, 1, 0],
    ...               [1, 1, 1],
    ...               [0, 1, 1]])
    >>> nearest_correlation_matrix(X)
    array([[1.        , 0.76068..., 0.15729...],
           [0.76068..., 1.        , 0.76068...],
           [0.15729..., 0.76068..., 1.        ]])
    >>> H = np.array([[1,   0.5, 0.1],
    ...               [0.5,   1, 0.5],
    ...               [0.1, 0.5, 1]])
    >>> nearest_correlation_matrix(X, weights=H)
    array([[1.        , 0.94171..., 0.77365...],
           [0.94171..., 1.        , 0.94171...],
           [0.77365..., 0.94171..., 1.        ]])
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError("Input argument `matrix` must be np.ndarray.")
    if not matrix.ndim == 2 and matrix.shape[0] == matrix.shape[1]:
        raise ValueError("Input argument `matrix` must be square.")

    # Switch to notation used in the paper
    G = matrix.copy()
    H = np.ones_like(G) if weights is None else weights

    if not isinstance(H, np.ndarray):
        raise TypeError("Input argument `weights` must be np.ndarray.")
    if not (H.shape == G.shape):
        raise ValueError("Argument `weights` must have same shape as `matrix`.")

    # To constrain Y to be Positive Symmetric Definite (PSD), you need to
    # either set PSD=True here, or add the special constraint 'Y >> 0'. See:
    # https://www.cvxpy.org/tutorial/constraints/index.html#semidefinite-matrices
    X = cp.Variable(shape=G.shape, PSD=True)

    # Objective and constraints for minimizing the weighted frobenius norm.
    # This is equation (3) in the paper. We set (X - eps * I) >> 0 as an extra
    # constraint. Mathematically this is not needed, but numerically it helps
    # by nudging the solution slightly more, so the minimum eigenvalue is > 0.
    objective = cp.norm(cp.multiply(H, X - G), "fro")
    eps_identity = (eps / G.shape[0]) * 10
    constraints = [cp.diag(X) == 1.0, (X - eps_identity * np.eye(G.shape[0])) >> 0]

    # For solver options, see:
    # https://www.cvxpy.org/tutorial/solvers/index.html#setting-solver-options
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver="SCS", verbose=verbose, eps=eps)
    X = X.value.copy()  # Copy over solution

    # We might get small eigenvalues due to numerics. Attempt to fix this by
    # recursively calling the solver with smaller values of epsilon. This is
    # an extra fail-safe that is very rarely triggered on actual data.
    is_symmetric = np.allclose(X, X.T)
    is_PD = np.linalg.eig(X)[0].min() > 0
    if not (is_symmetric and is_PD) and (eps > 1e-14):
        if verbose:
            print(f"Recursively calling solver with eps := {eps} / 10")
        return nearest_correlation_matrix(G, weights=H, eps=eps / 10, verbose=verbose)

    return X


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
