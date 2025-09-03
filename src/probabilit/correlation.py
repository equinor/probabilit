"""
Correlation
-----------

Inducing correlations, working with correlation matrices, etc.


Using Iman-Conover with Latin Hybercube sampling
------------------------------------------------

Sample on the unit hypercube using LatinHypercube

>>> import scipy as sp
>>> sampler = sp.stats.qmc.LatinHypercube(d=2, seed=42, scramble=True)
>>> samples = sampler.random(n=100)

Map to distributions

>>> X = np.vstack((sp.stats.triang(0.5).ppf(samples[:, 0]),
...                sp.stats.gamma.ppf(samples[:, 1], a=1))).T

Induce correlations

>>> float(sp.stats.pearsonr(*X.T).statistic)
0.065898...
>>> correlation_matrix = np.array([[1, 0.3], [0.3, 1]])
>>> transform = ImanConover().set_target(correlation_matrix)
>>> X_transformed = transform(X)
>>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
0.279652...
"""

import numpy as np
import scipy as sp
import abc

import contextlib
import os
import itertools

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


class CorrelatorError(Exception):
    pass


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


def _is_positive_definite(X):
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False


class Correlator(abc.ABC):
    def set_target(self, correlation_matrix):
        """Set target correlation matrix."""
        if not isinstance(correlation_matrix, np.ndarray):
            raise TypeError("Input argument `correlation_matrix` must be NumPy array.")
        if not correlation_matrix.ndim == 2:
            raise ValueError("Correlation matrix must be square.")
        if not correlation_matrix.shape[0] == correlation_matrix.shape[1]:
            raise ValueError("Correlation matrix must be square.")
        if not np.allclose(np.diag(correlation_matrix), 1.0):
            raise ValueError("Correlation matrix must have 1.0 on diagonal.")
        if not np.allclose(correlation_matrix.T, correlation_matrix):
            raise ValueError("Correlation matrix must be symmetric.")
        if not _is_positive_definite(correlation_matrix):
            raise ValueError("Correlation matrix must be positive definite.")

        self.C = correlation_matrix.copy()
        self.P = np.linalg.cholesky(self.C)
        return self

    def _validate_X(self, X, check_rows_cols=True):
        """Validate array X of shape (observations, variables)."""
        if not (hasattr(self, "C") and hasattr(self, "P")):
            raise CorrelatorError("User must call `set_target` first.")

        if not isinstance(X, np.ndarray):
            raise TypeError("Input argument `X` must be NumPy array.")
        if not X.ndim == 2:
            raise ValueError("Correlation matrix must be square.")

        N, K = X.shape

        if self.P.shape[0] != K:
            msg = f"Shape of `X` ({X.shape}) does not match shape of "
            msg += f"correlation matrix ({self.P.shape})"
            raise ValueError(msg)

        if check_rows_cols and N <= K:
            msg = f"The matrix X must have rows > columns. Got shape: {X.shape}"
            raise ValueError(msg)

        return N, K


class Cholesky(Correlator):
    """Create a Cholesky transform.

    Parameters
    ----------
    correlation_matrix : ndarray
        Target correlation matrix of shape (K, K). The Iman-Conover will
        try to induce a correlation on the data set X so that corr(X) is
        as close to `correlation_matrix` as possible.

    Examples
    --------
    Create a desired correction of 0.7 and a data set X with no correlation.
    >>> correlation_matrix = np.array([[1, 0.7], [0.7, 1]])
    >>> rng = np.random.default_rng(4)
    >>> X = rng.normal(size=(9, 2))
    >>> sp.stats.pearsonr(*X.T).statistic.round(6)
    np.float64(-0.025582)

    >>> transform = Cholesky().set_target(correlation_matrix)
    >>> X_transformed = transform(X)
    >>> sp.stats.pearsonr(*X_transformed.T).statistic.round(6)
    np.float64(0.7)

    Verify that mean and std is the same before and after:
    >>> np.mean(X, axis=0)
    array([-0.63531692,  0.70114825])
    >>> np.mean(X_transformed, axis=0)
    array([-0.63531692,  0.70114825])
    >>> np.std(X, axis=0)
    array([1.11972638, 0.75668173])
    >>> np.std(X_transformed, axis=0)
    array([1.11972638, 0.75668173])

    """

    def set_target(self, correlation_matrix):
        super().set_target(correlation_matrix)
        return self

    def __call__(self, X):
        """Transform an input matrix X.

        Parameters
        ----------
        X : ndarray
            Input matrix of shape (N, K). This is the data set that we want to
            induce correlation structure on. X must have at least K + 1
            independent rows, because corr(X) cannot be singular.

        Returns
        -------
        ndarray
            Output matrix of shape (N, K). This data set will have a
            correlation structure that is more similar to `correlation_matrix`.

        """
        self._validate_X(X)
        N, K = X.shape

        # Remove existing mean and std from marginal distributions
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        X_n = (X - mean) / std

        # Removing existing correlation
        cov = np.cov(X_n, rowvar=False, ddof=0)
        P = np.linalg.cholesky(cov)  # P is lower triangular

        # Compute X_n = X_n @ inv(P).T @ self.P.T
        # Several numerical improvements are available to us here:
        # (1) Evaluate left-to-right or right-to-left depending on sizes
        # (2) Do not invert the matrix, instead solve triangular systems
        # (3) Use LAPACK routine TRMM (scipy.linalg.blas.dtrmm)
        # I choose to implement (1) and (2), but avoid the LACACK calls for now.

        # When it comes to evaluation order (point (1) above), it's better to
        # evaluate left-to-right if N < K, and right-to-left if N > K.
        # Since N > K (must have rows > columns), we evaluate right-to-left
        transform = sp.linalg.solve_triangular(P.T, self.P.T, lower=False)
        return mean + X_n @ (transform * std)


class ImanConover(Correlator):
    """Create an Iman-Conover transform.

    Parameters
    ----------
    correlation_matrix : ndarray
        Target correlation matrix of shape (K, K). The Iman-Conover will
        try to induce a correlation on the data set X so that corr(X) is
        as close to `correlation_matrix` as possible.

    Notes
    -----
    The implementation follows the original paper:
    Iman, R. L., & Conover, W. J. (1982). A distribution-free approach to
    inducing rank correlation among input variables. Communications in
    Statistics - Simulation and Computation, 11(3), 311-334.
    https://www.tandfonline.com/doi/epdf/10.1080/03610918208812265?needAccess=true
    https://www.uio.no/studier/emner/matnat/math/STK4400/v05/undervisningsmateriale/A%20distribution-free%20approach%20to%20rank%20correlation.pdf

    Other useful sources:
    - https://blogs.sas.com/content/iml/2021/06/16/geometry-iman-conover-transformation.html
    - https://blogs.sas.com/content/iml/2021/06/14/simulate-iman-conover-transformation.html
    - https://aggregate.readthedocs.io/en/stable/5_technical_guides/5_x_iman_conover.html

    Examples
    --------
    Create a desired correction of 0.7 and a data set X with no correlation.
    >>> correlation_matrix = np.array([[1, 0.7], [0.7, 1]])
    >>> transform = ImanConover().set_target(correlation_matrix)
    >>> X = np.array([[0, 0  ],
    ...               [0, 0.5],
    ...               [0,  1 ],
    ...               [1, 0  ],
    ...               [1, 0.5],
    ...               [1, 1  ]])
    >>> X_transformed = transform(X)
    >>> X_transformed
    array([[0. , 0. ],
           [0. , 0. ],
           [0. , 0.5],
           [1. , 0.5],
           [1. , 1. ],
           [1. , 1. ]])

    The original data X has no correlation at all, while the transformed
    data has correlation that is closer to the desired correlation structure:

    >>> sp.stats.pearsonr(*X.T).statistic.round(6)
    np.float64(0.0)
    >>> sp.stats.pearsonr(*X_transformed.T).statistic.round(6)
    np.float64(0.816497)

    Achieving the exact correlation structure might be impossible. For the
    input matrix above, there is no permutation of the columns that yields
    the exact desired correlation of 0.7. Iman-Conover is a heuristic that
    tries to get as close as possible.

    With many samples, we get good results if the data are normal:

    >>> rng = np.random.default_rng(42)
    >>> X = rng.normal(size=(1000, 2))
    >>> X_transformed = transform(X)
    >>> sp.stats.pearsonr(*X_transformed.T).statistic.round(6)
    np.float64(0.697701)

    But if the data are far from normal (here:lognormal), the results are
    not as good. This is because correlation is induced in a normal space
    before the result is mapped back to the original marginal distributions.

    >>> rng = np.random.default_rng(42)
    >>> X = rng.lognormal(size=(1000, 2))
    >>> X_transformed = transform(X)
    >>> sp.stats.pearsonr(*X_transformed.T).statistic.round(6)
    np.float64(0.592541)
    """

    def set_target(self, correlation_matrix):
        super().set_target(correlation_matrix)
        return self

    def __call__(self, X):
        """Transform an input matrix X.

        The output will have the same marginal distributions, but with
        induced correlation.

        Parameters
        ----------
        X : ndarray
            Input matrix of shape (N, K). This is the data set that we want to
            induce correlation structure on. X must have at least K + 1
            independent rows, because corr(X) cannot be singular.

        Returns
        -------
        ndarray
            Output matrix of shape (N, K). This data set will have a
            correlation structure that is more similar to `correlation_matrix`.

        """
        self._validate_X(X)
        N, K = X.shape

        # STEP ONE - Use van der Waerden scores to transform data to
        # approximately multivariate normal (but with correlations).
        # The new data has the same rank correlation as the original data.
        ranks = sp.stats.rankdata(X, axis=0) / (N + 1)
        normal_scores = sp.stats.norm.ppf(ranks)  # + np.random.randn(N, K) * epsilon

        # STEP TWO - Remove correlations from the transformed data
        empirical_correlation = np.corrcoef(normal_scores, rowvar=False)
        if not _is_positive_definite(empirical_correlation):
            msg = "Rank data correlation not positive definite."
            msg += "There are perfect correlations in the ranked data."
            msg += "Supply more data (rows in X) or sample differently."
            raise ValueError(msg)

        decorrelation_matrix = np.linalg.cholesky(empirical_correlation)

        # We exploit the fact that Q is lower-triangular and avoid the inverse.
        # X = N @ inv(Q)^T  =>  X @ Q^T = N  =>  (Q @ X^T)^T = N
        decorrelated_scores = sp.linalg.solve_triangular(
            decorrelation_matrix, normal_scores.T, lower=True
        ).T

        # STEP THREE - Induce correlations in transformed space
        correlated_scores = decorrelated_scores @ self.P.T

        # STEP FOUR - Map back to original space using ranks, ensuring
        # that marginal distributions are preserved
        result = np.empty_like(X)
        for k in range(K):
            # If row j is the k'th largest in `correlated_scores`, then
            # we map the k'th largest entry in X to row j.
            ranks = sp.stats.rankdata(correlated_scores[:, k]).astype(int) - 1
            result[:, k] = np.sort(X[:, k])[ranks]

        return result


class PermutationCorrelator(Correlator):
    def __init__(
        self,
        *,
        weights=None,
        iterations=1000,
        tol=0.01,
        correlation_type="pearson",
        seed=None,
        verbose=False,
    ):
        """Create a PermutationCorrelator instance, which induces correlations
        between variables in X by randomly shuffling rows within each column.

        Parameters
        ----------
        weights : 2d numpy array or None, optional
            Elementwise weights for the target correlation matrix.
            The default is None, which corresponds to uniform weights.
        iterations : int, optional
            Maximal number of iterations to run. Each iterations consists of
            one loop over all variables. Choosing 0 means infinite iterations.
            The default is 1000.
        tol : float, optional
            Tolerance for stopping criteria. Will stop when
            norm(desired_corr - actual_corr) < tol. The default is 0.05.
        correlation_type : str, optional
            Either "pearson" or "spearman". The default is "pearson".
        seed : int or None, optional
            A seed for the random number generator. The default is None.
        verbose : bool, optional
            Whether or not to print information. The default is False.

        Notes
        -----
        The paper "Correlation control in small-sample Monte Carlo type
        simulations I: A simulated annealing approach" by Vořechovský et al.
        proposes using simulated annealing. We implement a simple randomized
        hill climbing procedure instead, because it is good enough.
          - https://www.sciencedirect.com/science/article/pii/S0266892009000113
          - https://en.wikipedia.org/wiki/Hill_climbing

        Examples
        --------
        >>> rng = np.random.default_rng(42)
        >>> X = rng.normal(size=(100, 2))
        >>> float(sp.stats.pearsonr(*X.T).statistic)
        0.1573...
        >>> correlation_matrix = np.array([[1, 0.7], [0.7, 1]])
        >>> perm_trans = PermutationCorrelator(seed=0).set_target(correlation_matrix)
        >>> X_transformed = perm_trans(X)
        >>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
        0.6824...

        For large matrices, it often makes sense to first use Iman-Conover
        to get a good initial solution, then give it to PermutationCorrelator.
        Start by creating a large correlation matrix:

        >>> variables = 25
        >>> correlation_matrix = np.ones((variables, variables)) * 0.7
        >>> np.fill_diagonal(correlation_matrix, 1.0)
        >>> perm_trans = PermutationCorrelator(iterations=1000, tol=1e-6,
        ...                                    seed=0, verbose=True)
        >>> perm_trans = perm_trans.set_target(correlation_matrix)

        Create data X, then transform using Iman-Conover:

        >>> X = rng.normal(size=(10 * variables, variables))
        >>> perm_trans._error(X) # Initial error
        0.4846...
        >>> ic_trans = ImanConover().set_target(correlation_matrix)
        >>> X_ic = ic_trans(X)
        >>> perm_trans._error(X_ic) # Error after Iman-Conover
        0.0071...
        >>> X_ic_pc = perm_trans(X_ic)
        Running permutation correlator for 1000 iterations.
         Iter    100  Error: 0.0071 Swaps:  7
         Iter    200  Error: 0.0068 Swaps:  5
         Iter    300  Error: 0.0059 Swaps:  3
         Iter    400  Error: 0.0045 Swaps:  2
         Iter    500  Error: 0.0035 Swaps:  1
         Iter    600  Error: 0.0023 Swaps:  1
         Iter    700  Error: 0.0017 Swaps:  1
         Iter    800  Error: 0.0013 Swaps:  1
         Iter    900  Error: 0.0011 Swaps:  1
         Iter   1000  Error: 0.0010 Swaps:  1

        >>> perm_trans._error(X_ic_pc) # Error after Iman-Conover + permutation
        0.0009511...
        """
        corr_types = {"pearson": self._pearson, "spearman": self._spearman}

        if not (weights is None or np.all(weights > 0)):
            raise ValueError("`weights` must have positive entries.")
        if not (isinstance(iterations, int) and iterations >= 0):
            raise ValueError("`iterations` must be non-negative integer.")
        if not isinstance(tol, float) and tol > 0:
            raise ValueError("`tol` must be a positive float.")
        if not (isinstance(correlation_type, str) and correlation_type in corr_types):
            raise ValueError(
                f"`correlation_type` must be one of: {tuple(corr_types.keys())}"
            )
        if not (seed is None or isinstance(seed, int)):
            raise TypeError("`seed` must be None or an integer")
        if not isinstance(verbose, bool):
            raise TypeError("`verbose` must be boolean")

        self.iters = iterations
        self.tol = tol
        self.correlation_func = corr_types[correlation_type]
        self.rng = np.random.default_rng(seed)
        self.verbose = verbose

    def set_target(self, correlation_matrix, *, weights=None):
        """Set the target correlation matrix."""
        super().set_target(correlation_matrix)
        weights = np.ones_like(self.C) if weights is None else weights
        self.weights = weights / np.sum(weights)
        self.triu_indices = np.triu_indices(self.C.shape[0], k=1)
        return self

    def _pearson(self, X):
        """Given a matrix X of shape (m, n), return a matrix of shape (n, n)
        with Pearson correlation coefficients."""
        # The majority of runtime is spent computing correlation coefficients.
        # Any attempt to speed up this code should focus on that.
        # It's possible to compute the difference is the objective function
        # without explicitly computing the empirical correlation afresh in
        # every iteration. If X has shape (m, n), then this can take the
        # runtime from O(m*n*n) to O(n), but it requires Python-loops and
        # bookkeeping.
        return np.corrcoef(X, rowvar=False)

    def _spearman(self, X):
        """Given a matrix X of shape (m, n), return a matrix of shape (n, n)
        with Spearman correlation coefficients."""
        if X.shape[1] == 2:
            spearman_corr = sp.stats.spearmanr(X).statistic
            return np.array([[1.0, spearman_corr], [spearman_corr, 1.0]])
        else:
            return sp.stats.spearmanr(X).statistic

    @staticmethod
    def _swap(X, i, j, k):
        """Swap rows i and j in column k inplace."""
        X[i, k], X[j, k] = X[j, k], X[i, k]

    def _error(self, X):
        """Compute RMSE over upper triangular part of corr(X) - C."""
        # TODO: An optimization is to compute the error only for the swapped
        # variable. If k is swapped, then only row and col for k is changed.
        corr = self.correlation_func(X)  # Correlation matrix
        idx = self.triu_indices  # Get upper triangular indices (ignore diag)
        weighted_residuals_sq = self.weights[idx] * (corr[idx] - self.C[idx]) ** 2.0
        return float(np.sqrt(np.sum(weighted_residuals_sq)))

    def __call__(self, X):
        """Cycle through through columns (variables), and for each
        column it swaps random rows (observations). If the result
        leads to a smaller error (correlation closer to target), then it is
        kept. If not we try again.

        Parameters
        ----------
        X : np.ndarray
            A matrix with shape (observations, variables).

        Returns
        -------
        A copy of X where rows within each column are shuffled.
        """
        self._validate_X(X, check_rows_cols=False)  # Allow rows <= columns

        num_obs, num_vars = X.shape
        if not (isinstance(X, np.ndarray) and X.ndim == 2):
            raise ValueError("`X` must be a 2D numpy array.")
        if not num_vars == self.C.shape[0]:
            raise ValueError(
                "Number of variables in `X` does not match `correlation_matrix`."
            )

        if self.verbose:
            print(
                f"Running permutation correlator for {self.iters if self.iters else 'inf'} iterations."
            )

        def subiters(n, i):
            """Number of sub-iterations (swaps) per iteration."""
            # Use longer swap lengths in early iterations. The last half
            # of the iterations will use 1 sub-iteration. The second half of the
            # first half will use 2, and so forth. The pattern is:
            # n = 2 => [2, 1]
            # n = 4 => [3, 2, 1, 1]
            # n = 4 => [4, 3, 2, 2, 1, 1, 1, 1]
            # n = 8 => [5, 4, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
            # This function computes a closed-form solution to indexing such
            # a list (of length n) at index i.
            C = np.log2(n) + 1
            return int(np.ceil(C ** (1 - (2 * i / n))))

        def product(iterations_gen, variables_gen):
            # itertools.product only works for finite inputs, so we need this
            for i in iterations_gen:
                for j in variables_gen:
                    yield (i, j)

        # Set up loop generator
        iter_gen = range(1, self.iters + 1) if self.iters else itertools.count(1)
        loop_gen = product(iter_gen, range(num_vars))  # (iteration, k)

        # Set up variables that are tracked in the main loop
        current_X = X.copy()
        current_error = self._error(current_X)
        iter_no_change = 0

        # Main loop. For each iteration, k cycles through all variables.
        # This parametrizes the algorithm so iterations is less sensitive to k.
        for iteration, k in loop_gen:
            print_iter = iteration % (self.iters // 10) if self.iters else 1000
            num_swaps = subiters(n=self.iters if self.iters else 10_000, i=iteration)
            if self.verbose and print_iter == 0 and k == 0:
                print(
                    f" Iter {iteration:>6}  Error: {current_error:.6f} Swaps: {num_swaps:>2}"
                )

            # Create a sequence of swaps of length `num_swaps`
            swaps = list(self.rng.integers(0, high=num_obs, size=(num_swaps, 2)))
            for i, j in swaps:
                # Turn current_X into a new proposed X by swapping two observations
                # i and j in column (variable) k. The swap is done in-place.
                self._swap(current_X, i, j, k)

            proposed_error = self._error(current_X)

            # Termination critera
            if proposed_error < self.tol:
                if self.verbose:
                    print(
                        f""" Terminating at iteration {iteration} due to tolerance. Error: {current_error:.6f}"""
                    )
                return current_X

            # The proposed X was better
            if proposed_error < current_error:
                current_error = proposed_error
                iter_no_change = 0

            # The proposed X was worse
            else:
                for i, j in reversed(swaps):  # Swap indices back
                    self._swap(current_X, i, j, k)
                iter_no_change += 1

        return current_X


def decorrelate(X, remove_variance=True):
    """Removes correlations or covariance from data X.

    Examples
    --------
    >>> X = np.array([[1. , 1. ],
    ...               [2. , 1.1],
    ...               [2.1, 3. ]])
    >>> X_decorr = decorrelate(X)

    The result has covariance equal to the identity matrix:

    >>> np.cov(X_decorr, rowvar=False).round(6)
    array([[1., 0.],
           [0., 1.]])

    The mean is preserved:

    >>> np.allclose(np.mean(X, axis=0), np.mean(X_decorr, axis=0))
    True

    The variance is removed:

    >>> np.var(X, axis=0)
    array([0.24666667, 0.84666667])
    >>> np.var(X_decorr, axis=0, ddof=1)
    array([1., 1.])

    We can optionally decorrelate while preserving the variance:

    >>> X_decorr = decorrelate(X, remove_variance=False)
    >>> np.cov(X_decorr, rowvar=False).round(6)
    array([[0.246667, 0.      ],
           [0.      , 0.846667]])
    >>> np.allclose(np.mean(X, axis=0), np.mean(X_decorr, axis=0))
    True
    """
    mean = np.mean(X, axis=0)
    var = np.var(X, axis=0, ddof=0)
    cov = np.cov(X, rowvar=False)

    L = np.linalg.cholesky(cov)  # L @ L.T = cov
    if not remove_variance:
        L = L / np.sqrt(var)

    # Computes X = (X - mean) @ inv(L).T
    X = sp.linalg.solve_triangular(L, (X - mean).T, lower=True).T

    return mean + X


if __name__ == "__main__":
    import pytest
    import matplotlib.pyplot as plt

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])

    rng = np.random.default_rng(2)
    p = 10
    n = 999

    sampler = sp.stats.qmc.Halton(d=p, seed=42, scramble=False)
    samples = sampler.random(n=n)
    samples = np.vstack(
        [rng.permutation(np.linspace(0 + 1e-6, 1 - 1e-6, num=n)) for k in range(p)]
    ).T

    X = np.zeros_like(samples)
    for j in range(X.shape[1]):
        func_i = int(rng.integers(0, 4))
        func = [
            sp.stats.uniform(),
            sp.stats.expon(),
            sp.stats.norm(),
            sp.stats.lognorm(s=1),
        ]
        func = func[func_i]
        # func = sp.stats.uniform()
        X[:, j] = func.ppf(samples[:, j]) * rng.normal(loc=5, scale=2)

    plt.figure(figsize=(4, 4))
    plt.title("Original data set")
    plt.scatter(X[:, 0], X[:, 1], s=1)
    plt.show()

    target = np.ones((p, p)) * 0.5
    np.fill_diagonal(target, 1.0)

    # Create permutation correlator
    correlator = PermutationCorrelator(verbose=True, tol=1e-4, iterations=10_000)
    correlator.set_target(target)

    # First try cholesky and
    X_chol = Cholesky().set_target(target)(X)
    print(f"Cholesky error: {correlator._error(X_chol):.4f}")
    plt.figure(figsize=(4, 4))
    plt.title(f"Cholesky error: {correlator._error(X_chol):.4f}")
    plt.scatter(X_chol[:, 0], X_chol[:, 1], s=1)
    plt.show()

    # First try cholesky and
    X_ic = ImanConover().set_target(target)(X)
    print(f"ImanConover error: {correlator._error(X_ic):.4f}")
    plt.figure(figsize=(4, 4))
    plt.title(f"ImanConover error: {correlator._error(X_ic):.4f}")
    plt.scatter(X_ic[:, 0], X_ic[:, 1], s=1)
    plt.show()

    X_pc = correlator(X)
    print(f"PermutationCorrelator error: {correlator._error(X_pc):.4f}")
    plt.figure(figsize=(4, 4))
    plt.title(f"PermutationCorrelator error: {correlator._error(X_pc):.4f}")
    plt.scatter(X_pc[:, 0], X_pc[:, 1], s=1)
    plt.show()
