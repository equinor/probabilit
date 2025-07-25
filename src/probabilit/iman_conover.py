"""
An implementation of the Iman-Conover transformation.

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
>>> transform = ImanConover(correlation_matrix)
>>> X_transformed = transform(X)
>>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
0.279652...
"""

import numpy as np
import scipy as sp
import abc


def _is_positive_definite(X):
    try:
        np.linalg.cholesky(X)
        return True
    except np.linalg.LinAlgError:
        return False


class Correlator(abc.ABC):
    def __init__(self, correlation_matrix):
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

    def _validate_X(self, X):
        """Validate array X of shape (observations, variables)."""

        if not isinstance(X, np.ndarray):
            raise TypeError("Input argument `X` must be NumPy array.")
        if not X.ndim == 2:
            raise ValueError("Correlation matrix must be square.")

        N, K = X.shape

        if self.P.shape[0] != K:
            msg = f"Shape of `X` ({X.shape}) does not match shape of "
            msg += f"correlation matrix ({self.P.shape})"
            raise ValueError(msg)

        if N <= K:
            msg = f"The matrix X must have rows > columns. Got shape: {X.shape}"
            raise ValueError(msg)

        return N, K


class Cholesky(Correlator):
    def __init__(self, correlation_matrix):
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

        >>> transform = Cholesky(correlation_matrix)
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
        super().__init__(correlation_matrix)

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
    def __init__(self, correlation_matrix):
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
        >>> transform = ImanConover(correlation_matrix)
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
        super().__init__(correlation_matrix)

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

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])
