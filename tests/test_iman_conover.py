import numpy as np
import pytest
import scipy as sp
from scipy.stats import spearmanr

from probabilit.iman_conover import ImanConover, decorrelate


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sample_data(rng):
    N = 1000
    K = 3
    X = np.zeros((N, K))
    X[:, 0] = rng.normal(0, 1, N)
    X[:, 1] = rng.exponential(1, N)
    X[:, 2] = rng.uniform(0, 1, N)

    # Ensure positive definite correlation matrix by constructing from random matrix
    A = rng.normal(size=(K, K))
    C = A @ A.T  # This creates a positive definite matrix
    # Convert to correlation matrix
    d = np.sqrt(np.diag(C))
    C = C / d[:, None] / d[None, :]

    return X, C


def test_preserves_marginal_distributions(rng, sample_data):
    X, C = sample_data

    X_transformed = ImanConover(C)(X)
    for k in range(X.shape[1]):
        assert np.allclose(np.sort(X[:, k]), np.sort(X_transformed[:, k]))


def test_achieves_target_correlations(sample_data):
    X, C = sample_data

    X_transformed = ImanConover(C)(X)
    rank_corr = spearmanr(X_transformed)[0]
    assert np.allclose(rank_corr, C, atol=0.05)


def test_invalid_correlation_matrix(rng):
    N, K = 100, 3
    X = rng.normal(size=(N, K))
    C_invalid = np.array(
        [
            [1.0, 0.7, -0.3],
            [0.8, 1.0, 0.5],  # Non-symmetric
            [-0.3, 0.5, 1.0],
        ]
    )

    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        ImanConover(C_invalid)(X)


def test_extreme_correlations(rng):
    N, K = 1000, 3
    X = rng.normal(size=(N, K))

    # Create extreme but valid correlation matrix
    # Using the fact that a correlation matrix with ones on diagonal
    # and same value rho everywhere else is positive definite if
    # rho > -1/(K-1) where K is matrix size
    rho = 0.99  # High correlation but still allows matrix to be positive definite
    C_extreme = np.ones((K, K)) * rho
    np.fill_diagonal(C_extreme, 1.0)

    # Verify matrix is positive definite before test
    eigenvals = np.linalg.eigvals(C_extreme)
    assert np.all(eigenvals > 0), "Test correlation matrix is not positive definite"

    X_extreme = ImanConover(C_extreme)(X)
    rank_corr_extreme = spearmanr(X_extreme)[0]
    assert np.allclose(rank_corr_extreme, C_extreme, atol=0.05)


def test_correlation_matrix_validation(rng):
    N = 100
    K = 3
    X = rng.normal(size=(N, K))

    # Create non-positive definite matrix
    C_invalid = np.array([[1.0, 2.0, 0.3], [2.0, 1.0, 0.2], [0.3, 0.2, 1.0]])

    with pytest.raises((ValueError, np.linalg.LinAlgError)):
        ImanConover(C_invalid)(X)


def test_input_validation(rng):
    N = 100
    K = 3
    X = rng.normal(size=(N, K))
    C = np.array(
        [
            [1.0, 0.5],  # Wrong size
            [0.5, 1.0],
        ]
    )

    with pytest.raises(ValueError):
        ImanConover(C)(X)


def test_orthogonality_precision(rng):
    """Test that uncorrelated variables remain very close to orthogonal"""
    N = 100
    K = 4

    # Create target correlation matrix with some zeros
    C_target = np.array(
        [
            [1.0, 0.0, 0.5, 0.0],
            [0.0, 1.0, 0.0, 0.5],
            [0.5, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 1.0],
        ]
    )

    X = rng.normal(size=(N, K))
    X_transformed = ImanConover(C_target)(X)
    rank_corr = spearmanr(X_transformed)[0]

    # Get the elements that should be zero
    zero_mask = C_target == 0
    achieved_zeros = rank_corr[zero_mask]

    # With variance reduction, these should be very close to zero
    # Using much tighter tolerance than regular correlation tests
    # Threshold found by running thousands of tests.
    # This will fail much more often if we do not use the variance
    # reduction technique described in the paper.
    assert np.all(np.abs(achieved_zeros) < 0.12), (
        "Zero correlations not maintained with sufficient precision"
    )


class TestImanConover:
    @pytest.mark.parametrize("seed", range(100))
    def test_marginals_and_correlation_distance(self, seed):
        rng = np.random.default_rng(seed)

        n_variables = rng.integers(2, 100)
        n_observations = n_variables * 10

        # Create a random correlation matrix and a random data matrix
        A = rng.normal(size=(n_variables * 2, n_variables))
        desired_corr = 0.9 * np.corrcoef(A, rowvar=False) + 0.1 * np.eye(n_variables)
        X = rng.normal(size=(n_observations, n_variables))

        # Tranform the data
        transform = ImanConover(desired_corr)
        X_transformed = transform(X)

        # Check that all columns (variables) have equal marginals.
        # In other words, Iman-Conover can permute each column individually,
        # but they should have identical entries before and after.
        for j in range(X.shape[1]):
            assert np.allclose(np.sort(X[:, j]), np.sort(X_transformed[:, j]))

        # After the Iman-Conover transform, the distance between the desired
        # correlation matrix should be smaller than it was before.
        X_corr = np.corrcoef(X, rowvar=False)
        distance_before = sp.linalg.norm(X_corr - desired_corr, ord="fro")

        X_trans_corr = np.corrcoef(X_transformed, rowvar=False)
        distance_after = sp.linalg.norm(X_trans_corr - desired_corr, ord="fro")

        assert distance_after <= distance_before

    def test_identity_correlation_matrix(self):
        rng = np.random.default_rng(42)

        n_observations = 5
        n_variables = 3
        rng = np.random.default_rng(42)

        # Create a random correlation matrix and a random data matrix
        desired_corr = np.identity(n_variables)
        transform = ImanConover(desired_corr)

        # Create data and decorrelate it completely
        X = rng.normal(size=(n_observations, n_variables))
        X = decorrelate(X, remove_variance=True)
        assert np.allclose(np.corrcoef(X, rowvar=False), np.eye(n_variables))

        # Transform it to identity correlation, which it already has
        transform = ImanConover(desired_corr)
        X_transformed = transform(X)

        assert np.allclose(X, X_transformed)

    def test_dataset_with_unity_correlation_in_ranks(self):
        # This dataset is interesting because while the correlation
        # between the variables is ~0.6, when the data is ranked the
        # correlation becomes 1. Rank(row) = [1, 2, 3] for both rows.
        X = np.array([[1.0, 1], [2.0, 1.1], [2.1, 3]])

        desired_corr = np.identity(2)

        transform = ImanConover(desired_corr)
        with pytest.raises(ValueError):
            transform(X)


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
