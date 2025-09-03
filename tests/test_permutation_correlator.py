from probabilit.correlation import PermutationCorrelator
import numpy as np
import pytest
import scipy as sp


class TestPermutationCorrelator:
    @pytest.mark.parametrize("seed", range(25))
    def test_convergence(self, seed):
        # With default parameters, convergence should be decent
        # regardless of the size of the problem
        rng = np.random.default_rng(seed)

        n_variables = rng.integers(2, 20)
        n_observations = n_variables * rng.integers(5, 50)

        # Create a correlation matrix and a random data matrix
        desired_corr = np.ones((n_variables, n_variables)) * 0.5
        np.fill_diagonal(desired_corr, val=1.0)
        X = rng.normal(size=(n_observations, n_variables))

        # Tranform the data
        transform = PermutationCorrelator(seed=seed).set_target(desired_corr)
        X_transformed = transform(X)

        actual_corr = np.corrcoef(X_transformed, rowvar=False)
        rmse = np.sqrt(np.mean((actual_corr - desired_corr) ** 2))
        assert rmse < 0.1

    @pytest.mark.parametrize("seed", range(10))
    def test_marginals_and_correlation_distance(self, seed):
        rng = np.random.default_rng(seed)

        n_variables = rng.integers(2, 10)
        n_observations = n_variables * 10

        # Create a random correlation matrix and a random data matrix
        A = rng.normal(size=(n_variables * 2, n_variables))
        desired_corr = 0.9 * np.corrcoef(A, rowvar=False) + 0.1 * np.eye(n_variables)
        X = rng.normal(size=(n_observations, n_variables))

        # Tranform the data
        transform = PermutationCorrelator(seed=0, iterations=10)
        transform = transform.set_target(desired_corr)
        X_transformed = transform(X)

        # Check that all columns (variables) have equal marginals.
        for j in range(X.shape[1]):
            assert np.allclose(np.sort(X[:, j]), np.sort(X_transformed[:, j]))

        # After using the PermutationCorrelator, the distance to the
        # desired correlation matrix should be smaller than it was before.
        X_corr = np.corrcoef(X, rowvar=False)
        distance_before = sp.linalg.norm(X_corr - desired_corr, ord="fro")

        X_trans_corr = np.corrcoef(X_transformed, rowvar=False)
        distance_after = sp.linalg.norm(X_trans_corr - desired_corr, ord="fro")

        assert distance_after <= distance_before

    def test_dataset_with_more_variables_than_observations(self):
        rng = np.random.default_rng(42)
        X = rng.normal(size=(5, 10))

        desired_corr = np.identity(10)
        transform = PermutationCorrelator(seed=0).set_target(desired_corr)
        X_trans = transform(X)
        assert transform._error(X_trans) < transform._error(X)


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
