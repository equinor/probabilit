from probabilit.correlation import (
    Permutation,
    CorrelationMatrix,
    SwapIndexGenerator,
)
import numpy as np
import pytest
import scipy as sp


class TestSwapIndexGenerator:
    @pytest.mark.parametrize("seed", range(10))
    def test_disjoint(self, seed):
        rng = np.random.default_rng(seed)
        n = rng.integers(5, 15)
        generator = SwapIndexGenerator(rng=rng, n=n)

        for _ in range(50):
            size = rng.integers(1, 20)
            i, j = generator(size)
            assert not set(i).intersection(set(j))


class TestCorrelationMatrix:
    @pytest.mark.parametrize("seed", range(10))
    @pytest.mark.parametrize("correlation_type", ["pearson", "spearman"])
    def test_squences_of_swaps(self, seed, correlation_type):
        def corr(X, correlation_type):
            if correlation_type == "pearson":
                return np.corrcoef(X, rowvar=False)
            elif correlation_type == "spearman":
                return sp.stats.spearmanr(X).statistic
            else:
                raise ValueError("Invalid correlation type")

        rng = np.random.default_rng(seed)
        X = rng.normal(size=(9, 4))

        correlation_matrix = CorrelationMatrix(X, correlation_type=correlation_type)

        # Test that the CorrelationMatrix computes correlation correctly
        observed_corr = corr(X, correlation_type)
        np.testing.assert_allclose(observed_corr, correlation_matrix[:, :], atol=1e-12)

        # Test two consequtive single swap
        # --------------------------------
        X_swapped = np.copy(X)
        correlation_matrix.commit(col=1, i=0, j=1)
        X_swapped[0, 1], X_swapped[1, 1] = X_swapped[1, 1], X_swapped[0, 1]

        observed_corr = corr(X_swapped, correlation_type)
        np.testing.assert_allclose(observed_corr, correlation_matrix[:, :], atol=1e-12)

        # Test another swap on the same matrix
        correlation_matrix.commit(col=1, i=2, j=3)
        X_swapped[2, 1], X_swapped[3, 1] = X_swapped[3, 1], X_swapped[2, 1]

        observed_corr = corr(X_swapped, correlation_type)
        np.testing.assert_allclose(observed_corr, correlation_matrix[:, :], atol=1e-12)

        # Test two swaps on the same matrix at once
        # -----------------------------------------
        correlation_matrix = CorrelationMatrix(X, correlation_type=correlation_type)
        X_swp = np.copy(X)
        correlation_matrix.commit(col=1, i=[0, 1], j=[2, 3])
        X_swp[[0, 1], 1], X_swp[[2, 3], 1] = X_swp[[2, 3], 1], X_swp[[0, 1], 1]

        observed_corr = corr(X_swp, correlation_type)
        np.testing.assert_allclose(observed_corr, correlation_matrix[:, :], atol=1e-12)

        # Check that committing back and forth leads to the same matrix
        before = np.copy(correlation_matrix.X)
        correlation_matrix.commit(col=1, i=[0, 1], j=[2, 3])
        midway = np.copy(correlation_matrix.X)
        correlation_matrix.commit(col=1, i=[0, 1], j=[2, 3])
        after = np.copy(correlation_matrix.X)

        np.testing.assert_allclose(before, after)
        assert not np.allclose(before, midway)

        # Test a single swap
        a, b = correlation_matrix.X[0, 2], correlation_matrix.X[5, 2]
        correlation_matrix.commit(col=2, i=0, j=5)
        np.testing.assert_allclose(a, correlation_matrix.X[5, 2])
        np.testing.assert_allclose(b, correlation_matrix.X[0, 2])

    @pytest.mark.parametrize("seed", range(100))
    def test_swap(self, seed):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(9, 4))

        # Test a single swap
        correlation_matrix = CorrelationMatrix(X)
        a, b = correlation_matrix.X[0, 2], correlation_matrix.X[5, 2]
        correlation_matrix.commit(col=2, i=0, j=5)
        np.testing.assert_allclose(a, correlation_matrix.X[5, 2])
        np.testing.assert_allclose(b, correlation_matrix.X[0, 2])

    @pytest.mark.parametrize("seed", range(100))
    @pytest.mark.parametrize("correlation_type", ["pearson", "spearman"])
    def test_that_many_single_equals_one_large_swap(self, seed, correlation_type):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(9, 4))

        # Test a chain of swaps
        correlation_matrix = CorrelationMatrix(X, correlation_type=correlation_type)
        a, b = [1, 3, 2], [6, 4, 5]
        for i, j in zip(a, b):
            correlation_matrix.commit(col=2, i=i, j=j)
        X_singles = np.copy(correlation_matrix.X)

        correlation_matrix = CorrelationMatrix(X, correlation_type=correlation_type)
        correlation_matrix.commit(col=2, i=a, j=b)
        X_multiples = np.copy(correlation_matrix.X)

        np.testing.assert_allclose(X_singles, X_multiples)


class TestPermutation:
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
        transform = Permutation(random_state=seed).set_target(desired_corr)
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
        transform = Permutation(random_state=0, iterations=10)
        transform = transform.set_target(desired_corr)
        X_transformed = transform(X)

        # Check that all columns (variables) have equal marginals.
        for j in range(X.shape[1]):
            assert np.allclose(np.sort(X[:, j]), np.sort(X_transformed[:, j]))

        # After using the Permutation, the distance to the
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
        transform = Permutation(random_state=0).set_target(desired_corr)
        X_trans = transform(X)

        cor_X_trans = np.corrcoef(X_trans, rowvar=False)
        cor_X = np.corrcoef(X, rowvar=False)
        assert transform._error(cor_X_trans, desired_corr) < transform._error(
            cor_X, desired_corr
        )


if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
