from probabilit.correlation import PermutationCorrelator
import numpy as np
import pytest
import scipy as sp

class TestPermutationCorrelator:
    
    
    @pytest.mark.parametrize("seed", range(25))
    def test_recommended_parameters(self, seed):
        rng = np.random.default_rng(seed)

        n_variables = rng.integers(2, 10)
        n_observations = n_variables * 5

        # Create a correlation matrix and a random data matrix
        desired_corr = np.ones((n_variables, n_variables)) * 0.7
        np.fill_diagonal(desired_corr, val=1.0)
        X = rng.normal(size=(n_observations, n_variables))

        # Tranform the data
        transform = PermutationCorrelator(seed=seed).set_target(desired_corr)        
        X_transformed = transform(X)
        
        rel_err = (transform._error(X_transformed) / transform._error(X))
        assert rel_err < 0.1
    
    
    @pytest.mark.parametrize("seed", range(1))
    def test_marginals_and_correlation_distance(self, seed):
        rng = np.random.default_rng(seed)

        n_variables = rng.integers(2, 100)
        n_observations = n_variables * 10

        # Create a random correlation matrix and a random data matrix
        A = rng.normal(size=(n_variables * 2, n_variables))
        desired_corr = 0.9 * np.corrcoef(A, rowvar=False) + 0.1 * np.eye(n_variables)
        X = rng.normal(size=(n_observations, n_variables))

        # Tranform the data
        transform = PermutationCorrelator(seed=0, 
                                          iterations=50,
                                          max_iter_no_change=10)
        transform = transform.set_target(desired_corr)
        X_transformed = transform(X)

        # Check that all columns (variables) have equal marginals.
        # In other words, Iman-Conover can permute each column individually,
        # but they should have identical entries before and after.
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
    
    def test_dataset_with_unity_correlation_in_ranks(self):
        # This dataset is interesting because while the correlation
        # between the variables is ~0.6, when the data is ranked the
        # correlation becomes 1. Rank(row) = [1, 2, 3] for both rows.
        X = np.array([[1.0, 1], [2.0, 1.1], [2.1, 3]])

        desired_corr = np.identity(2)

        transform = PermutationCorrelator(seed=0).set_target(desired_corr)
        X_trans = transform(X)
        assert np.allclose(X_trans, np.array([[1. , 1.1],
               [2.1, 1. ],
               [2. , 3. ]]))
        
        
if __name__ == "__main__":
    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
