import numpy as np
import pytest
from probabilit.correlation import nearest_correlation_matrix


class TestNearestCorrelationMatrix:
    @pytest.mark.parametrize("variables", range(2, 100, 10))
    def test_nearest_correlation_matrix(self, variables):
        """Test that we can cholesky decompose the solution."""

        rng = np.random.default_rng(variables)

        # Create a correlation matrix
        observations = rng.normal(size=(variables * 2, variables))
        matrix = np.corrcoef(observations, rowvar=False)

        # Taking the cholesky decomposition should work just fine
        np.linalg.cholesky(matrix)

        # Mess it up
        matrix = matrix + rng.normal(size=matrix.shape, scale=0.1)
        matrix = matrix - np.identity(variables) * np.mean(np.diag(matrix))

        # Now the cholesky decomposition should fail
        with pytest.raises(np.linalg.LinAlgError):
            np.linalg.cholesky(matrix)

        # Adjust the matrix to its nearest correlation matrix
        correlation_matrix = nearest_correlation_matrix(matrix)

        # Taking the cholesky decomposition should work now
        np.linalg.cholesky(correlation_matrix)

        # Diagonal entries should be 1.0 and the matrix should be symmetric
        assert np.allclose(np.diag(correlation_matrix), 1.0)
        assert np.allclose(correlation_matrix, correlation_matrix.T)

    def test_nearest_correlation_matrix_on_matlab_example(self):
        """These matrices are from the 'nearcorr' docs:
        https://www.mathworks.com/help/stats/nearcorr.html
        """
        # The matrix we want to adjust to become a correlation matrix
        A = np.array(
            [
                [1.0, 0.0, 0.0, 0.0, -0.936],
                [0.0, 1.0, -0.55, -0.3645, -0.53],
                [0.0, -0.55, 1.0, -0.0351, 0.0875],
                [0.0, -0.3645, -0.0351, 1.0, 0.4557],
                [-0.936, -0.53, 0.0875, 0.4557, 1.0],
            ]
        )

        W = np.array(
            [
                [0.0, 1.0, 0.1, 0.15, 0.25],
                [1.0, 0.0, 0.05, 0.025, 0.15],
                [0.1, 0.05, 0.0, 0.25, 1.0],
                [0.15, 0.025, 0.25, 0.0, 0.25],
                [0.25, 0.15, 1.0, 0.25, 0.0],
            ]
        )

        matlab_Y = np.array(
            [
                [1.0, 0.0014, 0.0287, -0.0222, -0.8777],
                [0.0014, 1.0, -0.498, -0.7268, -0.4567],
                [0.0287, -0.498, 1.0, -0.0358, 0.0878],
                [-0.0222, -0.7268, -0.0358, 1.0, 0.4465],
                [-0.8777, -0.4567, 0.0878, 0.4465, 1.0],
            ]
        )

        # The smallest eigenvalue of A is -0.1244...
        # The smallest eigenvalue of Y is 1.088e-06
        Y = nearest_correlation_matrix(A, weights=W)

        # Matlab output has 4 digits, so atol is set to 1e-4 here
        assert np.allclose(Y, matlab_Y, atol=1e-4)


if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "-l"])
