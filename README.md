# probabilit

A Python package for Monte Carlo sampling.

## Low-level API

The low-level API contains Numpy functions for working with random variables.
The two most important ones are (1) the `nearest_correlation_matrix` function and and (2) the `ImanConover` class.

**Fixing user-supplied correlation matrices.**
The function `nearest_correlation_matrix` can be used to fix user-specified correlation matrices, which are often not valid.
Below a user has specified some correlations, but the resulting correlation matrix has a negative eigenvalue and is not positive definite.

```pycon
>>> import numpy as np
>>> from probabilit.correlation import nearest_correlation_matrix
>>> X = np.array([[1, 0.9, 0],
...               [0.9, 1, 0.8],
...               [0, 0.8, 1]])
>>> np.linalg.eigvals(X) # Not a valid correlation matrix
array([-0.20415946,  1.        ,  2.20415946])
>>> nearest_correlation_matrix(X)
array([[1.        , 0.77523696, 0.07905637],
       [0.77523696, 1.        , 0.69097837],
       [0.07905637, 0.69097837, 1.        ]])
>>> np.linalg.eigvals(nearest_correlation_matrix(X))
array([2.07852823e+00, 9.21470108e-01, 1.66710188e-06])

```

**Inducing correlations on samples.**
The class `ImanConover` can be used to induce correlations on uncorrelated variables.
There's not guarantee that we're able to achieve the desired correlation structure, but in practice we can often get close.

```pycon
>>> import scipy as sp
>>> from probabilit.iman_conover import ImanConover
>>> sampler = sp.stats.qmc.LatinHypercube(d=2, seed=42, scramble=True)
>>> samples = sampler.random(n=100)
>>> X = np.vstack((sp.stats.triang(0.5).ppf(samples[:, 0]),
...                sp.stats.gamma.ppf(samples[:, 1], a=1))).T

```

Now we can induce correlations:

```pycon
>>> format(sp.stats.pearsonr(*X.T).statistic, ".8f")
'0.06589800'
>>> correlation_matrix = np.array([[1, 0.3], [0.3, 1]])
>>> transform = ImanConover(correlation_matrix)
>>> X_transformed = transform(X)
>>> format(sp.stats.pearsonr(*X_transformed.T).statistic, ".8f")
'0.27965287'
```

## Modeling


```pycon
>>> from probabilit.modeling import Distribution
>>> a = Distribution("norm", loc=2, scale=1)
>>> a.sample(5, random_state=42)

```