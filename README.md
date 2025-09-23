# probabilit

A small Python package for Monte Carlo modeling.

- User friendly API with a modeling language.
- Built on scipy and numpy.
- Supports composite distributions (e.g. the mean of a distribution can be a distribution).
- Supports Quasi-Monte Carlo sampling, e.g. Sobol, Halton and LHS.
- Supports inducing correlations with Iman-Conover.

## Modeling

The modeling API lets you create a computational graph, where each node is either a distribution, a constant or a transformation.
Once the method `.sample()` is called on a node, each ancestor node is sampled in turn.

**Example 1 - Height.**
What is the probability that a man is taller than a woman?

```pycon
>>> from probabilit.modeling import Distribution
>>> male_height = Distribution("norm", loc=176, scale=7.1)
>>> female_height = Distribution("norm", loc=162.5, scale=7.1)
>>> statistic = (male_height > female_height)
>>> samples = statistic.sample(999, random_state=0)
>>> float(samples.mean())
0.9039...

```

When `statistic` is sampled in the code above, the ancestor nodes `male_height` and `female_height` are sampled too.
In each node, results are stored in the `samples_` attribute.

```pycon
>>> import pandas as pd
>>> pd.Series(male_height.samples_).describe()
count    999.000000
mean     176.096916
std        7.326152
min      153.210671
25%      171.209087
50%      176.044660
75%      180.948085
max      201.216617
dtype: float64

```


**Example 2 - Bird survival.**
This example illustrates _composite distributions_, where the argument to one distribution is another distribution.
Suppose we have a distribution governing the number off eggs per bird nest for a certain species, and each hatched bird a survival probability.
What is the distribution of the number of birds that survive per nest?

```pycon
>>> eggs_per_nest = Distribution("poisson", mu=3)
>>> survivial_prob = 0.4
>>> survived = Distribution("binom", n=eggs_per_nest, p=survivial_prob)
>>> survived.sample(9, random_state=0) # Sample a few values only
array([2., 1., 1., 2., 2., 2., 2., 0., 0.])

```

**Example 3 - Mutual fund.**
Suppose we save 1200 units of money per year and that the yearly interest rate has a distribution `N(1.11, 0.15)`.
How much money will we have after 20 years?

```pycon
>>> saved_per_year = 1200
>>> returns = 0
>>> for year in range(20):
...     interest = Distribution("norm", loc=1.11, scale=0.15)
...     returns = returns * interest + saved_per_year
>>> samples = returns.sample(999, random_state=42)
>>> samples.mean(), samples.std()
(np.float64(76583.58738496085), np.float64(33483.2245611436))

```

## Low-level API

The low-level API contains numpy functions for working with random variables.
The two most important ones are (1) the `nearest_correlation_matrix` function and (2) the `ImanConover` class.

**Fixing user-supplied correlation matrices.**
The function `nearest_correlation_matrix` can be used to fix user-specified correlation matrices, which are often invalid.
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
The class `ImanConover` can be used to induce correlations on uncorrelated variables without changing the marginal distributions.
There's no guarantee that we're able to achieve the desired correlation structure, but in practice we often get close.

```pycon
>>> import scipy as sp
>>> from probabilit.correlation import ImanConover
>>> sampler = sp.stats.qmc.LatinHypercube(d=2, seed=42, scramble=True)
>>> samples = sampler.random(n=100)
>>> X = np.vstack((sp.stats.triang(0.5).ppf(samples[:, 0]),
...                sp.stats.gamma.ppf(samples[:, 1], a=1))).T

```

Now we can induce correlations:

```pycon
>>> float(sp.stats.pearsonr(*X.T).statistic)
0.065898...
>>> correlation_matrix = np.array([[1, 0.3], [0.3, 1]])
>>> transform = ImanConover().set_target(correlation_matrix)
>>> X_transformed = transform(X)
>>> float(sp.stats.pearsonr(*X_transformed.T).statistic)
0.279652...

```
