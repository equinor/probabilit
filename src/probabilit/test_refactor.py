# %%
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from probabilit.correlation import Cholesky

sampler = sp.stats.qmc.LatinHypercube(d=2, seed=42, scramble=True)
samples = sampler.random(n=100)
plt.scatter(*samples.T)

# %%
X = np.vstack(
    (sp.stats.triang(0.5).ppf(samples[:, 0]), sp.stats.gamma.ppf(samples[:, 1], a=1))
).T
float(sp.stats.pearsonr(*X.T).statistic)

# %%
plt.scatter(*X.T)
# %%
#
float(sp.stats.pearsonr(*X.T).statistic)
correlation_matrix = np.array([[1, 0.6], [0.6, 1]])
transform = Cholesky().set_target(correlation_matrix)
X_transformed = transform(X)
float(sp.stats.pearsonr(*X_transformed.T).statistic)
# %%
plt.scatter(*X_transformed.T)
