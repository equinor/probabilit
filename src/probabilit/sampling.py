import warnings
import numbers
import pytensor
import numpy as np

from pytensor.graph.traversal import graph_inputs
from pytensor.compile import SharedVariable
from pytensor.tensor.random.type import RandomGeneratorType

SIZE = pytensor.shared(np.array(1.0, dtype="int64"), name="size")


def sample(
    nodes,
    size=1,
    *,
    random_state=None,
    method=None,
    correlator="composite",
    compile_kwargs: dict | None = None,
) -> np.ndarray:
    """Sample the current node and assign attribute `samples_` to nodes.

    Parameters
    ----------
    size : int, optional
        Number of samples to draw.
    random_state : np.random.Generator, int or None, optional
        A random state for the random number generator. The default is None.
    method : str, optional
        Sampling method, one of "lhs" (qmc.LatinHypercube), "halton"
        (qmc.Halton) or "sobol" (qmc.Sobol). The default is None, which
        means pseudo-random sampling.
    correlator : Correlator or str, optional
        A Correlator instance or a string in {"cholesky", "imanconover",
       "permutation", "composite"}. The default is "composite", which first
        runs Iman-Conover, then runs the Permutation correlator on the result.
    compile_kwargs : dict, optional
        Addional keyword arguments passed to pytensor.function

    Returns
    -------
    np.ndarray
        An array of samples, with length `size`.

    Examples
    --------
    >>> result = 2 * Distribution("expon", scale=1/3)
    >>> result.sample(random_state=0)
    array([0.53058301])
    >>> result.sample(size=5, random_state=0)
    array([0.53058301, 0.83728718, 0.6154821 , 0.52480077, 0.36736566])
    >>> result.sample(size=5, random_state=0, method="lhs")
    array([1.11212876, 0.273718  , 0.03808862, 0.5702549 , 0.83779147])

    Set a custom correlator by giving a Correlator type.
    The API of a correlator is:

        1. correlator = Correlator(correlation_matrix)
        2. X_corr = correlator(X_samples)  # shape (samples, variable)

    >>> from probabilit.correlation import Cholesky, ImanConover
    >>> from scipy.stats import pearsonr
    >>> a, b = Distribution("uniform"), Distribution("expon")
    >>> corr_mat = np.array([[1, 0.6], [0.6, 1]])
    >>> result = (a + b).correlate(a, b, corr_mat=corr_mat)

    >>> s = result.sample(25, random_state=0, correlator=Cholesky())
    >>> float(pearsonr(a.samples_, b.samples_).statistic)
    0.600000...
    >>> float(np.min(b.samples_)) # Cholesky does not preserve marginals!
    -0.35283...

    >>> s = result.sample(25, random_state=0, correlator=ImanConover())
    >>> float(pearsonr(a.samples_, b.samples_).statistic)
    0.617109...
    >>> float(np.min(b.samples_)) # ImanConover does preserve marginals
    0.062115...
    """
    if method is not None:
        warnings.warn("Only None method supporte so far")

    if not isinstance(size, numbers.Integral):
        raise TypeError("`size` must be a positive integer")
    if not size > 0:
        raise ValueError("`size` must be a positive integer")

    # Set batch size of RV nodes
    SIZE.set_value(size)

    # Seed shared RNGs
    sym_rngs = [
        v
        for v in graph_inputs(nodes if isinstance(nodes, tuple | list) else [nodes])
        if isinstance(v, SharedVariable) and isinstance(v.type, RandomGeneratorType)
    ]
    if len(sym_rngs) > 0:
        rngs = np.random.default_rng(random_state).spawn(len(sym_rngs))
        for sym_rng, rng in zip(sym_rngs, rngs, strict=True):
            sym_rng.set_value(rng, borrow=True)

    # TODO: Cache fn
    fn = pytensor.function([], nodes, **(compile_kwargs or {}))
    # fn.dprint(print_shape=True)
    return fn()

    d = self.num_distribution_nodes()  # Dimensionality of sampling

    # Draw a quantiles of random variables in [0, 1] using a method
    methods = {
        "lhs": sp.stats.qmc.LatinHypercube,
        "halton": sp.stats.qmc.Halton,
        "sobol": sp.stats.qmc.Sobol,
    }
    if method is None:  # Pseudo-random sampling
        random_state = check_random_state(random_state)
        quantiles = random_state.random((size, d))
    else:  # Quasi-random sampling
        sampler = methods[method.lower().strip()](d=d, rng=random_state)
        quantiles = sampler.random(n=size)

    return self.sample_from_quantiles(
        quantiles,
        correlator=correlator,
        gc_strategy=gc_strategy,
        random_state=random_state,
    )
