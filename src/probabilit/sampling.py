import warnings
import numbers
import pytensor
import numpy as np
from scipy.stats.qmc import Sobol, LatinHypercube, Halton

from pytensor.graph.fg import FunctionGraph
from pytensor.tensor.random.op import RandomVariable
from pytensor.graph.traversal import graph_inputs, ancestors
from pytensor.compile import SharedVariable
from pytensor.tensor.random.type import RandomGeneratorType
from pytensor.tensor import TensorVariable

from probabilit.icdf import icdf

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
    if not isinstance(size, numbers.Integral):
        raise TypeError("`size` must be a positive integer")
    if not size > 0:
        raise ValueError("`size` must be a positive integer")

    is_sequence = isinstance(nodes, list | tuple)
    if not is_sequence:
        nodes = [nodes]

    # TODO: choose correlator only at runtime
    if method is not None:
        nodes = replace_rvs_by_qmc_samples(nodes, method=method)

    # Seed shared RNGs
    sym_rngs = [
        v
        for v in graph_inputs(nodes)
        if isinstance(v, SharedVariable) and isinstance(v.type, RandomGeneratorType)
    ]

    if method is not None:
        # It's actually more strict, there should be one RV. Now there's 1 RNG for every RV, but it could change!
        assert len(sym_rngs) <= 1, "There should be at most 1 RNG when using QMC"

    if len(sym_rngs) > 0:
        rngs = np.random.default_rng(random_state).spawn(len(sym_rngs))
        for sym_rng, rng in zip(sym_rngs, rngs, strict=True):
            sym_rng.set_value(rng, borrow=True)

    # Set batch size of RV nodes
    SIZE.set_value(size)

    # TODO: Cache fn
    fn = pytensor.function([], nodes, **(compile_kwargs or {}))
    fn.dprint(print_shape=True)
    res = fn()
    return res if is_sequence else res[0]


def toposort_replace(nodes, replacements):
    fgraph = FunctionGraph(outputs=nodes, clone=False)
    toposort_index = {apply: i for i, apply in enumerate(fgraph.toposort())}

    def key_fn(x):
        replacee, replacement = x
        replacee_owner = replacee.owner
        if replacee_owner is None:
            return -1
        return toposort_index[replacee_owner]

    sorted_replacements = sorted(
        tuple(replacements.items()),
        key=key_fn,
    )
    fgraph.replace_all(sorted_replacements, import_missing=True)
    return fgraph.outputs


class QMCRV(RandomVariable):
    name = "QMC"
    signature = "()->(d)"
    __props__ = (*RandomVariable.__props__, "method")

    def __init__(self, *args, method, **kwargs):
        self.method = method
        match method:
            case "LatinHypercube":
                self.scipy_method = LatinHypercube
            case "Sobol":
                self.scipy_method = Sobol
            case "Halton":
                self.scipy_method = Halton
            case _:
                raise ValueError("Method not supported")
        super().__init__(*args, **kwargs)

    def _supp_shape_from_params(self, dist_params, param_shapes=None):
        [d] = dist_params
        return (d.squeeze(),)

    @classmethod
    def rng_fn(cls, rng, d, size=None):
        d = np.squeeze(d)
        if d.ndim > 0:
            raise ValueError(f"d must be a scalar, got {d}")

        default_size = size in (None, ())
        if default_size:
            qmc_size = 1
        elif len(size) == 1:
            qmc_size = size
        else:
            qmc_size = int(np.prod(size))

        qmc_draws = self.scipy_method(d=d, rng=rng).random(n=qmc_size)
        if default_size:
            return qmc_draws.squeeze(0)
        elif len(size) == 1:
            return qmc_draws
        else:
            return qmc_draws.reshape((*size, d))


qmc_sobol = QMCRV(method="Sobol")
qmc_lhs = QMCRV(method="LatinHypercube")
qmc_halton = QMCRV(method="Halton")


def replace_rvs_by_qmc_samples(nodes: list[TensorVariable], method: str):
    """
    Replace random variables in a graph with quasi-Monte Carlo samples.

    nodes = [(z := normal(x := normal, y := exponential)]
    len_rvs = 3
    qmc_samples = QMC(size=(Size, len(RVs)))
    new_x = icdf_normal(qmc_samples[:, 0], 0, 1)
    new_y = icdf_normal(qmc_samples[:, 1], 1)
    new_z = icdf_normal(qmc_samples[:, 2], new_x, new_y)

    """
    rvs_in_graph = [
        v
        for v in ancestors(nodes if isinstance(nodes, list | tuple) else nodes)
        if (v.owner is not None and isinstance(v.owner.op, RandomVariable))
    ]
    # d = pt.prod([rv.size // SIZE for rv in rvs_in_graph])
    d = len(rvs_in_graph)

    # Draw a quantiles of random variables in [0, 1] using a method
    match method.lower():
        case "lhs":
            rv_op = qmc_lhs
        case "sobol":
            rv_op = qmc_sobol
        case "halton":
            rv_op = qmc_halton
        case _:
            raise ValueError(f"Unrecognized method {method}")

    qmc_draws = rv_op(d, size=SIZE)

    replacements = {
        rv: icdf(
            rv.owner.op,
            qmc_draws[..., i],
            *rv.owner.op.dist_params(rv.owner),
        )
        for i, rv in enumerate(rvs_in_graph)
    }

    return toposort_replace(nodes, replacements)
