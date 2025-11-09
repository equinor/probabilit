import numbers

import numpy as np
import pytensor
import pytensor.tensor as pt
from pytensor.graph.fg import FunctionGraph
from pytensor.graph.replace import vectorize_graph
from pytensor.tensor import TensorVariable
from pytensor.tensor.random.op import RandomVariable
from scipy.stats import uniform
from scipy.stats.qmc import Halton, LatinHypercube, Sobol

from probabilit.quantiles import quantile


def toposort_replace(
    fgraph: FunctionGraph, replacements: dict[TensorVariable:TensorVariable]
) -> None:
    toposort_index = {apply: i for i, apply in enumerate(fgraph.toposort())}

    def key_fn(x):
        replacee, _replacement = x
        replacee_owner = replacee.owner
        if replacee_owner is None:
            return -1
        return toposort_index[replacee_owner]

    sorted_replacements = sorted(
        tuple(replacements.items()),
        key=key_fn,
        reverse=True,
    )
    fgraph.replace_all(sorted_replacements, import_missing=True)


def sample(
    nodes,
    size=1,
    *,
    random_state=None,
    method=None,
    correlator="composite",
    compile_kwargs: dict | None = None,
) -> np.ndarray | list[np.ndarray]:
    """Sample nodes.

    # TODO: choose correlator only when this function is called
    """
    if not isinstance(size, numbers.Integral):
        raise TypeError("`size` must be a positive integer")
    if not size > 0:
        raise ValueError("`size` must be a positive integer")

    is_sequence = isinstance(nodes, list | tuple)
    if not is_sequence:
        nodes = [nodes]

    # Get dimensionality of random nodes
    fg = FunctionGraph(outputs=nodes, clone=True, copy_inputs=False)
    rvs_in_graph = [
        apply.out for apply in fg.toposort() if isinstance(apply.op, RandomVariable)
    ]
    d_fn = pytensor.function([], [rv.shape for rv in rvs_in_graph], mode="FAST_COMPILE")
    rvs_shapes = d_fn()
    rv_sizes = [int(np.prod(rv_shape)) for rv_shape in rvs_shapes]
    d = sum(rv_sizes)

    # Replace random variables by icdf of QMC samples
    qmc_samples = pt.tensor("qmc_samples", dtype="float64", shape=(None, int(d)))
    # Use a dummy core_qmc_samples that will later be replaced
    # by the vectorized QMC samples with extra size dimension
    core_qmc_samples = pt.tensor("core_qmc_samples", dtype="float64", shape=(int(d),))
    replacements = {}
    counter = 0
    for rv, rv_shape, rv_size in zip(rvs_in_graph, rvs_shapes, rv_sizes, strict=True):
        rv_quantiles = core_qmc_samples[counter : counter + rv_size].reshape(rv_shape)
        replacements[rv] = quantile(
            rv.owner.op,
            rv_quantiles,
            *rv.owner.op.dist_params(rv.owner),
        ).astype(rv.dtype)
        counter += rv_size

    toposort_replace(fg, replacements)
    core_qmc_nodes = fg.outputs
    qmc_nodes = vectorize_graph(core_qmc_nodes, replace={core_qmc_samples: qmc_samples})

    # Compile function to get QMC nodes from QMC samples
    # TODO: Cache fn for same nodes / methods
    qmc_nodes_fn = pytensor.function([qmc_samples], qmc_nodes, **(compile_kwargs or {}))
    assert not any(
        apply.out
        for apply in qmc_nodes_fn.maker.fgraph.apply_nodes
        if isinstance(apply.op, RandomVariable)
    )
    # qmc_nodes_fn.dprint(print_type=True, print_memory_map=True)

    if method is None:
        qmc_samples_np = uniform.rvs(size=(size, d), random_state=random_state)
    else:
        qmc_samples_np = {
            "sobol": Sobol,
            "lhs": LatinHypercube,
            "halton": Halton,
        }[method](d=d, rng=random_state).random(n=size)

    qmc_nodes_np = qmc_nodes_fn(qmc_samples_np)
    return qmc_nodes_np if is_sequence else qmc_nodes_np[0]
