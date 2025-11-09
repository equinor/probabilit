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
from probabilit.utils import extract_shape_of_nodes


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
    correlator=None,
    compile_kwargs: dict | None = None,
    dprint_sample_fn: bool = False,
) -> np.ndarray | list[np.ndarray]:
    """Sample nodes."""
    if correlator is not None:
        raise ValueError(
            "correlator is now implemented as a function during graph construction."
        )
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
    rv_shapes = extract_shape_of_nodes(rvs_in_graph)
    rv_sizes = [pt.prod(rv_shape) for rv_shape in rv_shapes]

    d_fn = pytensor.function([], pt.sum(rv_sizes), mode="FAST_COMPILE")
    d = int(d_fn())

    # Replace random variables by icdf of QMC samples
    qmc_samples = pt.tensor("qmc_samples", dtype="float64", shape=(None, d))
    # Use a dummy core_qmc_samples that will later be replaced
    # by the vectorized QMC samples with extra size dimension
    core_qmc_samples = pt.tensor("core_qmc_samples", dtype="float64", shape=(d,))
    replacements = {}
    counter = 0
    for rv, rv_shape, rv_size in zip(rvs_in_graph, rv_shapes, rv_sizes, strict=True):
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
    qmc_nodes_fn = pytensor.function(
        [qmc_samples],
        qmc_nodes,
        **(compile_kwargs or {}),
        # For constant functions qmc_samples won't be used
        on_unused_input="ignore",
    )
    assert not any(
        apply.out
        for apply in qmc_nodes_fn.maker.fgraph.apply_nodes
        if isinstance(apply.op, RandomVariable)
    )
    if dprint_sample_fn:
        qmc_nodes_fn.dprint(print_type=True, print_memory_map=True)

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
