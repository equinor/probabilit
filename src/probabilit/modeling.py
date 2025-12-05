"""
Modeling
--------

Probabilit lets the user perform Monte-Carlo sampling using a high-level
modeling language that creates a lazy computational graph.
When a node in the graph is sampled, all ancestor nodes are sampled in turn
and samples are propagated down in the graph, from parent nodes to child nodes.

For instance, to compute the shipping cost of a box where we are uncertain
about the measurements:

>>> rng = np.random.default_rng(42)
>>> box_height = Distribution("norm", loc=0.5, scale=0.01)
>>> box_width = Distribution("norm", loc=1, scale=0.01)
>>> box_depth = Distribution("norm", loc=0.8, scale=0.01)
>>> box_volume = box_height * box_width * box_depth
>>> price_per_sqm = 50
>>> price = box_volume * price_per_sqm
>>> samples = price.sample(999, random_state=rng)
>>> float(np.mean(samples))
20.00876430957...

Distributions are built on top of scipy, so "norm" refers to the name of the
normal distribution as given in `scipy.stats`, and the arguments to the
distribution must match those given by `scipy.stats.norm` (here `loc` and
`scale`). Scipy supports almost one hundred probability distributions,
and all of them are also available in probabilit.


Composite distributions
-----------------------

Here is another example demonstrating composite distributions, where an
argument to one distribution is another distribution:

>>> eggs_per_nest = Distribution("poisson", mu=3)
>>> survivial_prob = Distribution("beta", a=10, b=15)
>>> survived = Distribution("binom", n=eggs_per_nest, p=survivial_prob)
>>> survived.sample(9, random_state=rng)
array([0., 1., 2., 0., 3., 1., 1., 0., 2.])


The modeling language
---------------------

To understand and examine the modeling language, we can perform computations
using constants. The computational graph carries out arithmetic operations
when the model is sampled. Mixing numbers with nodes in an expression is allowed,
but at least one expression or term must be a probabilit class instance:

>>> a = Constant(1)
>>> (a * 3 + 5).sample(5, random_state=rng)
array([8, 8, 8, 8, 8])
>>> Add(10, 5, 5).sample(5, random_state=rng)
array([20, 20, 20, 20, 20])

Let us build a more complicated expression with distributions:

>>> a = Distribution("norm", loc=5, scale=1)
>>> b = Distribution("expon", scale=1)
>>> expression = a**b + a * b + 5 * b

It is possible to convert distributions to scipy objects

>>> normal = Distribution("norm", loc=Constant(2)**3, scale=1)
>>> dist = normal.to_scipy()
>>> dist.rvs(5, random_state=42)
array([8.49671415, 7.8617357 , 8.64768854, 9.52302986, 7.76584663])

Every unique node in this expression can be found by calling `.nodes()`:

>>> for node in sorted(set(expression.nodes()), key=lambda n:n._id):
...     print(node)
Distribution("norm", loc=5, scale=1)
Distribution("expon", scale=1)
Power(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1))
Multiply(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1))
Add(Power(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1)), Multiply(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1)))
Constant(5)
Multiply(Distribution("expon", scale=1), Constant(5))
Add(Add(Power(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1)), Multiply(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1))), Multiply(Distribution("expon", scale=1), Constant(5)))

Sampling any node is done by calling the `.sample()` method:

>>> expression.sample(5, random_state=rng)
array([ 5.04060084,  5.19163254, 13.09590433, 20.23600678,  4.24147296])

Sampling the expression has the side effect that `.samples_` is populated on
*every* ancestor node in the expression, for instance:

>>> a.samples_
array([3.9254551 , 6.0788841 , 4.68923246, 3.48160505, 4.26149282])

Here is an even more complex expression, showcasing some mathematical functions:

>>> a = Distribution("norm", loc=0, scale=1)
>>> b = Distribution("norm", loc=0, scale=2)
>>> c = Distribution("norm", loc=0, scale=3)
>>> expression = a*a - Add(a, b, c) + Abs(b)**Abs(c) + Exp(1 / Abs(c))
>>> expression.sample(5, random_state=rng)
array([ -4.9730559 , 103.74625257,   3.45199868,  14.50692883,
        31.07178136])

Nodes are hashable and can be used in sets, so __hash__ and __eq__ must both
be defined. Therefore we cannot use `==` for modeling; equality in that context
has another meaning. Use the Equal node instead. This is only relevant in cases
when equality operators is part of a model. For distribution defined on real
numbers (e.g. Normal) equality does not make sense since the probability that two
floats are equal is zero.

>>> dice1 = Distribution("uniform", loc=1, scale=6) // 1
>>> dice2 = Distribution("uniform", loc=1, scale=6) // 1
>>> equal_result = Equal(dice1, dice2)
>>> float(equal_result.sample(999, random_state=42).mean())
0.166...

Empirical distributions may also be used. They wrap np.quantile and take the
same arguments. For instance, to sample a dice we can pass `closest_observation`:

>>> dice = EmpiricalDistribution([1, 2, 3, 4, 5, 6], method="closest_observation")
>>> dice.sample(9, random_state=42)
array([2, 6, 4, 4, 1, 1, 1, 5, 4])

To sample from a non-parametric distribution defined by data,
similar to a kernel density estimate:

>>> cost = EmpiricalDistribution([200, 200, 300, 250, 225])
>>> cost.sample(9, random_state=42)
array([212.45401188, 290.14286128, 248.19939418, 234.86584842,
       200.        , 200.        , 200.        , 273.23522915,
       235.11150117])


Correlations
------------
To correlate samples, use the `.correlate()` method on any node that is a
descendant of the nodes you wish to correlate:

>>> from scipy.stats import pearsonr
>>> a, b = Distribution("uniform"), Distribution("expon")
>>> corr_mat = np.array([[1, 0.5], [0.5, 1]])
>>> s = (a + b).correlate(a, b, corr_mat=corr_mat).sample(99, random_state=0)

We can verify that the correlation is close to the desired value of 0.5:

>>> float(pearsonr(a.samples_, b.samples_).statistic)
0.49245...


Multivariate distributions
--------------------------
Support for multivariate distributions (MVD) is implemented, but is limited:

  1. the MVD must be a source node (its arguments cannot be other distributions)
  2. its return values *must* be unpacked as marginals (slices)
  3. only pseudo-random sampling is possible (LHS, Sobol, etc. is ignored)

For instance, to create a Dirichlet distribution, we must unpack it as follows:

>>> d1, d2 = MultivariateDistribution("dirichlet", alpha=[1, 2])
>>> d1
MarginalDistribution(Distribution("dirichlet", alpha=[1, 2]), d=0)

Since the Direchlet distribution is defined on an (n-1) dimensional simplex,
the sum of the marginals is always 1. We can check this by computing:

>>> (d1 + d2).sample(5, random_state=0)
array([1., 1., 1., 1., 1.])

Each marginal has values between 0 and 1:

>>> d2.samples_.round(3)
array([0.559, 0.324, 0.284, 0.524, 0.599])

Here is an example with a multivariate normal distribution:

>>> cov = np.array([[1, 0.5], [0.5, 1]])
>>> n1, n2 = MultivariateDistribution("multivariate_normal", mean=[1, 2], cov=cov)
>>> n1.sample(5, random_state=0)
array([0.72058767, 3.13703525, 2.38930155, 1.50866787, 0.77018653])
>>> (n1 + n2).sample(5, random_state=0)
array([2.52848604, 5.31650094, 5.20076878, 4.06217341, 1.40748585])


Samplers
--------

By default sampling uses pseudo-random numbers. To use e.g. latin hybercube
sampling, pass the `method` argument into `.sample()`.

>>> dice = EmpiricalDistribution([1, 2, 3, 4, 5, 6], method="closest_observation")
>>> float(dice.sample(9, random_state=1, method="lhs").mean())
3.222...
>>> float(dice.sample(9, random_state=1, method=None).mean())
1.888...

To retain more control, use the `sample_from_quantiles` method directly instead.
The quantiles are passed to the inverse CDF (percent point function) when sampling:

>>> from scipy.stats.qmc import LatinHypercube
>>> d = expression.num_distribution_nodes()
>>> hypercube = LatinHypercube(d=d, rng=rng, optimization="random-cd")
>>> hypercube_samples = hypercube.random(5) # Draw 5 samples
>>> expression.sample_from_quantiles(hypercube_samples)
array([ 8.2746582 ,  1.47340322,  1.71799415,  6.21188211, 41.63293414])


Garbage collection
------------------
By default the `.samples_` attribute is set on every ancestor when sampling.
This can lead to memory overhead in large graphs. A garbage collection
strategy can be set to keep the memory constrained:

Setting `gc_strategy=[]` removes `.samples_` on all nodes except the final node:

>>> a = Distribution("norm")
>>> intermediate_result = (a + a)**2 - a
>>> final_result = Exp(intermediate_result)
>>> final_result.sample(3, random_state=42, gc_strategy=[]).round(3)
array([2.0730000e+00, 1.0532374e+04, 2.4920000e+00])

Verif that `.samples_` was not set on ancestors:

>>> hasattr(a, "samples_")
False
>>> hasattr(intermediate_result, "samples_")
False

To keep intermediate results on some nodes, pass them into `gc_strategy`:

>>> final_result.sample(3, random_state=42, gc_strategy=[a]).round(3)
array([2.0730000e+00, 1.0532374e+04, 2.4920000e+00])
>>> hasattr(a, "samples_")
True
>>> hasattr(intermediate_result, "samples_")
False

When passing `gc_strategy=None` (the default), probabilit will not do any GC.


Functions
---------

If you have a function that is not an arithmetic expression, you can still
Monte-Carlo simulate through it with the `scalar_transform` decorator, which
will pass each sample through the computation node in a loop:

>>> def function(a, b):
...     if a > 0:
...         return a * b
...     else:
...         return 0
>>> function = scalar_transform(function)

Now we can create a computational graph:

>>> a = Distribution("norm", loc=0, scale=1)
>>> b = Distribution("norm", loc=0, scale=2)
>>> expression = function(a, b) # Function is not actually evaluated here

Now sample 'through' the function:

>>> expression.sample(5, random_state=rng)
array([0.        , 0.        , 0.45555522, 0.        , 0.        ])

This type of sampling is slower, since it passes each sample in turn and
is not vectorized. The advantage is that the function can be a complex
simulation model, or read and write files, etc.

"""

import operator
import functools
import numpy as np
import scipy as sp
import numbers
from scipy import stats
import abc
import itertools
import networkx as nx
from scipy._lib._util import check_random_state
from probabilit.correlation import (
    nearest_correlation_matrix,
    ImanConover,
    Cholesky,
    Composite,
    Permutation,
)
from probabilit.utils import build_corrmat, zip_args
from probabilit.garbage_collector import GarbageCollector
import copy


# =============================================================================
# FUNCTIONS
# =============================================================================


def python_to_prob(argument):
    """Convert basic Python types to probabilit types."""
    if isinstance(argument, numbers.Number):
        return Constant(argument)
    elif isinstance(argument, Node):
        return argument
    else:
        raise ValueError(f"Type not compatible with probabilit: {argument}")


# =============================================================================
# COMPUTATIONAL GRAPH AND MODELING LANGUAGE
# =============================================================================
#
# There are three main types of Node instances, they are:
#   - Constant:      numbers like 2 or 5.5, which are always source nodes in the graph
#   - Distribution:  typically source nodes, but can have parents if composite
#   - Transform:     arithmetic operations like + or **, or general functions,
#                    must always have parents
#
# An expression such as:
#
#  mu = Distribution("norm", loc=0, scale=1)
#  normal  = Distribution("norm", loc=mu, scale=1)
#  result = mu + normal - 2
#
# Is represented by a graph such as:
#
#        mu --> normal
#         \       /
#          \     /
#           v   v
#             +         2
#              \       /
#               \     /
#                v   v
#                  -
#               (result)
#
# Where:
#   * "mu" is a Distribution and a source node
#   * "normal" is a Distribution, but not a source node (has "mu" as parent)
#   * "+" is a Transform (has "mu" and "normal" as parents)
#   * "2" is a Constant and a source node
#   * "-" (the result) is a Transform (has "+" and "2" as parents)
#
# Some further terminology:
#   * The _ancestors_ of node "-" are {"+", "2", "mu", "normal"}
#   * result.sample() samples the expression "mu + normal - 2" by propagating
#     through the graph, parents first. More specifically, each category of nodes
#     (Constant, Transform and Distribution) have their own internal _sample methods
#     and we propagate the sampling of the graph in the following way:
#      (1) "mu" is sampled first by calling the internal _sample method
#          of the Distribution class.
#      (2) Then "normal" is sampled using the samples of "mu" and the same
#          _sample method as in (1).
#      (3) Thereafter, "+" is sampled using the samples of both "mu" and "normal",
#          and the internal _sample method of the Transform class.
#      (4) Finally, "-" is sampled last, using the samples of "+" and "2",
#          and the same _sample method as in (3).
#     Since the node "2" has no parents (as is the case for all Constant nodes),
#     it can be sampled at any time prior to sampling "-".
#     A topological ordering of the graph makes sure that parents are always
#     sampled first, and the internal _sample methods of the Distributon and
#     the Transform class rely on the samples of parent nodes (in the case where
#     parents exist).


class Node(abc.ABC):
    """A node in the computational graph."""

    id_iter = itertools.count()  # Every node gets a unique ID

    def __init__(self):
        self._id = next(self.id_iter)
        self._correlations = []

    def __eq__(self, other):
        if not isinstance(other, Node):
            return NotImplemented
        # Needed for set() to work on Node. Equality in models must use Equal()
        return self._id == other._id

    def __hash__(self):
        return self._id

    def copy(self):
        """Copy the Node, including the entire graph above it.

        Examples
        --------
        >>> mu = Distribution("norm", loc=0, scale=1)
        >>> a = Distribution("norm", loc=mu, scale=Constant(0.5))
        >>> a2 = a.copy()
        >>> a is a2
        False
        >>> a2.kwargs["loc"] == a.kwargs["loc"]
        True
        >>> a2.kwargs["loc"] is a.kwargs["loc"]
        False
        """
        # Map from ID to new object copy
        id_to_new = dict()

        def update(item):
            """Given an item, use the ID to map to new object copy."""
            if isinstance(item, Node):
                return id_to_new[item._id]
            return copy.deepcopy(item)

        # Go through nodes in topologial order, guaranteeing that parents
        # have always been copied over to new graph when children are copied.
        for node in nx.topological_sort(self.to_graph()):
            # Copy the node itself and update the mapping
            copied = copy.copy(node)  # Copy a node WITHOUT copying the graph
            id_to_new[copied._id] = copied

            # Copy samples if they exist
            if hasattr(copied, "samples_"):
                copied.samples_ = np.copy(copied.samples_)

            copied._correlations = copy.deepcopy(copied._correlations)

            # Now that the node has been updated, update references to parents
            # to point to Nodes in the new copied graph instead of the old one.
            if isinstance(copied, (AbstractDistribution, ScalarFunctionTransform)):
                copied.args = tuple(update(arg) for arg in copied.args)
                copied.kwargs = {k: update(v) for (k, v) in copied.kwargs.items()}
            elif isinstance(copied, (VariadicTransform, BinaryTransform)):
                copied.parents = tuple(update(p) for p in copied.parents)
            elif isinstance(copied, UnaryTransform):
                copied.parent = update(copied.parent)
            elif isinstance(copied, Constant):
                copied.value = update(copied.value)
            else:
                pass

        return id_to_new[self._id]

    def nodes(self):
        """Yields `self` and all ancestors using depth-first-search.

        Examples
        --------
        >>> expression = Distribution("norm") -  2**Constant(2)
        >>> for node in expression.nodes():
        ...     print(node)
        Subtract(Distribution("norm"), Power(Constant(2), Constant(2)))
        Power(Constant(2), Constant(2))
        Constant(2)
        Constant(2)
        Distribution("norm")
        """
        queue = [(self)]
        while queue:
            yield (node := queue.pop())
            queue.extend(node.get_parents())

    def num_distribution_nodes(self):
        """Number of unique ancestor nodes that are distribution nodes."""
        return sum(
            1 for node in set(self.nodes()) if isinstance(node, AbstractDistribution)
        )

    def sample(
        self,
        size=1,
        *,
        random_state=None,
        method=None,
        correlator="composite",
        gc_strategy=None,
    ):
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
        gc_strategy : None or list, optional
            If None, no garbage collection is performed and the attribute
            `.samples_` will be set on all nodes. If an empty list [], then
            all nodes except the final one will be garbage collected. If a list
            of Node instances, then those will be garbage collected.

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
        0.6...
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

    def sample_from_quantiles(
        self, quantiles, *, correlator="composite", gc_strategy=None, random_state=None
    ):
        """Use samples from an array of quantiles in [0, 1] to sample all
        distributions. The array must have shape (dimensionality, num_samples)."""
        assert nx.is_directed_acyclic_graph(self.to_graph())
        size, n_dim = quantiles.shape
        assert n_dim == self.num_distribution_nodes()

        # Get the correct correlator class based on strings
        CORRELATOR_MAP = {
            "imanconover": ImanConover(),
            "cholesky": Cholesky(),
            "permutation": Permutation(random_state=random_state),
            "composite": Composite(random_state=random_state),
        }
        if isinstance(correlator, str):
            correlator = correlator.lower().strip()
            valid_corrs = set(CORRELATOR_MAP.keys())
            if correlator not in valid_corrs:
                raise ValueError(f"`{correlator=}` not in {valid_corrs}")
            correlator = CORRELATOR_MAP[correlator]  # Map to instance

        # Prepare columns of quantiles, one column for each Distribution
        columns = iter(list(quantiles.T))

        # Clear any samples that might exist in the graph
        for node in set(self.nodes()):
            if hasattr(node, "samples_"):
                delattr(node, "samples_")

        # Set up garbage collection
        gc = GarbageCollector(strategy=gc_strategy).set_sink(self)

        # Keep track of all nodes that are sampled and later garbarge-collected.
        # If we do not keep track of these then they will be sampled twice.
        # We will skip sampling a node if either (1) samples_ is set or (2)
        # the node has previously been sampled and garbage collected.
        garbage_collected = set()

        def topo_sample(G, gc, garbage_collected):
            """Sample nodes in a graph G in topological order.

            Both the arguments `gc` (garbage collector) and `garbage_collected`
            will be updated in place (mutated).
            """

            for node in nx.topological_sort(G):
                # Skip if samples already exists or the node was sampled previously
                if hasattr(node, "samples_") or node in garbage_collected:
                    continue
                elif isinstance(node, Constant):
                    node.samples_ = node._sample(size=size)  # Draw constants
                elif isinstance(node, AbstractDistribution):
                    node.samples_ = node._sample(q=next(columns))  # Sample distr
                elif isinstance(node, Transform):
                    node.samples_ = node._sample()  # Propagate through transform
                else:
                    raise TypeError(
                        "Node must be Constant, AbstractDistribution or Transform."
                    )

                is_numeric = (node.samples_ is not None) and np.issubdtype(
                    node.samples_.dtype, np.number
                )
                if is_numeric and not np.all(np.isfinite(node.samples_)):
                    msg = f"Sampling gave non-finite values: {node}\n{node.samples_}"
                    raise ValueError(msg)

                # Tell the garbage collector that we sampled this node.
                # If the reference counter reaches zero (a parent has no unsampled
                # children), then the `.samples_` attribute of the parent might
                # be deleted (if the garbage collection strategy allows it).
                garbage_collected.update(gc.decrement_and_delete(node))

        # If there are no correlations to induce, then we can simply go through
        # the graph in topological order and sample it.
        # If there are correlations, then we must first sample up until and
        # including nodes that are to be correlated, then correlate them, then
        # sample the remaining graph. For instance, consider the graph:
        # A ----> B ---> [C] ---> D
        #                         |
        #                         v
        # E ---> [F] ---> G ----> result
        # If nodes C and F are to be correlated, then we sample A -> B -> C,
        # followed by E -> F, once we have samples on nodes C and F we
        # can correlate those permuting the order of the samples (ImanConover).
        # Finally we keep sampling D -> G -> result.

        G = self.to_graph()

        # Go through all ancestor nodes and create a list [(var, corr), ...]
        # that contains all correlations we must induce
        correlations = []
        for node in set(self.nodes()):
            if hasattr(node, "_correlations"):
                correlations.extend(node._correlations)

        variable_sets = [set(variables) for (variables, _) in correlations]
        # Map all variables to integers to associate them with a column
        corr_variables = list(functools.reduce(set.union, variable_sets, set()))
        # Ensure consistent ordering for reproducible results
        corr_variables = sorted(corr_variables, key=lambda n: n._id)

        # Check that the set of variables that the user wants to correlate
        # are allowed. The condition is that each variable and its ancestors
        # must be disjoint sets. For instance, in the graph A -> B we cannot
        # correlate A and B. In the graph A <- B -> C we cannot correlate
        # A and C. In general we can only correlate nodes whose correlation
        # cannot potentially be determined already from the graph structure.
        seen = set()
        for variable in corr_variables:
            var_plus_ancestors = set(variable.nodes())

            # If the variable, or any ancestor, has already been seen
            if seen.intersection(var_plus_ancestors):
                msg = f"Cannot correlate node: {variable}\n"
                msg += "This variable is an ancestor of more than one variables\n"
                msg += "that you wish to correlate. But this relationship can\n"
                msg += "potentially already induce a correlation.\n"
                msg += (
                    "For instance, in the graph A -> B you cannot correlate A and B.\n"
                )
                msg += "In the graph A <- B -> C you cannot correlate A and C."
                raise ValueError(msg)
            else:
                seen.update(var_plus_ancestors)

        # Check that no correlation has been specified twice
        variable_sets = [set(variables) for (variables, _) in correlations]
        for vars1, vars2 in itertools.combinations(variable_sets, 2):
            common = vars1.intersection(vars2)
            if len(common) > 1:
                raise ValueError(f"Correlations specified more than once: {common}")

        # Sample up untill nodes that we induce correlations on. In this graph:
        # A ----> B ---> [C] ---> D
        #                         |
        #                         v
        # E ---> [F] ---> G ----> result
        # this means sampling up until and including C and F.
        for variable in corr_variables:
            ancestors = G.subgraph(nx.ancestors(G, variable).union({variable}))
            topo_sample(ancestors, gc=gc, garbage_collected=garbage_collected)

        # Map to correlations
        var_to_int = {v: i for (i, v) in enumerate(corr_variables)}
        correlations = [
            (tuple(var_to_int[var] for var in variables), corrmat)
            for (variables, corrmat) in correlations
        ]

        # If there are any correlations to induce, do so
        if correlations:
            # Induce correlations
            correlation_matrix = build_corrmat(correlations)
            correlation_matrix = nearest_correlation_matrix(correlation_matrix)

            # Set the target (goal) correlation matrix
            correlator = correlator.set_target(correlation_matrix)

            # Concatenate samples, correlate them (shift rows in each col), then re-assign
            samples_input = np.vstack([var.samples_ for var in corr_variables]).T
            samples_ouput = correlator(samples_input)
            for var, sample in zip(corr_variables, samples_ouput.T):
                var.samples_ = np.copy(sample)

        # Sample all the way to the end. In this graph:
        # A ----> B ---> [C] ---> D
        #                         |
        #                         v
        # E ---> [F] ---> G ----> result
        # this would mean sampling from D and G.
        topo_sample(self.to_graph(), gc=gc, garbage_collected=garbage_collected)
        return self.samples_

    def _is_initial_sampling_node(self):
        """A node is an initial sample node iff:
        (1) It is a Distribution
        (2) None of its ancestors are Distributions (all are Constant/Transform)"""

        is_distribution = isinstance(self, AbstractDistribution)
        ancestors = set(self.nodes()) - set([self])
        ancestors_distr = any(
            isinstance(node, AbstractDistribution) for node in ancestors
        )
        return is_distribution and not ancestors_distr

    def correlate(self, *variables, corr_mat):
        """Store correlations on variables.

        When `.correlate(*variables)` is called on a node, the variables must
        be ancestors of that node. The order of the variables should match the
        order of the rows/columns in the correlation matrix.

        Examples
        --------
        >>> a = Distribution("expon", 1)
        >>> b = Distribution("poisson", 1)
        >>> corr_mat = np.array([[1, 0.5], [0.5, 1]])

        Correlations can be induced on any child node.

        >>> import scipy as sp
        >>> result = (a + b)
        >>> result = result.correlate(a, b, corr_mat=corr_mat)
        >>> _ = result.sample(999, random_state=0)
        >>> float(sp.stats.pearsonr(a.samples_, b.samples_).statistic)
        0.483035...

        """
        assert corr_mat.ndim == 2
        assert corr_mat.shape[0] == corr_mat.shape[1]
        assert corr_mat.shape[0] == len(variables)
        assert len(variables) == len(set(variables))
        nodes = set(self.nodes())
        for var in variables:
            if var not in nodes:
                raise ValueError(f"{var} is not an ancestor of {self}")

        valid_diag = np.allclose(np.diag(corr_mat), 1)
        valid_entries = np.allclose(np.clip(corr_mat, -1, 1), corr_mat)
        if not (valid_diag and valid_entries):
            raise ValueError(
                "Correlation matrix must have entries in [-1, 1] and 1 on diagonal."
            )

        self._correlations.append((list(variables), np.copy(corr_mat)))
        return self

    def to_graph(self):
        """Convert the computational graph to a networkx MultiDiGraph."""
        nodes = list(self.nodes())

        # Special case if there is only one node
        if len(nodes) == 1:
            G = nx.MultiDiGraph()
            G.add_node(self)
            return G

        # General case
        edge_list = [
            (ancestor, node)
            for node in nodes
            for ancestor in node.get_parents()
            if not node.is_source_node
        ]
        return nx.MultiDiGraph(edge_list)


class OverloadMixin:
    """Overloads dunder (double underscore) methods for easier modeling."""

    def __add__(self, other):
        return Add(self, other)

    def __radd__(self, other):
        return Add(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __rmul__(self, other):
        return Multiply(self, other)

    def __floordiv__(self, other):
        return FloorDivide(self, other)

    def __rfloordiv__(self, other):
        return FloorDivide(other, self)

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __mod__(self, other):
        return Mod(self, other)

    def __rmod__(self, other):
        return Mod(other, self)

    def __sub__(self, other):
        return Subtract(self, other)

    def __rsub__(self, other):
        return Subtract(other, self)

    def __pow__(self, other):
        return Power(self, other)

    def __rpow__(self, other):
        return Power(other, self)

    def __neg__(self):
        return Negate(self)

    def __abs__(self):
        return Abs(self)

    def __lt__(self, other):
        return LessThan(self, other)

    def __le__(self, other):
        return LessThanOrEqual(self, other)

    def __gt__(self, other):
        return GreaterThan(self, other)

    def __ge__(self, other):
        return GreaterThanOrEqual(self, other)

    # TODO: __eq__ (==) and __ne__ (!=) are not implemented here,
    # because they are used in set(nodes), which relies upon
    # both equality checks and __hash__.


class Constant(Node, OverloadMixin):
    """A constant is a number or a string. If the value is a string, sampling returns an array of the string value.

    Examples
    --------
    >>> Constant(2)._sample(5)
    array([2, 2, 2, 2, 2])
    >>> Constant("car")._sample(5)
    array(['car', 'car', 'car', 'car', 'car'], dtype='<U3')
    """

    is_source_node = True  # A Constant is always a source node

    def __init__(self, value):
        self.value = value.value if isinstance(value, Constant) else value
        super().__init__()

    def _sample(self, size=None):
        if size is None:
            return self.value
        return np.array([self.value] * size)

    def get_parents(self):
        yield from []  # A Constant does not have any parents

    def __repr__(self):
        return f"{type(self).__name__}({self.value})"


class AbstractDistribution(Node, OverloadMixin, abc.ABC):
    pass


class Distribution(AbstractDistribution):
    """A distribution is a sampling node with or without ancestors."""

    def __init__(self, distr, *args, **kwargs):
        self.distr = distr
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    def __repr__(self):
        args = ", ".join(repr(arg) for arg in self.args)
        kwargs = ", ".join(f"{k}={repr(v)}" for (k, v) in self.kwargs.items())
        out = f'{type(self).__name__}("{self.distr}"'
        if args:
            out += f", {args}"
        if kwargs:
            out += f", {kwargs}"
        return out + ")"

    def to_scipy(self):
        if not self._is_initial_sampling_node():
            raise Exception(
                "To convert a distribution to a scipy object, "
                "it must be an initial sampling node (no ancestors can be Distributions)"
            )

        node = self.copy()  # do not mutate self

        try:
            distribution = getattr(stats, node.distr)
        except AttributeError:
            raise AttributeError(f"{self.distr!r} is not a valid scipy distribution")

        def to_number(arg):
            """Unpack argument to a number in case parents are Constant/Transform"""
            return arg.sample(1)[0] if isinstance(arg, Node) else arg

        args = tuple(to_number(arg) for arg in node.args)
        kwargs = {k: to_number(v) for (k, v) in node.kwargs.items()}

        return distribution(*args, **kwargs)

    def _sample(self, q):
        def unpack(arg):
            """Unpack distribution arguments (parents) to arrays if Node."""
            return arg.samples_ if isinstance(arg, Node) else arg

        # Parse the arguments and keyword arguments for the distribution
        args = tuple(unpack(arg) for arg in self.args)
        kwargs = {k: unpack(v) for (k, v) in self.kwargs.items()}

        # Sample from the distribution with inverse CDF
        distribution = getattr(stats, self.distr)
        try:
            return distribution(*args, **kwargs).ppf(q)
        except AttributeError:
            # Multivariate distributions do not have .ppf()
            # isinstance(distribution, (multi_rv_generic, multi_rv_frozen))
            seed = int(q[0] * 2**20)  # Seed based on q
            return distribution(*args, **kwargs).rvs(size=len(q), random_state=seed)

    def get_parents(self):
        # A distribution only has parents if it has parameters that are Nodes
        for arg in self.args + tuple(self.kwargs.values()):
            if isinstance(arg, Node):
                yield arg

    @property
    def is_source_node(self):
        return list(self.get_parents()) == []


class EmpiricalDistribution(AbstractDistribution):
    """A distribution is a sampling node with or without ancestors.

    A thin wrapper around numpy.quantile."""

    is_source_node = True

    def __init__(self, data, **kwargs):
        self.data = np.array(data)
        self.kwargs = kwargs
        super().__init__()

    def __repr__(self):
        return f"{type(self).__name__}()"

    def _sample(self, q):
        return np.quantile(a=self.data, q=q, **self.kwargs)

    def get_parents(self):
        yield from []  # A EmpiricalDistribution does not have any parents


class CumulativeDistribution(AbstractDistribution):
    """A distribution defined by cumulative quantiles.

    Examples
    --------
    >>> distr = CumulativeDistribution([0, 0.2, 0.8, 1], [10, 15, 20, 25])
    >>> distr._sample(np.linspace(0, 1, num=6))
    array([10.        , 15.        , 16.66666667, 18.33333333, 20.        ,
           25.        ])
    >>> distr.sample(9, random_state=42)
    array([16.45450099, 23.76785766, 19.43328285, 18.32215403, 13.90046601,
           13.89986301, 11.4520903 , 21.65440364, 18.3426251 ])
    """

    is_source_node = True

    def __init__(self, quantiles, cumulatives):
        self.q = np.array(quantiles)
        self.cumulatives = np.array(cumulatives)
        if not np.all(np.diff(self.q) > 0):
            raise ValueError("The quantiles must be strictly increasing.")
        if not np.all(np.diff(self.cumulatives) > 0):
            raise ValueError("The cumulatives must be strictly increasing.")
        if not (np.isclose(np.min(self.q), 0) and np.isclose(np.max(self.q), 1)):
            raise ValueError("Lowest quantile must be 0 and highest must be 1.")
        super().__init__()

    def __repr__(self):
        return f"{type(self).__name__}(quantiles={repr(self.q)}, cumulatives={repr(self.cumulatives)})"

    def _sample(self, q):
        # Inverse CDF sampling
        return np.interp(x=q, xp=self.q, fp=self.cumulatives)

    def get_parents(self):
        yield from []


class DiscreteDistribution(AbstractDistribution):
    """A discrete (or categorical) distribution defined by values and probabilities.

    Examples
    --------
    >>> distr = DiscreteDistribution([10, 15, 20], probabilities=[0.2, 0.3, 0.5])
    >>> distr._sample(np.linspace(0, 1, num=5, endpoint=False))
    array([10, 15, 15, 20, 20])
    >>> distr = DiscreteDistribution(["A", "B", "C", "D", "E", "F"])
    >>> distr.sample(9, random_state=42)
    array(['C', 'F', 'E', 'D', 'A', 'A', 'A', 'F', 'D'], dtype='<U1')
    """

    is_source_node = True

    def __init__(self, values, probabilities=None):
        self.values = np.array(values)
        if probabilities is None:
            self.probabilities = np.ones(len(self.values), dtype=float)
            self.probabilities = self.probabilities / np.sum(self.probabilities)
        else:
            self.probabilities = np.array(probabilities)

        if not len(self.values) == len(self.probabilities):
            raise ValueError(
                f"Length mismatch: {len(self.values)=}  {len(self.probabilities)=}"
            )
        if not np.isclose(np.sum(self.probabilities), 1.0):
            raise ValueError(f"Probabilities must sum to 1. {sum(self.probabilities)=}")
        if np.any(self.probabilities < 0):
            raise ValueError("Probabilities are not non-negative.")
        super().__init__()

    def __repr__(self):
        return f"{type(self).__name__}(values={repr(self.values)}, probabilities={repr(self.probabilities)})"

    def _sample(self, q):
        cumulative_probabilities = np.cumsum(self.probabilities)
        idx = np.searchsorted(cumulative_probabilities, v=q, side="right")
        return self.values[idx]

    def get_parents(self):
        yield from []


# ========================================================


class Transform(Node, OverloadMixin, abc.ABC):
    """Transform nodes represent arithmetic operations."""

    is_source_node = False

    def __repr__(self):
        parents = ", ".join(repr(parent) for parent in self.get_parents())
        return f"{type(self).__name__}({parents})"


class VariadicTransform(Transform):
    """Parent class for variadic transforms (must be associative), e.g.
    Add(arg1, arg2, arg3, arg4, ...)
    Multiply(arg1, arg2, arg3, arg4, ...)

    """

    def __init__(self, *args):
        self.parents = tuple(python_to_prob(arg) for arg in args)
        super().__init__()

    def _sample(self, size=None):
        samples = (parent.samples_ for parent in self.parents)
        return functools.reduce(self.op, samples)

    def get_parents(self):
        yield from self.parents


class Add(VariadicTransform):
    op = operator.add


class Multiply(VariadicTransform):
    op = operator.mul


class Max(VariadicTransform):
    op = np.maximum


class Min(VariadicTransform):
    op = np.minimum


class All(VariadicTransform):
    op = np.logical_and


class Any(VariadicTransform):
    op = np.logical_or


class Avg(VariadicTransform):
    def _sample(self, size=None):
        # Avg(a, Avg(b, c)) !=  Avg(Avg(a, b), c), so we override _sample()
        samples = tuple(parent.samples_ for parent in self.parents)
        return np.average(np.vstack(samples), axis=0)


class NoOp(VariadicTransform):
    """Sample all ancestor variables, but do nothing else."""

    def _sample(self, size=None):
        tuple(parent.samples_ for parent in self.parents)


class BinaryTransform(Transform):
    """Class for binary transforms, such as Divide, Power, Subtract, etc."""

    def __init__(self, *args):
        self.parents = tuple(python_to_prob(arg) for arg in args)
        super().__init__()

    def _sample(self, size=None):
        samples = (parent.samples_ for parent in self.parents)
        return self.op(*samples)

    def get_parents(self):
        yield from self.parents


class FloorDivide(BinaryTransform):
    op = np.floor_divide


class Mod(BinaryTransform):
    op = np.mod


class Divide(BinaryTransform):
    op = operator.truediv


class Power(BinaryTransform):
    op = operator.pow


class Subtract(BinaryTransform):
    op = operator.sub


class Equal(BinaryTransform):
    op = np.equal


class NotEqual(BinaryTransform):
    op = np.not_equal


class LessThan(BinaryTransform):
    op = operator.lt


class LessThanOrEqual(BinaryTransform):
    op = operator.le


class GreaterThan(BinaryTransform):
    op = operator.gt


class GreaterThanOrEqual(BinaryTransform):
    op = operator.ge


class IsClose(BinaryTransform):
    op = np.isclose


class UnaryTransform(Transform):
    """Class for unary tranforms, i.e. functions that take one argument, such
    as Abs(), Exp(), Log()."""

    def __init__(self, arg):
        self.parent = python_to_prob(arg)
        super().__init__()

    def _sample(self, size=None):
        return self.op(self.parent.samples_)

    def get_parents(self):
        yield self.parent


class Negate(UnaryTransform):
    op = operator.neg


class Abs(UnaryTransform):
    op = operator.abs


class Log(UnaryTransform):
    op = np.log


class Exp(UnaryTransform):
    op = np.exp


class Floor(UnaryTransform):
    op = np.floor


class Ceil(UnaryTransform):
    op = np.ceil


class Sign(UnaryTransform):
    op = np.sign


class Sqrt(UnaryTransform):
    op = np.sqrt


class Square(UnaryTransform):
    op = np.square


class Log10(UnaryTransform):
    op = np.log10


# Trigonometric functions
class Sin(UnaryTransform):
    op = np.sin


class Cos(UnaryTransform):
    op = np.cos


class Tan(UnaryTransform):
    op = np.tan


class Arcsin(UnaryTransform):
    op = np.arcsin


class Arccos(UnaryTransform):
    op = np.arccos


class Arctan(UnaryTransform):
    op = np.arctan


class Arctan2(BinaryTransform):
    op = np.arctan2


# Hyperbolic functions
class Sinh(UnaryTransform):
    op = np.sinh


class Cosh(UnaryTransform):
    op = np.cosh


class Tanh(UnaryTransform):
    op = np.tanh


class Arcsinh(UnaryTransform):
    op = np.arcsinh


class Arccosh(UnaryTransform):
    op = np.arccosh


class Arctanh(UnaryTransform):
    op = np.arctanh


class ScalarFunctionTransform(Transform):
    """A general-purpose transform using a function that takes scalar arguments
    and returns a scalar result."""

    def __init__(self, func, args, kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs
        super().__init__()

    def _sample(self, size=None):
        def unpack(arg):
            return arg.samples_ if isinstance(arg, Node) else itertools.repeat(arg)

        # Sample arguments
        args = tuple(unpack(arg) for arg in self.args)
        kwargs = {k: unpack(v) for (k, v) in self.kwargs.items()}

        return np.array(
            [
                self.func(*args_i, **kwargs_i)
                for (args_i, kwargs_i) in zip_args(args, kwargs)
            ]
        )

    def get_parents(self):
        # A function has have parents if its arguments are Nodes
        for arg in self.args + tuple(self.kwargs.values()):
            if isinstance(arg, Node):
                yield arg


def scalar_transform(func):
    """Transform a function, so that when it is called it is converted to
    a ScalarFunctionTransform."""

    @functools.wraps(func)
    def transformed_function(*args, **kwargs):
        return ScalarFunctionTransform(func, args, kwargs)

    return transformed_function


class MarginalDistribution(Transform):
    """A maginal distribution is a 'slice' of a multivariate distribution.

    Examples
    --------
    >>> distr = Distribution("multinomial", n=10, p=[0.1, 0.2, 0.7])
    >>> marginal_distr = MarginalDistribution(distr, d=0)
    >>> marginal_distr
    MarginalDistribution(Distribution("multinomial", n=10, p=[0.1, 0.2, 0.7]), d=0)
    >>> marginal_distr.sample(5, random_state=0).astype(int)
    array([2, 1, 2, 1, 1])
    """

    is_source_node = False

    def __init__(self, distr, d):
        self.distr = distr
        self.d = d
        super().__init__()

    def _sample(self):
        # Simply slice the parent
        return np.atleast_2d(self.distr.samples_)[:, self.d]

    def get_parents(self):
        yield self.distr

    def __repr__(self):
        return f"{type(self).__name__}({self.distr}, d={self.d})"


def MultivariateDistribution(distr, *args, **kwargs):
    """Factory function that yields marginal distributions.

    Examples
    --------
    >>> p = [0.2, 0.3, 0.5]  # Probability of each category
    >>> m1, m2, m3 = MultivariateDistribution("multinomial", n=10, p=p)
    >>> m1.sample(5, random_state=0).astype(int)
    array([3, 2, 4, 2, 1])

    Each category should sum to n=10:
    >>> (m1 + m2 + m3).sample(5, random_state=0).astype(int)
    array([10, 10, 10, 10, 10])
    """
    distr = Distribution(distr, *args, **kwargs)

    # Get dimensionality by sampling once
    d = len(distr._sample(q=[0.5]).squeeze())
    yield from (MarginalDistribution(distr, d=i) for i in range(d))


# ========================================================
if __name__ == "__main__":
    import pytest

    pytest.main(args=[__file__, "--doctest-modules", "-v", "--capture=sys"])

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    a = Distribution("norm", loc=0, scale=1)
    b = Distribution("norm", loc=0, scale=1)
    c = Distribution("norm", loc=0, scale=1)

    expression = a + b
    corr_mat = np.array([[1.0, 0.8], [0.8, 1.0]])
    expression.correlate(a, b, corr_mat=corr_mat)

    expression = expression + c
    expression.correlate(b, c, corr_mat=corr_mat)

    import matplotlib.pyplot as plt

    expression.sample(999, random_state=rng)

    plt.figure(figsize=(3, 2))
    plt.scatter(a.samples_, b.samples_, s=2)
    plt.show()

    d1, d2, d3 = MultivariateDistribution("dirichlet", alpha=[1, 2, 3])

    # =========================

    cost = EmpiricalDistribution(data=[1, 2, 3, 3, 3, 3])
    norm = Distribution("norm", loc=cost, scale=1)
    (norm**2).sample(99, random_state=42)
