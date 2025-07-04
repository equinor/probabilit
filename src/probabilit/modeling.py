"""
Modeling
--------

Probabilit lets the user perform Monte-Carlo sampling using a high-level
modeling language, which creates a computational graph.

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
20.00139737515...

Distributions are built on top of scipy, so "norm" refers to the name of the
normal distribution as given in `scipy.stats`, and the arguments to the distribution
must also match those given by `scipy.stats.norm`.

Here is another example showing composite distributions, where the argument
to one distribution is another distribution:

>>> eggs_per_nest = Distribution("poisson", mu=3)
>>> survivial_prob = Distribution("beta", a=10, b=15)
>>> survived = Distribution("binom", n=eggs_per_nest, p=survivial_prob)
>>> survived.sample(9, random_state=rng)
array([1., 1., 1., 0., 2., 1., 1., 0., 2.])

To understand and examine the modeling language, we can do some computations
with constants. The computational graph carries out arithmetic operations lazily
once a model is sampled. Mixing numbers with nodes is allowed, but at least one
expression or term must be a probabilit class instance:

>>> a = Constant(1)
>>> (a * 3 + 5).sample(5, random_state=rng)
array([8, 8, 8, 8, 8])
>>> Add(10, 5, 5).sample(5, random_state=rng)
array([20, 20, 20, 20, 20])

Let us build a more compliated expression:

>>> a = Distribution("norm", loc=5, scale=1)
>>> b = Distribution("expon", scale=1)
>>> expression = a**b + a * b + 5 * b

Every unique node in this expression can be found:

>>> for node in set(expression.nodes()):
...     print(node)
Distribution("norm", loc=5, scale=1)
Distribution("expon", scale=1)
Power(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1))
Multiply(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1))
Add(Power(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1)), Multiply(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1)))
Constant(5)
Multiply(Distribution("expon", scale=1), Constant(5))
Add(Add(Power(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1)), Multiply(Distribution("norm", loc=5, scale=1), Distribution("expon", scale=1))), Multiply(Distribution("expon", scale=1), Constant(5)))

Sampling the expression is simple enough:

>>> expression.sample(5, random_state=rng)
array([ 2.70764145, 36.58578812,  7.07064239,  1.84433247,  3.90951632])

Sampling the expression has the side effect that `.samples_` is populated on
*every* node in the expression, for instance:

>>> a.samples_
array([4.51589278, 4.37788659, 5.25960812, 5.80609507, 4.33770499])

To sample using e.g. Latin Hypercube, do the following:

>>> from scipy.stats.qmc import LatinHypercube
>>> d = expression.get_number_of_distribution_nodes()
>>> hypercube = LatinHypercube(d=d, rng=rng)
>>> hypercube_samples = hypercube.random(5) # Draw 5 samples
>>> expression.sample_from_cube(hypercube_samples)
array([ 1.20438726, 12.40283222,  5.02130766, 16.45109076, 77.12874028])

Here is an even more complex expression:

>>> a = Distribution("norm", loc=0, scale=1)
>>> b = Distribution("norm", loc=0, scale=2)
>>> c = Distribution("norm", loc=0, scale=3)
>>> expression = a*a - Add(a, b, c) + Abs(b)**Abs(c) + Exp(1 / Abs(c))
>>> expression.sample(5, random_state=rng)
array([ 4.70542018, 14.43250192,  6.74494838, -0.14020459, -3.27334554])

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
>>> expression = function(a, b) # Function is not actually called here

Now sample 'through' the function:

>>> expression.sample(5, random_state=rng)
array([0.        , 0.        , 0.45555522, 0.        , 0.        ])
"""

import operator
import functools
import numpy as np
import numbers
from scipy import stats
import abc
import itertools
import networkx as nx
from scipy._lib._util import check_random_state
from probabilit.iman_conover import ImanConover
from probabilit.correlation import nearest_correlation_matrix
import copy


# =============================================================================
# FUNCTIONS
# =============================================================================


def zip_args(args, kwargs):
    """Zip argument and keyword arguments for repeated function calls.

    Examples
    --------
    >>> args = ((1, 2, 3), itertools.repeat(None))
    >>> kwargs = {"a": (5, 6, 7), "b": itertools.repeat(9)}
    >>> for args_i, kwargs_i in zip_args(args, kwargs):
    ...     print(args_i, kwargs_i)
    (1, None) {'a': 5, 'b': 9}
    (2, None) {'a': 6, 'b': 9}
    (3, None) {'a': 7, 'b': 9}
    """
    zipped_args = zip(*args) if args else itertools.repeat(args)
    zipped_kwargs = zip(*kwargs.values()) if kwargs else itertools.repeat(kwargs)

    for args_i, kwargs_i in zip(zipped_args, zipped_kwargs):
        yield args_i, dict(zip(kwargs.keys(), kwargs_i))


def build_corrmat(correlations):
    """Given a list of [(indices1, corrmat1), (indices2, corrmat2), ...],
    create a big correlation matrix.

    Examples
    --------
    >>> correlations = [((0, 2), np.array([[1, 0.5], [0.5, 1]]))]
    >>> build_corrmat(correlations)
    array([[1. , 0. , 0.5],
           [0. , 1. , 0. ],
           [0.5, 0. , 1. ]])
    """
    # TODO: If no correlation is given, we implicitly assume zero.
    # For instance, if no correlation between indices (0, 3) is given
    # in the input data, then C[0, 3] = C[3, 0] = 0.0, which is strictly
    # speaking not the same (no preference vs. preference for 0 corr)
    n = max(max(idx) for (idx, _) in correlations)
    C = np.eye(n + 1, dtype=float)

    for idx_i, corrmat_i in correlations:
        C[np.ix_(idx_i, idx_i)] = corrmat_i

    return C


def python_to_prob(argument):
    """Convert basic Python types to probabilit types."""
    if isinstance(argument, numbers.Number):
        return Constant(argument)
    return argument


# =============================================================================
# COMPUTATIONAL GRAPH AND MODELING LANGUAGE
# =============================================================================
#
# There are three main types of Node instances, they are:
#   - Constant:     numbers like 2 or 5.5, which are always source nodes
#   - Distribution: typically source nodes, but can be non-source if composite
#   - Transform:    arithmetic operations like + or **, or general functions
#
# |              | source node | non-source node |
# |--------------|-------------|-----------------|
# | Constant     |             | N/A             |
# | Distribution |             |                 |
# | Transform    | N/A         |                 |
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
#   * "b" is a Distribution, but not a source node
#   * "+" is a Transform
#   * "2" is a Constant and a source node
#   * "-" (the result) is a Transform
#
# Some further terminology:
#   * The _parents_ of node "-" are {"+", "2"}
#   * The _ancestors_ of node "-" are {"+", "2", "mu", "normal"}
#   * A node is said to be an _initial sampling node_ iff
#      (1) The node is a Distribution
#      (2) None of its ancestors are Distributions
#     For instance, in the graph above, the node "mu" is an initial sampling node.
#     Initial sampling nodes are the nodes that we can impose correlations on.
#     We cannot impose correlations on "normal" above, since its correlation
#     is determined by the graph structure.


class Node(abc.ABC):
    """A node in the computational graph."""

    id_iter = itertools.count()  # Every node gets a unique ID

    def __init__(self):
        self._id = next(self.id_iter)
        self._correlations = []

    def __eq__(self, other):
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
            if isinstance(copied, (Distribution, ScalarFunctionTransform)):
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

    def get_number_of_distribution_nodes(self):
        return sum(1 for node in set(self.nodes()) if isinstance(node, Distribution))

    def sample(self, size=None, random_state=None):
        """Sample the current node and assign to all node.samples_."""
        size = 1 if size is None else size
        random_state = check_random_state(random_state)

        # Draw a cube of random variables in [0, 1]
        cube = random_state.random((size, self.get_number_of_distribution_nodes()))

        return self.sample_from_cube(cube)

    def sample_from_cube(self, cube):
        """Use samples from a cube of quantiles in [0, 1] to sample all
        distributions. The cube must have shape (dimensionality, num_samples)."""
        assert nx.is_directed_acyclic_graph(self.to_graph())
        size, n_dim = cube.shape
        assert n_dim == self.get_number_of_distribution_nodes()

        # Prepare columns of quantiles, one column for each Distribution
        columns = iter(list(cube.T))

        # Clear any samples that might exist in the graph
        for node in set(self.nodes()):
            if hasattr(node, "samples_"):
                delattr(node, "samples_")

        # Start with initial sampling nodes, which contain independent variables
        initial_sampling_nodes = set(
            node for node in self.nodes() if node._is_initial_sampling_node()
        )

        # Loop over all initial sampling nodes
        G = self.to_graph()
        for node in initial_sampling_nodes:
            # Sample all ancestors
            ancestors = G.subgraph(nx.ancestors(G, node))
            for ancestor in nx.topological_sort(ancestors):
                assert isinstance(ancestor, (Constant, Distribution))
                ancestor.samples_ = ancestor._sample(size=size)

            # Sample the node
            assert isinstance(node, Distribution)
            node.samples_ = node._sample(q=next(columns))

        # Go through all ancestor nodes and create a list [(var, corr), ...]
        correlations = []
        for node in set(self.nodes()):
            if hasattr(node, "_correlations"):
                correlations.extend(node._correlations)

        # Check that each variable to correlate is an initial sampling node
        for variables, _ in correlations:
            for variable in variables:
                if variable not in initial_sampling_nodes:
                    raise ValueError("Cannot correlate variable: {variable}")

        # Check that no correlation has been specified twice
        variable_sets = [set(variables) for (variables, _) in correlations]
        for vars1, vars2 in itertools.combinations(variable_sets, 2):
            common = vars1.intersection(vars2)
            if len(common) > 1:
                raise ValueError("Correlations specified more than once: {common}")

        # Map all variables to integers
        all_variables = list(functools.reduce(set.union, variable_sets, set()))
        var_to_int = {v: i for (i, v) in enumerate(all_variables)}
        correlations = [
            (tuple(var_to_int[var] for var in variables), corrmat)
            for (variables, corrmat) in correlations
        ]

        # If there are any correlations to induce, do so
        if correlations:
            # Induce correlations
            correlation_matrix = build_corrmat(correlations)
            correlation_matrix = nearest_correlation_matrix(correlation_matrix)

            iman_conover = ImanConover(correlation_matrix)

            # Concatenate samples, correlate them (shift rows in each col), then re-assign
            samples_input = np.vstack([var.samples_ for var in all_variables]).T
            samples_ouput = iman_conover(samples_input)
            for var, sample in zip(all_variables, samples_ouput.T):
                var.samples_ = np.copy(sample)

        # Iterate from leaf nodes and up to parent
        for node in nx.topological_sort(G):
            if hasattr(node, "samples_"):
                continue
            elif isinstance(node, Constant):
                node.samples_ = node._sample(size=size)
            elif isinstance(node, Distribution):
                node.samples_ = node._sample(q=next(columns))
            elif isinstance(node, Transform):
                node.samples_ = node._sample()
            else:
                raise TypeError("Node must be Constant, Distribution or Transform.")

        return self.samples_

    def _is_initial_sampling_node(self):
        """A node is an initial sample node iff:
        (1) It is a Distribution
        (2) None of its ancestors are Distributions (all are Constant/Transform)"""
        is_distribution = isinstance(self, Distribution)
        ancestors = set(self.nodes()) - set([self])
        ancestors_distr = any(isinstance(node, Distribution) for node in ancestors)
        return is_distribution and not ancestors_distr

    def correlate(self, *variables, corr_mat):
        """Impose correlations on variables.

        When `.correlate(*variables)` is called on a node, the variables must
        be ancestors of that node. The order of the variables should match the
        order of the rows/columns in the correlation matrix.s

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
        0.433649...

        """
        assert corr_mat.ndim == 2
        assert corr_mat.shape[0] == corr_mat.shape[1]
        assert corr_mat.shape[0] == len(variables)
        assert len(variables) == len(set(variables))
        nodes = set(self.nodes())
        assert all(var in nodes for var in variables)
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
            if not node.is_leaf
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

    def __truediv__(self, other):
        return Divide(self, other)

    def __rtruediv__(self, other):
        return Divide(other, self)

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


class Constant(Node, OverloadMixin):
    """A constant is a number."""

    is_leaf = True  # A Constant is always a leaf node

    def __init__(self, value):
        self.value = value.value if isinstance(value, Constant) else value
        super().__init__()

    def _sample(self, size=None):
        if size is None:
            return self.value
        return np.ones(size, dtype=type(self.value)) * self.value

    def get_parents(self):
        return []  # A Constant does not have any parents

    def __repr__(self):
        return f"{type(self).__name__}({self.value})"


class Distribution(Node, OverloadMixin):
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

    def _sample(self, q):
        def unpack(arg):
            """Unpack distribution arguments (parents) to arrays if Node."""
            return arg.samples_ if isinstance(arg, Node) else arg

        # Parse the arguments and keyword arguments for the distribution
        args = tuple(unpack(arg) for arg in self.args)
        kwargs = {k: unpack(v) for (k, v) in self.kwargs.items()}

        # Sample from the distribution with inverse CDF
        distribution = getattr(stats, self.distr)
        return distribution(*args, **kwargs).ppf(q)

    def get_parents(self):
        # A distribution only has parents if it has parameters that are Nodes
        for arg in self.args + tuple(self.kwargs.values()):
            if isinstance(arg, Node):
                yield arg

    @property
    def is_leaf(self):
        return list(self.get_parents()) == []


# ========================================================


class Transform(Node, abc.ABC, OverloadMixin):
    """Transform nodes represent arithmetic operations."""

    is_leaf = False

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
        return functools.reduce(self.op, samples, self.initial)

    def get_parents(self):
        yield from self.parents


class Add(VariadicTransform):
    op = operator.add
    initial = 0


class Multiply(VariadicTransform):
    op = operator.mul
    initial = 1


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


class Divide(BinaryTransform):
    op = operator.truediv


class Power(BinaryTransform):
    op = operator.pow


class Subtract(BinaryTransform):
    op = operator.sub


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
