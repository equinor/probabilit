import collections


class GarbageCollector:
    """The GarbageCollector cleans up `.samples_` attributes as the computational
    graph is sampled. By 'garbage collection' and 'cleaning up' we mean deleting
    the `.samples_` attribute.

    Parameters
    ----------
    strategy : None or list, optional
        If None (the default), no nodes are garbage collected. If a list of
        nodes, then those nodes and the sink are NOT garbage collected. An
        empty list means all nodes except the sink will be garbage collected.
    """

    def __init__(self, strategy=None):
        assert strategy is None or isinstance(strategy, list)
        self.strategy = strategy

    def set_sink(self, sink):
        """Set the sink node, whose samples will always be kept."""
        self.sink = sink

        # The user wants to keep `.samples_` on all nodes => do nothing
        if self.strategy is None:
            return self

        # Initialize the reference counter, keeping track of the number of
        # unsampled children of all nodes. Once a node has no unsampled children
        # (i.e. all children are sampled), that node can safely be GC'ed.
        self._unsampled_children = collections.defaultdict(int)
        for node in self.sink.nodes():
            for parent in node.get_parents():
                self._unsampled_children[parent] += 1

        return self

    def decrement_and_delete(self, node):
        """Decrement the reference counter (number of unsampled children for
        each parent) and delete `.samples_` is the reference count is zero.

        Returns the nodes that were garbage collected.
        """
        if not hasattr(self, "sink"):
            raise ValueError("You must call 'set_sink' first.")

        # Nodes that were garbage collected
        garbage_collected = []

        # The user wants to keep `.samples_` on all nodes => do nothing
        if self.strategy is None:
            return []

        for parent in node.get_parents():
            # Decrement counter
            self._unsampled_children[parent] -= 1

            # Garbage collect if possible
            zero_count = self._unsampled_children[parent] == 0
            protected = parent in self.strategy
            if zero_count and not protected:
                del parent.samples_
                garbage_collected.append(parent)

            assert self._unsampled_children[parent] >= 0

        return garbage_collected
