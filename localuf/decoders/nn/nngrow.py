from typing import Iterator, Optional, Set, Dict, List
from localuf.type_aliases import Node, Edge
import networkx as nx
from collections import deque


class ClusterBase:
    """Base class for clusters in the growing strategy."""

    def __init__(self, _id: int) -> None:
        self._cluster_id: int = _id
        self._size: int = 0
        self._n_syndrome: int = 0
        self._preceding: Optional["ClusterBase"] = None
        self._locked: bool = False
        self._has_boundary: bool = False
        self._syndrome_nodes: Set[Node] = set()  # Track syndrome nodes directly
        self._boundary_nodes: Set[Node] = set()  # Track boundary nodes directly

    def __repr__(self) -> str:
        """String representation of the cluster."""
        status = "LOCKED" if self.locked() else "ACTIVE"
        parity = "ODD" if self.odd() else "EVEN"
        return f"Cluster(id={self._cluster_id}, size={self._size}, syndromes={self._n_syndrome}, {parity}, {status})"

    def __iter__(self) -> Iterator[Node]:
        """Iterate over all nodes in the cluster."""
        raise NotImplementedError("Subclasses must implement __iter__")

    def odd(self) -> bool:
        return bool(self._n_syndrome % 2)

    def root(self) -> "ClusterBase":
        """Get the root of this cluster's tree."""
        if self._preceding is None:
            return self
        self._preceding = self._preceding.root()
        return self._preceding

    def locked(self) -> bool:
        """Check if this cluster is locked.
        The root cluster has priority in locking.
        """
        return self._locked if self._preceding is None else self._preceding.locked()

    def lock(self) -> None:
        """Lock this cluster."""
        if self._preceding is not None:
            self.root()._locked = True
        else:
            self._locked = True

    def add(self, node: Node, syndrome: Set[Node], boundary_nodes: Set[Node]) -> None:
        """Add a node to this cluster."""
        assert not self.locked()
        assert self._preceding is None
        self._size += 1
        if node in syndrome:
            self._syndrome_nodes.add(node)
            self._n_syndrome += 1
            if self._n_syndrome % 2 == 0:  # Even number of syndromes - unlock
                self._locked = False
            else:  # Odd number of syndromes - lock
                self.lock()
        if node in boundary_nodes:
            self._boundary_nodes.add(node)
            self._has_boundary = True
            self.lock()

    @property
    def syndrome_nodes(self):
        return self._syndrome_nodes

    @property
    def boundary_nodes(self):
        return self._boundary_nodes


class Cluster(ClusterBase):
    """A simple cluster containing nodes."""

    def __init__(self, _id: int, original: set[Node], syndrome: Set[Node], boundary_nodes: Set[Node]) -> None:
        super().__init__(_id)
        self.nodes: Set[Node] = original.copy()

        # Initialize syndrome and boundary tracking
        self._syndrome_nodes = {n for n in original if n in syndrome}
        self._boundary_nodes = {n for n in original if n in boundary_nodes}

        self._n_syndrome = len(self._syndrome_nodes)
        self._size = len(original)
        self._has_boundary = bool(self._boundary_nodes)
        self._locked = self._has_boundary or not self.odd()

    def __iter__(self) -> Iterator[Node]:
        return iter(self.nodes)

    def add(self, node: Node, syndrome: Set[Node], boundary_nodes: Set[Node]) -> None:
        """Add a node to this cluster."""
        self.nodes.add(node)
        super().add(node, syndrome, boundary_nodes)


class MergedCluster(ClusterBase):
    """A cluster merged from two or more clusters."""

    def __init__(self, cluster1: ClusterBase, cluster2: ClusterBase, _id: int) -> None:
        super().__init__(_id)
        self.elements: List[ClusterBase] = [cluster1, cluster2]
        self.nodes: Set[Node] = set()  # Only for nodes directly added to this cluster

        assert cluster1.root() == cluster1
        assert cluster2.root() == cluster2
        assert cluster1.root() != cluster2.root(), "Cannot merge the same cluster"

        # Combine syndrome and boundary nodes from subclusters
        self._syndrome_nodes = cluster1._syndrome_nodes.union(cluster2.root()._syndrome_nodes)
        self._boundary_nodes = cluster1._boundary_nodes.union(cluster2.root()._boundary_nodes)

        cluster1._preceding = self
        cluster2._preceding = self
        # The merged cluster should be locked / activated depending on:
        # 1. The parity of the new cluster
        # 2. The boundary nodes
        self._size = cluster1._size + cluster2._size
        self._n_syndrome = len(self._syndrome_nodes)
        self._has_boundary = bool(self._boundary_nodes)
        self._locked = not ((not self.odd()) or self._has_boundary)

    def __iter__(self) -> Iterator[Node]:
        # First yield nodes directly added to this cluster
        for node in self.nodes:
            yield node
        # Then yield nodes from sub-clusters
        for element in self.elements:
            for node in element:
                yield node

    def add(self, node: Node, syndrome: Set[Node], boundary_nodes: Set[Node]) -> None:
        """Add a node to this merged cluster."""
        self.nodes.add(node)
        super().add(node, syndrome, boundary_nodes)


def grow_clusters(
    g: nx.Graph, original_clusters: List[Set[Node]], syndrome: Set[Node], boundary_nodes: Set[Node]
) -> List[Set[Node]] | None:
    """
    Implementation of the 'grow' strategy for NN decoder.

    Args:
        g: The graph representing the code
        original_clusters: The clusters from fusion stage
        syndrome: Set of syndrome nodes
        boundary_nodes: Set of boundary nodes

    Returns:
        A list of valid (even parity) clusters
    """
    # Classify each cluster as io, ie, bo, be
    io_clusters = []  # internal odd clusters (activated)
    locked_clusters = []  # ie, bo, be clusters (locked)

    # Create Cluster objects for each set of nodes
    cluster_id = 0
    clusters = []

    for nodes in original_clusters:
        cluster = Cluster(cluster_id, nodes, syndrome, boundary_nodes)
        cluster_id += 1

        is_boundary = any(node in boundary_nodes for node in nodes)
        clusters.append(cluster)

        # Classify cluster. Lock even clusters and clusters with boundary nodes.
        if is_boundary or not cluster.odd():
            cluster.lock()
            locked_clusters.append(cluster)
        else:
            io_clusters.append(cluster)

    # Map each node to its cluster
    node_cluster: Dict[Node, ClusterBase] = {}
    for cluster in clusters:
        for node in cluster:
            node_cluster[node] = cluster

    # Map edges to their growth state: 0 (zero), 1 (half), 2 (full)
    edge_growth: Dict[Edge, int] = {}
    for u, v in g.edges:
        edge_growth[(u, v)] = 0
        edge_growth[(v, u)] = 0

    # Start with all nodes in activated (io) clusters
    node_queue = deque()
    for cluster in io_clusters:
        for node in cluster:
            node_queue.append(node)

    # Keep track of active cluster roots
    active_clusters = set(io_clusters)

    # Keep track of all root clusters for efficient processing at the end
    all_root_clusters: Set[ClusterBase] = set()
    for cluster in clusters:
        all_root_clusters.add(cluster)

    # Grow algorithm. Elements in node_queue are the `boundary` of clusters.
    while node_queue and active_clusters:
        u = node_queue.popleft()
        assert u in node_cluster, f"Node {u} not found in node_cluster"
        if node_cluster[u].root().locked():
            continue

        # If the edges around `u` are all fully grown, skip it.
        if all(edge_growth[(u, v)] == 2 for v in g.neighbors(u)):
            continue

        node_queue.append(u)

        for v in g.neighbors(u):
            edge = (u, v)
            # Handle edge growth
            if edge_growth[edge] == 0:
                edge_growth[edge] = 1
            elif edge_growth[edge] == 1:
                edge_growth[edge] = 2

                # Edge is fully grown, handle the other endpoint
                if v not in node_cluster:
                    # If v is not assigned to any cluster, Add v to u's cluster.
                    # After this, the cluster might be locked.
                    u_cluster = node_cluster[u].root()
                    u_cluster.add(v, syndrome, boundary_nodes)
                    node_cluster[v] = u_cluster
                    node_queue.append(v)
                elif node_cluster[v].root() != node_cluster[u].root():
                    # Merge the two clusters
                    u_cluster: Cluster | ClusterBase = node_cluster[u].root()
                    v_cluster: Cluster | ClusterBase = node_cluster[v].root()

                    # Only merge if u's cluster is still active
                    if not u_cluster.locked():
                        # Remove old root clusters from tracking set
                        all_root_clusters.discard(u_cluster)
                        all_root_clusters.discard(v_cluster)

                        merged = MergedCluster(u_cluster, v_cluster, cluster_id)
                        cluster_id += 1

                        # Add new root cluster to tracking set
                        all_root_clusters.add(merged)

                        # Update active clusters
                        if u_cluster in active_clusters:
                            active_clusters.remove(u_cluster)
                        if v_cluster in active_clusters:
                            active_clusters.remove(v_cluster)

                        if not merged.locked():
                            active_clusters.add(merged)
                            # Add all nodes from the merged cluster to the queue
                            for node in merged:
                                node_queue.append(node)

    # Collect final clusters based on tracked root clusters directly
    final_clusters: List[Set[Node]] = []

    for root in all_root_clusters:
        # We already have tracked syndrome and boundary nodes
        syndrome_nodes_in_cluster = root.syndrome_nodes
        boundary_nodes_in_cluster = root.boundary_nodes

        # Apply the rules for valid cluster selection
        valid_cluster: set[Node] = set()

        # Rule 1: If cluster has even syndromes, pick those syndromes
        if len(syndrome_nodes_in_cluster) % 2 == 0:
            valid_cluster = syndrome_nodes_in_cluster

            # Assertion to verify the cluster is valid
            assert len(valid_cluster) % 2 == 0, (
                f"Cluster should have " f"even number of syndromes: {valid_cluster}"
            )

        # Rule 2: If cluster has odd syndromes and boundary node(s), pick syndromes and one boundary node
        elif len(syndrome_nodes_in_cluster) % 2 == 1 and boundary_nodes_in_cluster:
            # Include all syndrome nodes and one boundary node
            valid_cluster = syndrome_nodes_in_cluster.union({next(iter(boundary_nodes_in_cluster))})

            # Assertion to verify the cluster is valid after adding boundary node
            assert len(valid_cluster) % 2 == 0, (
                f"Cluster should be " f"even after adding boundary node: {valid_cluster}"
            )

        else:
            # Skip illegal clusters (odd syndromes with no boundary node)
            assert False, (
                f"Illegal cluster: odd syndromes ({len(syndrome_nodes_in_cluster)}) " f"with no boundary node"
            )

        # Add the valid cluster to final results if it's not empty
        if valid_cluster:
            final_clusters.append(valid_cluster)

    return final_clusters
