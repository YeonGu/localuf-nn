from typing import Iterator, Optional, Set, Dict, List, Tuple
from localuf.type_aliases import Node, Edge
import networkx as nx
from collections import deque


class ClusterBase:
    """Base class for clusters in the growing strategy."""

    def __init__(self, _id: int) -> None:
        self.id: int = _id
        self.size: int = 0
        self.n_syndrome: int = 0
        self.preceding: Optional['ClusterBase'] = None
        self._locked: bool = False

    def __iter__(self) -> Iterator[Node]:
        """Iterate over all nodes in the cluster."""
        raise NotImplementedError("Subclasses must implement __iter__")

    def root(self) -> 'ClusterBase':
        """Get the root of this cluster's tree."""
        if self.preceding is None:
            return self
        self.preceding = self.preceding.root()
        return self.preceding

    def locked(self) -> bool:
        """Check if this cluster is locked."""
        return self._locked or (self.preceding is not None and self.root().locked())

    def lock(self) -> None:
        """Lock this cluster."""
        if self.preceding is not None:
            self.root()._locked = True
        else:
            self._locked = True

    def add(self, node: Node, syndrome: Set[Node], boundary_nodes: Set[Node]) -> None:
        """Add a node to this cluster."""
        self.size += 1
        if node in syndrome:
            self.n_syndrome += 1
            if self.n_syndrome % 2 == 0:  # Even number of syndromes - unlock
                self._locked = False
            else:  # Odd number of syndromes - lock
                self.lock()
        if node in boundary_nodes:
            self.lock()


class Cluster(ClusterBase):
    """A simple cluster containing nodes."""

    def __init__(self, _id: int) -> None:
        super().__init__(_id)
        self.nodes: Set[Node] = set()

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
        cluster1.preceding = self
        cluster2.preceding = self
        self.size = cluster1.root().size + cluster2.root().size
        self.n_syndrome = cluster1.root().n_syndrome + cluster2.root().n_syndrome
        self._locked = cluster1.root().locked() or cluster2.root().locked() or (self.n_syndrome % 2 == 1)

    def __iter__(self) -> Iterator[Node]:
        for element in self.elements:
            for node in element:
                yield node


def grow_clusters(
        g: nx.Graph,
        original_clusters: List[Set[Node]],
        syndrome: Set[Node],
        boundary_nodes: Set[Node]
) -> List[Set[Node]]:
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
        cluster = Cluster(cluster_id)
        cluster_id += 1

        is_boundary = any(node in boundary_nodes for node in nodes)

        # Add all nodes to the cluster
        for node in nodes:
            cluster.add(node, syndrome, boundary_nodes)

        clusters.append(cluster)

        # Classify cluster
        if is_boundary or cluster.n_syndrome % 2 == 0:
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

    # Grow algorithm
    while node_queue and active_clusters:
        u = node_queue.popleft()
        if u not in node_cluster or node_cluster[u].root().locked():
            continue

        for v in g.neighbors(u):
            edge = (u, v)

            # Handle edge growth
            if edge_growth[edge] == 0:
                edge_growth[edge] = 1
            elif edge_growth[edge] == 1:
                edge_growth[edge] = 2

                # Edge is fully grown, handle the other endpoint
                if v not in node_cluster:
                    # Add v to u's cluster
                    u_cluster = node_cluster[u].root()
                    u_cluster.add(v, syndrome, boundary_nodes)
                    node_cluster[v] = u_cluster
                    node_queue.append(v)
                elif node_cluster[v].root() != node_cluster[u].root():
                    # Merge the two clusters
                    u_cluster = node_cluster[u].root()
                    v_cluster = node_cluster[v].root()

                    # Only merge if u's cluster is still active
                    if not u_cluster.locked():
                        merged = MergedCluster(u_cluster, v_cluster, cluster_id)
                        cluster_id += 1

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

    # Collect final clusters
    final_clusters: List[Set[Node]] = []
    processed_roots = set()

    for node, cluster in node_cluster.items():
        root = cluster.root()
        if root not in processed_roots:
            processed_roots.add(root)
            nodes = set(root)
            final_clusters.append(nodes)

    return final_clusters
