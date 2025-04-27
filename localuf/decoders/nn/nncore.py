from typing import Any

from matplotlib import pyplot as plt

from localuf._base_classes import Code
from localuf.type_aliases import Node, Edge
import numpy as np
import networkx as nx


class NNCore:
    Cluster = set[Node]

    class STRATEGY:
        GROW = 1
        DYNAMIC_PEEL = 2
        NEAREST = 3

    def __init__(self, code: Code) -> None:
        self._code = code
        self._g = code.GRAPH.copy()

        """ 
        Build index table for boundary and internal nodes.
            Note that we need an extra dict to map Node --> index.
            Node<=>tuple[int, int], (x,y) coordination in the graph.
            Edge<=>tuple[Node, Node].

        Although the decoder previously used `int` as node index, we now prefer Node as the LocalUF library suggests.
        """
        self._boundary_nodes: tuple[Node, ...] = tuple(v for v in self._g if self._code.is_boundary(v))
        self._internal_nodes: tuple[Node, ...] = tuple(v for v in self._g if not self._code.is_boundary(v))
        self._boundary_node2idx: dict[Node, int] = {v: i for i, v in enumerate(self._boundary_nodes)}
        self._internal_node2idx: dict[Node, int] = {v: i for i, v in enumerate(self._internal_nodes)}
        self._syndrome_list: dict[Node, bool] = {}

        print(
            f"Boundary nodes: {self._boundary_nodes}\n Boundary node to index: {self._boundary_node2idx}\n"
            f"Internal nodes: {self._internal_nodes}\n Internal node to index: {self._internal_node2idx}"
        )

        # After that, we build an APSP table for the graph,
        # and find the nearest boundary node for each internal node.
        pred, apsp_result = nx.floyd_warshall_predecessor_and_distance(self._g, weight="weight")
        self._apsp: dict[Node, dict[Node, float]] = {a: dict(b) for a, b in apsp_result.items()}
        self._pred: dict[Node, dict[Node, Node]] = {a: dict(b) for a, b in pred.items()}

        self._distance_to_boundary: dict[Node, float] = {}
        self._nearest_boundary: dict[Node, Node] = {}
        for v in self._internal_nodes:
            distance_to_boundaries = {b: self._apsp[v][b] for b in self._boundary_nodes}
            nearest_item = min(distance_to_boundaries.items(), key=lambda x: x[1])
            self._nearest_boundary[v] = nearest_item[0]
            self._distance_to_boundary[v] = nearest_item[1]
        print(f"Distance to boundary: {self._distance_to_boundary}, target {self._nearest_boundary}\n")

    def decode(self, syndrome: set[Node], post_process: int) -> None:
        """
        The decode syndrome contains the stages below.
        - Fusion.
        - Evenize.
        - Match.

        Correction should be stored in `self.correction`.
        """

        self._syndrome_list: dict[Node, bool] = {v: True if v in syndrome else False for v in self._g}
        print(f"Syndrome list: {self._syndrome_list}\n")
        fusion_list: list[set[Node]] = self.fusion(syndrome)
        print(f"Fusion list: {fusion_list}\n")

        # Post process the invalid clusters.
        valid_clusters: list[set[Node]] | None = self.post_process(fusion_list, post_process)
        self.correct(valid_clusters)

    def fusion(self, syndrome: set[Node]) -> list[set[Node]]:
        """
        Foreach syndrome Node, find its nearest boundary node, and add corresponding edges in another "neighbor" graph.
        The connected components of this graph form the preliminary clusters.
        We use nested set to keep it immutable.
        """
        edge_list: list[Edge] = []
        for v in syndrome:
            assert v not in self._boundary_nodes
            # Find the nearest boundary node.
            nearest_boundary = self._nearest_boundary[v]
            _sssp: dict[Node, float] = self._apsp[v]
            nearest_node = None

            for node, dist in _sssp.items():
                if node != v and self._syndrome_list[node]:
                    nearest_node = node
                    node_dist = dist
                    break

            print(
                f"{v} nearest boundary: {nearest_boundary}, nearest node: {nearest_node}, distance: {node_dist}"
            )
            assert nearest_node is not None
            if node_dist > 2 * self._distance_to_boundary[v]:
                edge_list.append((v, nearest_boundary))
            else:
                edge_list.append((v, nearest_node))

        # Build a graph with the edges.
        fusion_graph = nx.Graph()
        fusion_graph.add_edges_from(edge_list)
        # Find connected components.
        connected_components = nx.connected_components(fusion_graph)
        # Collect the connected components into a list of sets.
        return [set(component) for component in connected_components]

    def post_process(self, fusion_res: list[set[Node]], strategy: int) -> list[set[Node]] | None:
        """Post process the invalid clusters.
        - GROW
        - DYNAMIC_PEEL
        - NEAREST
        """
        match strategy:
            case self.STRATEGY.GROW:
                return self.grow(original=fusion_res)
            case self.STRATEGY.DYNAMIC_PEEL:
                return self.dynamic_peel(original=fusion_res)
            case self.STRATEGY.NEAREST:
                return self.nearest(original=fusion_res)
            case _:
                raise NotImplementedError

    def correct(self, clusters: list[set[Node]] | None) -> None:
        self.correction: set[Edge] = set()
        if clusters is None:
            return

        def shortest_path(uv: tuple[Node, Node]) -> set[Edge]:
            u, v = uv
            path = []
            assert self._pred[u][v]
            while u != v:
                path.append((u, self._pred[u][v]))
                u = self._pred[u][v]
            return set(path)

        """ Correction should be stored in `self.correction`.
        Generate correction edges for each cluster.
        1. For each cluster, choose pairs (u,v), each pair has a shortest path p_i.
        2. For each cluster i, its match edges M_i are p_0 xor p_1 xor ... xor p_i.
        3. The correction edges are M_0 xor M_1 xor ... xor M_i.
        """
        for cluster in clusters:
            assert len(cluster) % 2 == 0, f"Cluster {cluster} size should be even"

            # Convert cluster to list for easier pairing
            cluster_nodes = list(cluster)

            # For each pair of nodes in the cluster, find the shortest path
            cluster_correction: set[Edge] = set()
            for i in range(0, len(cluster_nodes), 2):
                u, v = cluster_nodes[i], cluster_nodes[i + 1]
                # Find the shortest path between u and v
                path: set[Edge] = shortest_path((u, v))
                cluster_correction = cluster_correction.symmetric_difference(path)

            # Add path edges to correction (using symmetric difference to handle XOR of edges)
            self.correction = self.correction.symmetric_difference(cluster_correction)

    """ `io, bo, ie, be` stands for `internal / boundary odd / even cluster`.
    """

    def nearest(self, original: list[set[Node]]) -> list[set[Node]] | None:
        if not original:
            return None
        """
        #### Nearest
        Find nearest boundary node or odd cluster for invalid clusters.

        Remove boundary node from `bo`, forming `io, ie, be` clusters.
        Foreach `io` cluster, find its nearest `io` cluster. Choice is made between 2 options:
        1. merge `io` cluster with its nearest `io` cluster.
        2. add a boundary node to `io` cluster.
        """
        # Classify each cluster as internal/boundary and odd/even
        io_clusters = []  # internal odd clusters
        ie_clusters = []  # internal even clusters
        # bo_clusters = []  # boundary odd clusters
        be_clusters = []  # boundary even clusters

        for cluster in original:
            has_boundary = any(node in self._boundary_nodes for node in cluster)
            is_odd = len(cluster) % 2 == 1

            if has_boundary:
                if is_odd:
                    # This is a bo cluster. We need to remove the boundary node.
                    # This will form an internal (io or ie) cluster.
                    boundary_nodes_in_cluster = [node for node in cluster if node in self._boundary_nodes]
                    new_cluster = cluster.difference(set(boundary_nodes_in_cluster))
                    if not new_cluster:
                        assert False
                    (io_clusters if len(new_cluster) % 2 == 1 else ie_clusters).append(new_cluster)
                else:
                    be_clusters.append(cluster)
            else:  # internal cluster
                if is_odd:
                    io_clusters.append(cluster)
                else:
                    ie_clusters.append(cluster)

        # Process io clusters by finding nearest option for each
        # Create a copy of the list to iterate over to avoid modification during iteration
        io_clusters_original = io_clusters.copy()
        for io_cluster in io_clusters_original:
            # The io_clusters may have been removed before this point, so check and skip this cluster.
            if io_cluster not in io_clusters:
                continue

            # Option 1: Find nearest io_cluster to merge with
            min_dist_to_io = float("inf")
            nearest_io = None

            for other_io in io_clusters:
                if other_io == io_cluster:
                    continue
                # Calculate minimum distance between two clusters
                dist = min(self._apsp[n1][n2] for n1 in io_cluster for n2 in other_io)
                min_dist_to_io = dist if dist < min_dist_to_io else min_dist_to_io
                nearest_io = other_io if dist < min_dist_to_io else nearest_io

            # Option 2: Find nearest boundary node
            min_dist_to_boundary = float("inf")
            nearest_boundary = None

            for node in io_cluster:
                nearest_boundary_node = self._nearest_boundary[node]
                if nearest_boundary_node is None:
                    continue
                dist = self._apsp[node][nearest_boundary_node]
                if dist < min_dist_to_boundary:
                    min_dist_to_boundary = dist
                    nearest_boundary = nearest_boundary_node

            # Make the decision: merge or add boundary
            if nearest_io is not None and min_dist_to_io <= 2 * min_dist_to_boundary:
                assert io_cluster in io_clusters and nearest_io in io_clusters
                # Merge with nearest io cluster
                io_clusters.remove(io_cluster)
                io_clusters.remove(nearest_io)
                merged = io_cluster.union(nearest_io)
                # The merged cluster has even parity, so it becomes internal even
                ie_clusters.append(merged)
            else:
                # Add boundary node
                assert io_cluster in io_clusters
                io_clusters.remove(io_cluster)
                boundary_cluster = io_cluster.union({nearest_boundary})
                be_clusters.append(boundary_cluster)

        # Combine the processed clusters
        final_clusters: list[set[Node]] = ie_clusters + be_clusters

        """ Make several assertions to check the validity of the clusters.
        1. No internal odd clusters should remain.
        2. No duplicate nodes exist between result clusters.
        3. Internal nodes in original clusters should be the same as in result clusters.
        """
        assert not io_clusters
        original_internal_nodes: set[Node] = {node for cluster in original for node in cluster}
        result_internal_nodes = [node for cluster in final_clusters for node in cluster]
        assert len(result_internal_nodes) == len(
            set(result_internal_nodes)
        ), "Duplicate nodes found between clusters"
        assert original_internal_nodes == set(
            result_internal_nodes
        ), "Internal nodes in original and result clusters do not match"

        return final_clusters

    def grow(self, original: list[set[Node]]) -> list[set[Node]] | None:
        """Grow the clusters."""
        raise NotImplementedError

    def dynamic_peel(self, original: list[set[Node]]) -> list[set[Node]] | None:
        """Dynamic peel the clusters."""
        raise NotImplementedError
