# About Nearest Decoder

## Post-processing

The Nearest Decoder has a post-processing step that can be applied to the output of fusion result.

1. `grow`
2. `dynamic_peel`
3. `nearest`

### Nearest

`io, bo, ie, be` stands for `internal/boundary odd/even cluster`.

#### Nearest #1

Remove boundary node from `bo`, forming `io, ie, be` clusters.

Foreach `io` cluster, find its nearest `io` cluster. Choice is made between 2 options:

1. merge `io` cluster with its nearest `io` cluster.
2. add a boundary node to `io` cluster.

#### Nearest #2

The nearest #2 is an improved version of nearest #1. This strategy takes merging with `ie, be` clusters into account.

While `io` is merged with an `ie` cluster, an extra boundary node is added to the new cluster. Merging with `be` cluster
is the opposite, depending on the number of boundary nodes, we should remove odd number of boundary nodes.

### Dynamic peel

Remove all boundary nodes, with only internal nodes in the  `-> io, ie` clusters.

### Grow

`fusion` stage generated several clusters. The `grow` step uses the key insight from Union-Find decoders. The Union-Find
decoder utilizes the data structure of Union-Find to grow and merge clusters.

A cluster is a set of nodes, or merged from several clusters, forming a larger cluster (UF data structure). Here we use
a tree to represent the cluster.

``` Python
class ClusterBase:
    id: int
    size: int
    n_syndrome: int
    preceding: ClusterBase
    nodes: set[Node]
    
    @virtual
    def __iter__():
        ...... # Iterate nodes in the cluster
    @virtual
    def root() -> ClusterBase:
        ...... # Get the root id of this cluster
    def locked() -> bool:
        self._locked
        root().locked() # Get the state of this cluster
    
    def add(node: Node):
        ...... # Add a node to self.nodes
        self.size += 1
        if node in syndrome:
            assert self is locked and self.n_syndrome % 2 == 1
            self.n_syndrome += 1
            lock() this cluster (root)
        if node in boundary:
            assert self is locked
            lock() this cluster (root)

class Cluster(ClusterBase):
    nodes = [...]
    def __iter__():
        ...... # Iterate nodes in the cluster

class MergedCluster(ClusterBase):
    element: list[ClusterBase] = [...]
    def __iter__():
        ...... # Iterate nodes in the cluster
```

Initially `bo, ie, be` clusters are **locked**, only `io` clusters are **activated**. Growing and merging `io` clusters
like what UF decoders did will finally lead to a valid solution.

Each edge has three growth states: 0(zero), 1(half), 2(full). When an edge has growth value of 2 after being growed,
fetch the incident node on the other side of the edge.

If the node is not assigned to a cluster, it will be added to the very cluster of the edge. However, if the node is
already assigned to a cluster, we need to merge the two clusters. If the node is a boundary node, we need to

``` python
1. Each cluster is assigned as `io, ie, bo, be`.
2. `io` clusters are **activated**, while others are **locked**.

node_queue = [node in io (activate) clusters]
cluster_roots = [original clusters]
node_cluster = dict{node->cluster}
node_assigned = dict{node->bool}

edge_growth = dict{edge->int(0)}

while exists activate clusters:
    u = node_queue.pop()
    for e in u.adj_edges:
        (assert e.growth == 0/1/2)
        if e.growth == 2:
            continue
        else if e.growth == 0:
            e.growth = 1
        else if e.growth == 1:
            e.growth = 2
            # Add incident node to the cluster. Check if the 
            # node is already assigned to a cluster.
            v = e.get_other_node(u)
            if v.node_cluster is None:
                node_cluster[v] = node_cluster[u]
                node_cluster[u].add(v)
            else if v.cluster != u.cluster:
                cluster_roots.remove(
                    u.cluster.root and v.cluster.root)
                cluster_roots.append(
                    MergedCluster(u.cluster, v.cluster))
```