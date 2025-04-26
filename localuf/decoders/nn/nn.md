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

### Dynamic peel

Remove all boundary nodes, with only internal nodes in the  `-> io, ie` clusters.

### Grow

`fusion` stage generates several clusters. The `grow` step uses the key insight from Union-Find decoders.

Initially `bo, ie, be` clusters are **locked**, only `io` clusters are **activated**. growing and merging `io` clusters
like what UF decoders did will finally lead to a valid solution. 