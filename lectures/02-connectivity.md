# Network Connectivity

## Introduction

Connectivity is a fundamental concept in network analysis that describes how well-connected a network is and how information or resources can flow through it. Understanding connectivity helps us analyze network robustness, identify critical components, and understand the structure of complex systems.

## Basic Concepts

### Walks and Paths

**Walk**: A sequence of nodes and edges where each edge connects consecutive nodes in the sequence.

Mathematically, a walk can be defined as:
$$W = (v_0, e_1, v_1, e_2, v_2, ..., e_n, v_n)$$

where $v_i$ is a node and $e_i$ is an edge.

**Path**: A walk with distinct nodes (no repeated nodes).

**Key Differences:**
- A walk can visit the same node multiple times
- A path visits each node at most once
- All paths are walks, but not all walks are paths

### Connected Components

A **connected component** is a maximal subgraph where every pair of nodes is connected by a path.

**Types of Connectivity:**
1. **Strongly Connected**: In directed graphs, every node can reach every other node
2. **Weakly Connected**: In directed graphs, the underlying undirected graph is connected
3. **Connected**: In undirected graphs, there exists a path between any two nodes

## Connectivity Measures

### 1. Node Connectivity

**Definition**: The minimum number of nodes that must be removed to disconnect the graph.

**Mathematical Definition:**
$$\kappa(G) = \min_{S \subset V} |S|$$

where $S$ is a node cut set that disconnects the graph.

**Properties:**
- $\kappa(G) = 0$ if and only if $G$ is disconnected
- $\kappa(G) = 1$ if $G$ has a cut vertex
- $\kappa(G) \leq \delta(G)$ where $\delta(G)$ is the minimum degree

### 2. Edge Connectivity

**Definition**: The minimum number of edges that must be removed to disconnect the graph.

**Mathematical Definition:**
$$\lambda(G) = \min_{F \subset E} |F|$$

where $F$ is an edge cut set that disconnects the graph.

**Properties:**
- $\lambda(G) = 0$ if and only if $G$ is disconnected
- $\lambda(G) = 1$ if $G$ has a bridge
- $\lambda(G) \leq \delta(G)$ where $\delta(G)$ is the minimum degree

### 3. Menger's Theorem

**Node Version**: The maximum number of node-disjoint paths between two nodes equals the minimum number of nodes whose removal disconnects them.

**Edge Version**: The maximum number of edge-disjoint paths between two nodes equals the minimum number of edges whose removal disconnects them.

## Network Robustness

### Giant Component Analysis

**Definition**: In large networks, the largest connected component is called the giant component.

**Properties:**
- Size scales with network size
- Emerges at a critical threshold
- Important for network functionality

### Percolation Theory

**Site Percolation**: Random removal of nodes
**Bond Percolation**: Random removal of edges

**Critical Threshold**: The point at which the giant component emerges or disappears.

**Mathematical Framework:**
- $p_c$: Critical probability
- $S(p)$: Size of giant component as function of $p$
- $S(p) = 0$ for $p < p_c$
- $S(p) > 0$ for $p > p_c$

## Assortativity

### Degree Assortativity

**Definition**: The tendency of nodes to connect to other nodes with similar degrees.

**Mathematical Definition:**
$$r = \frac{\sum_{xy} xy(e_{xy} - a_x b_y)}{\sigma_a \sigma_b}$$

where:
- $e_{xy}$ is the fraction of edges connecting nodes of degree $x$ and $y$
- $a_x = \sum_y e_{xy}$ and $b_y = \sum_x e_{xy}$
- $\sigma_a$ and $\sigma_b$ are standard deviations

**Interpretation:**
- $r > 0$: Assortative (high-degree nodes connect to high-degree nodes)
- $r < 0$: Disassortative (high-degree nodes connect to low-degree nodes)
- $r = 0$: No degree correlation

### Types of Assortativity

1. **Assortative Networks**:
   - Social networks (people tend to connect to others with similar characteristics)
   - Collaboration networks
   - Examples: Facebook friendships, scientific collaborations

2. **Disassortative Networks**:
   - Technological networks (hubs connect to many low-degree nodes)
   - Biological networks
   - Examples: Internet, protein interaction networks

## Connectivity Algorithms

### Finding Connected Components

**Depth-First Search (DFS) Algorithm:**
```python
def find_components_dfs(graph):
    visited = set()
    components = []
    
    for node in graph.nodes():
        if node not in visited:
            component = []
            dfs_visit(graph, node, visited, component)
            components.append(component)
    
    return components
```

**Breadth-First Search (BFS) Algorithm:**
```python
def find_components_bfs(graph):
    visited = set()
    components = []
    
    for node in graph.nodes():
        if node not in visited:
            component = []
            bfs_visit(graph, node, visited, component)
            components.append(component)
    
    return components
```

### Computing Node and Edge Connectivity

**Ford-Fulkerson Algorithm**: For computing maximum flow and minimum cut.

**Stoer-Wagner Algorithm**: For computing global minimum cut.

## Applications

### 1. Network Design
- Designing robust communication networks
- Identifying critical infrastructure
- Planning transportation systems

### 2. Social Network Analysis
- Understanding community structure
- Identifying influential spreaders
- Analyzing information flow

### 3. Biological Networks
- Protein interaction networks
- Metabolic networks
- Gene regulatory networks

### 4. Technological Networks
- Internet topology
- Power grids
- Transportation networks

## Computational Complexity

| Problem | Time Complexity | Notes |
|---------|----------------|-------|
| Connected components | O(V + E) | Linear time |
| Node connectivity | O(V³) | Polynomial time |
| Edge connectivity | O(V³) | Polynomial time |
| Giant component | O(V + E) | Linear time |
| Assortativity | O(E) | Linear time |

## References

- Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
- Bollobás, B. (2001). Random Graphs. Cambridge University Press.
- Albert, R., & Barabási, A. L. (2002). Statistical mechanics of complex networks. Reviews of Modern Physics, 74(1), 47. 