# Signed Networks

## Introduction

Signed networks are graphs where edges can have positive (+) or negative (-) weights, representing relationships such as friendship/enmity, trust/distrust, or agreement/disagreement. These networks provide a richer representation of social and economic interactions than traditional unsigned networks.

## Basic Concepts

### 1. Signed Network Structure

**Definition**: A signed network is a graph $G = (V, E, \sigma)$ where:
- $V$ is the set of nodes
- $E$ is the set of edges
- $\sigma: E \rightarrow \{+1, -1\}$ is the sign function

**Edge Types**:
- **Positive edges** ($+1$): Friendship, trust, agreement
- **Negative edges** ($-1$): Enmity, distrust, disagreement

### 2. Adjacency Matrix Representation

**Signed Adjacency Matrix**:
$$A_{ij} = \begin{cases}
+1 & \text{if } (i,j) \text{ is a positive edge} \\
-1 & \text{if } (i,j) \text{ is a negative edge} \\
0 & \text{if no edge exists}
\end{cases}$$

## Balance Theory

### 1. Structural Balance

**Definition**: A signed network is balanced if it can be partitioned into two groups such that all positive edges are within groups and all negative edges are between groups.

**Mathematical Definition**: A triangle is balanced if the product of its edge signs is positive:
$$\sigma_{ij} \cdot \sigma_{jk} \cdot \sigma_{ki} = +1$$

### 2. Balance Conditions

**Triangle Types**:
1. **+++**: All positive edges (balanced) - "Friend of friend is friend"
2. **+--**: One positive, two negative edges (balanced) - "Enemy of enemy is friend"
3. **++-**: Two positive, one negative edge (unbalanced) - "Two friends with common enemy"
4. **---**: All negative edges (unbalanced) - "Three mutual enemies"

### 3. Balance Index

**Definition**: Fraction of balanced triangles in the network.

**Mathematical Definition**:
$$B = \frac{\text{Number of balanced triangles}}{\text{Total number of triangles}}$$


## Centrality Measures

### Issues with Standard Centrality

**Problems**:
1. **Shortest Paths**: How to define "shortest" when edges have different signs?
2. **Path Meaning**: A path through enemies is fundamentally different from a path through friends
3. **Edge Treatment**: Standard algorithms treat all edges equally, ignoring signs
4. **Existence**: Some measures (like eigenvector centrality) may not even exist for signed networks!

### 1. Signed Degree Centrality

**Definition**: Net degree considering both positive and negative connections.

**Mathematical Definition**:
$$d_i^{net} = d_i^+ - d_i^-$$

where:
- $d_i^+$ is the positive degree of node $i$
- $d_i^-$ is the negative degree of node $i$

**Interpretation**:
- High positive net degree: Well-liked, influential
- High negative net degree: Controversial, many enemies
- Near zero: Balanced relationships

### 2. Walk-Based Centrality 

Walk-based centrality is the most reliable method for signed networks. It properly accounts for how influence propagates along all paths with signs multiplying.

**Reference**: Liu et al. (2020) "A simple approach for quantifying node centrality in signed and directed social networks"

**Key Innovation**: Instead of shortest paths, consider ALL walks where effects propagate and signs multiply.

#### Direct Effect Formula

The direct effect of node $i$ on node $j$ is:
$$a_{ij} = \frac{\sigma_{ij}}{d_j}$$

where:
- $\sigma_{ij}$ is the sign of edge $(i,j)$ ($+1$ or $-1$)
- $d_j$ is the degree of node $j$

**Intuition**: Effect is stronger when target has fewer connections.

#### Indirect Effects and Sign Propagation

For a walk $i \rightarrow k \rightarrow j$, the indirect effect is:
$$\text{effect}(i \rightarrow j \text{ via } k) = a_{ik} \times a_{kj}$$

**Sign Propagation Rules**:
- $(+) \times (+) = +$ : Positive influence through positive intermediary
- $(+) \times (-) = -$ : Positive becomes negative through negative intermediary
- $(-) \times (-) = +$ : Negative through negative (enemy of enemy)

#### Total Effect (TE)

**Definition**: Sum of effects along ALL walks up to length $n$:
$$TE_{ij}^{(n)} = \sum_{l=1}^{n} \sum_{\text{walks of length } l} \text{effect along walk}$$

Total Effect of node $i$ on whole network:
$$TE_i = \sum_{j=1}^{N} TE_{ij}^{(n)}$$

**Interpretation**: Overall influence magnitude, regardless of sign.

#### Net Effect (NE)

**Definition**: Positive effects minus negative effects:
$$NE_{ij}^{(n)} = E_{ij}^{(n)+} - E_{ij}^{(n)-}$$

Net Effect of node $i$:
$$NE_i = \sum_{j=1}^{N} NE_{ij}^{(n)}$$

**Interpretation**: 
- $NE_i > 0$: Predominantly positive influence
- $NE_i < 0$: Predominantly negative influence
- $NE_i \approx 0$: Balanced positive and negative influence


### 3. Signed Betweenness Centrality (Approximation Only)

**Approach: Structure-Weighted Approximation**

```python
def signed_betweenness_approximation(G):
    """
    Approximate betweenness for signed networks.
    
    WARNING: This is an approximation with limitations.
    Consider using walk-based centrality instead.
    """
    # Create unsigned version for path counting
    G_unsigned = nx.Graph()
    G_unsigned.add_nodes_from(G.nodes())
    G_unsigned.add_edges_from(G.edges())
    
    # Calculate standard betweenness
    betweenness = nx.betweenness_centrality(G_unsigned)
    
    # Weight by local sign environment
    weighted_betweenness = {}
    for node in G.nodes():
        pos_edges = sum(1 for _, _, d in G.edges(node, data=True) 
                       if d.get('sign', 1) == 1)
        neg_edges = sum(1 for _, _, d in G.edges(node, data=True) 
                       if d.get('sign', 1) == -1)
        total_edges = pos_edges + neg_edges
        
        if total_edges > 0:
            sign_ratio = (pos_edges - neg_edges) / total_edges
            weighted_betweenness[node] = betweenness[node] * (1 + sign_ratio) / 2
        else:
            weighted_betweenness[node] = 0.0
    
    return weighted_betweenness
```

---

### 4. Signed Closeness Centrality

```python
def signed_closeness_harmonic(G):
    """
    Harmonic closeness using only positive edges.
    
    Only considers reachability through friendly relationships.
    """
    # Create subgraph with only positive edges
    G_positive = nx.Graph()
    G_positive.add_nodes_from(G.nodes())
    for u, v, data in G.edges(data=True):
        if data.get('sign', 1) == 1:
            G_positive.add_edge(u, v)
    
    closeness = {}
    for node in G.nodes():
        harmonic_sum = 0.0
        for target in G.nodes():
            if node != target:
                try:
                    distance = nx.shortest_path_length(G_positive, node, target)
                    harmonic_sum += 1.0 / distance
                except nx.NetworkXNoPath:
                    pass  # No path through positive edges
        
        n = len(G.nodes())
        closeness[node] = harmonic_sum / (n - 1) if n > 1 else 0
    
    return closeness
```

**Interpretation**:
- High closeness: Well-connected through friendly relationships
- Low closeness: Isolated or only reachable through enemies

---

### 5. Signed Eigenvector Centrality

The Perron-Frobenius theorem does NOT apply when adjacency matrix has negative entries.

**When Eigenvector Centrality Fails**:
1. No dominant positive eigenvalue
2. Multiple eigenvalues with same magnitude
3. Complex eigenvalues with imaginary components
4. Eigenvector with mixed signs (no clear interpretation)

**Mathematical Definition**:
$$x_i = \frac{1}{\lambda} \sum_j A_{ij} x_j$$

where $A_{ij}$ can be positive or negative.

**Python Implementation with Safety Checks**:

```python
def signed_eigenvector_centrality(G, tol=1e-6):
    """
    Calculate eigenvector centrality for signed networks.
    
    WARNING: May not exist! Always check return status.
    
    Returns
    -------
    centrality : dict or None
        Centrality values if computable, None otherwise
    status : str
        Status message explaining result
    """
    nodes = list(G.nodes())
    n = len(nodes)
    
    if n == 0:
        return None, "Empty graph"
    
    # Build signed adjacency matrix
    A = nx.adjacency_matrix(G, nodelist=nodes, weight='sign').toarray()
    
    # Find eigenvalues
    try:
        eigenvalues, eigenvectors = np.linalg.eig(A)
        eigenvalues = eigenvalues.real
    except:
        return None, "Failed to compute eigenvalues"
    
    # Find largest eigenvalue by magnitude
    max_idx = np.argmax(np.abs(eigenvalues))
    lambda_max = eigenvalues[max_idx]
    
    # Check 1: Dominant eigenvalue?
    sorted_eigs = sorted(np.abs(eigenvalues), reverse=True)
    if len(sorted_eigs) > 1 and np.abs(sorted_eigs[0] - sorted_eigs[1]) < tol:
        return None, "No dominant eigenvalue (multiple with same magnitude)"
    
    # Check 2: Positive eigenvalue?
    if lambda_max <= 0:
        return None, f"No positive dominant eigenvalue (λ={lambda_max:.4f})"
    
    # Get eigenvector
    eigenvector = eigenvectors[:, max_idx].real
    
    # Check 3: All same sign?
    if np.all(eigenvector >= 0) or np.all(eigenvector <= 0):
        eigenvector = np.abs(eigenvector)
        eigenvector = eigenvector / eigenvector.sum()
        centrality = {node: eigenvector[i] for i, node in enumerate(nodes)}
        return centrality, f"Success (λ={lambda_max:.4f})"
    else:
        return None, "Eigenvector has mixed signs"

# Usage
centrality, status = signed_eigenvector_centrality(G)
if centrality is None:
    print(f"Cannot compute: {status}")
    print("Using walk-based centrality instead")
    centrality = signed_walk_effect(G)
```

## Community Detection

### 1. Signed Modularity

**The Problem**: Standard Newman modularity treats all edges equally.

**Correct Formula**: Gómez et al. signed modularity with parameter $\alpha$:

$$Q_{signed} = \alpha \cdot Q(G^+) + (1-\alpha) \cdot Q(G^-)$$

Where:
- $Q(G^+)$: Modularity for positive edges (want within communities)
- $Q(G^-)$: Modularity for negative edges (want between communities)
- $\alpha \in [0,1]$: Balance parameter (typically 0.5)

### 2. Signed Spectral Clustering

**Algorithm**:
```python
def signed_spectral_clustering(G, num_communities=2):
    """Spectral clustering for signed networks"""
    nodes = list(G.nodes())
    n = len(nodes)
    
    # Create signed adjacency matrix
    A = nx.adjacency_matrix(G, weight='sign').toarray()
    
    # Compute signed Laplacian
    D = np.diag(np.sum(np.abs(A), axis=1))
    L = D - A
    
    # Find eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    
    # Use Fiedler vector (second smallest eigenvalue)
    fiedler_vector = eigenvectors[:, 1]
    
    # Partition based on sign
    communities = {}
    for i, node in enumerate(nodes):
        communities[node] = 0 if fiedler_vector[i] < 0 else 1
    
    return communities
```

### 3. Community Quality Metrics

```python
def analyze_community_quality(G, communities):
    """Analyze quality of detected communities in signed network"""
    internal_pos = 0  # Good
    internal_neg = 0  # Bad
    external_pos = 0  # Bad
    external_neg = 0  # Good
    
    for u, v, data in G.edges(data=True):
        sign = data['sign']
        same_community = (communities[u] == communities[v])
        
        if same_community:
            if sign == 1:
                internal_pos += 1
            else:
                internal_neg += 1
        else:
            if sign == 1:
                external_pos += 1
            else:
                external_neg += 1
    
    good_edges = internal_pos + external_neg
    bad_edges = internal_neg + external_pos
    quality = good_edges / (good_edges + bad_edges) if (good_edges + bad_edges) > 0 else 0
    
    return {
        'internal_positive': internal_pos,
        'internal_negative': internal_neg,
        'external_positive': external_pos,
        'external_negative': external_neg,
        'quality_score': quality
    }
```

---

## Link Prediction

### Balance-Based Prediction

**Principle**: Predict signs that maximize structural balance.

**Algorithm**:

```python
def predict_edge_sign(G, node1, node2):
    """
    Predict edge sign using balance theory.
    
    Returns
    -------
    predicted_sign : int (1 or -1)
        Predicted sign
    confidence : float
        Prediction confidence
    """
    # Find common neighbors
    neighbors1 = set(G.neighbors(node1))
    neighbors2 = set(G.neighbors(node2))
    common_neighbors = neighbors1.intersection(neighbors2)
    
    if len(common_neighbors) == 0:
        return None, 0.0
    
    vote_positive = 0
    vote_negative = 0
    
    for neighbor in common_neighbors:
        sign1 = G[node1][neighbor]['sign']
        sign2 = G[node2][neighbor]['sign']
        
        # Balance rule: if signs match → predict positive
        if sign1 * sign2 == 1:
            vote_positive += 1
        else:
            vote_negative += 1
    
    predicted_sign = 1 if vote_positive > vote_negative else -1
    confidence = max(vote_positive, vote_negative) / (vote_positive + vote_negative)
    
    return predicted_sign, confidence
```

---

## Complete Analysis Example

```python
import networkx as nx
import numpy as np

# Create signed network
G = nx.Graph()
G.add_edges_from([
    ('A', 'B', {'sign': 1}),
    ('B', 'C', {'sign': 1}),
    ('C', 'A', {'sign': 1}),  # Community 1: all friends
    ('D', 'E', {'sign': 1}),
    ('E', 'F', {'sign': 1}),  # Community 2: friends
    ('A', 'D', {'sign': -1}),
    ('B', 'E', {'sign': -1}),
    ('C', 'F', {'sign': -1})  # Between communities: enemies
])

# 1. Calculate balance index
balance_idx = calculate_balance_index(G)
print(f"Balance Index: {balance_idx:.3f}")

# 2. Calculate centrality (USE WALK-BASED - RECOMMENDED)
centrality = signed_walk_effect(G, max_steps=3)
print("\nTop 3 by Total Effect:")
sorted_nodes = sorted(centrality.items(), 
                     key=lambda x: x[1]['total_effect'], 
                     reverse=True)[:3]
for node, metrics in sorted_nodes:
    print(f"  {node}: TE={metrics['total_effect']:.3f}, "
          f"NE={metrics['net_effect']:.3f}")

# 3. Detect communities
communities = signed_spectral_clustering(G, num_communities=2)
print(f"\nCommunities: {communities}")

# 4. Calculate CORRECT signed modularity
Q_signed = signed_modularity(G, communities, alpha=0.5)
print(f"Signed Modularity: {Q_signed:.3f}")

# 5. Evaluate community quality
quality = analyze_community_quality(G, communities)
print(f"Community Quality: {quality['quality_score']:.3f}")

# 6. Link prediction
for node1 in ['A', 'D']:
    for node2 in ['B', 'E']:
        if not G.has_edge(node1, node2):
            pred_sign, conf = predict_edge_sign(G, node1, node2)
            if pred_sign:
                sign_str = '+' if pred_sign == 1 else '-'
                print(f"Predict {node1}-{node2}: {sign_str} "
                      f"(confidence: {conf:.2%})")
```
---

## References

### Critical Papers for Signed Networks

1. **Harary, F. (1953)**. "On the notion of balance of a signed graph." Michigan Mathematical Journal, 2(2), 143-146.
   - Original balance theory

2. **Cartwright, D., & Harary, F. (1956)**. "Structural balance: a generalization of Heider's theory." Psychological Review, 63(5), 277.
   - Extended balance theory

3. **Liu, W. C., Huang, L. C., Liu, C. W. J., & Jordán, F. (2020)**. "A simple approach for quantifying node centrality in signed and directed social networks." Applied Network Science, 5(1), 1-18.
   - **Walk-based centrality (RECOMMENDED METHOD)**
   - [Read online](https://appliednetsci.springeropen.com/articles/10.1007/s41109-020-00288-w)

4. **Bonacich, P., & Lloyd, P. (2004)**. "Calculating status with negative relations." Social Networks, 26(4), 331-338.
   - Eigenvector centrality for signed networks

5. **Traag, V. A., & Bruggeman, J. (2009)**. "Community detection in networks with positive and negative links." Physical Review E, 80(3), 036115.
   - Signed modularity

6. **Esmailian, P., & Jalili, M. (2015)**. "Community detection in signed networks: the role of negative ties in different scales." Scientific Reports, 5, 14339.
   - Shows inconsistencies in signed modularity

7. **Everett, M. G., & Borgatti, S. P. (2014)**. "Networks containing negative ties." Social Networks, 38, 111-120.
   - PN centrality measure

8. **Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010)**. "Predicting positive and negative links in online social networks." Proceedings of the 19th International Conference on World Wide Web, 641-650.
   - Link prediction in signed networks

