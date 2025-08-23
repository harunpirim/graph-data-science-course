# Signed Networks

## Introduction

Signed networks are graphs where edges can have positive or negative weights, representing relationships such as friendship/enmity, trust/distrust, or agreement/disagreement. These networks provide a richer representation of social and economic interactions than traditional unsigned networks.

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
1. **+++**: All positive edges (balanced)
2. **+--**: One positive, two negative edges (balanced)
3. **++-**: Two positive, one negative edge (unbalanced)
4. **---**: All negative edges (unbalanced)

### 3. Balance Index

**Definition**: Fraction of balanced triangles in the network.

**Mathematical Definition**:
$$B = \frac{\text{Number of balanced triangles}}{\text{Total number of triangles}}$$

## Centrality Measures

### 1. Signed Degree Centrality

**Definition**: Net degree considering both positive and negative connections.

**Mathematical Definition**:
$$d_i^{net} = d_i^+ - d_i^-$$

where:
- $d_i^+$ is the positive degree of node $i$
- $d_i^-$ is the negative degree of node $i$

### 2. Signed Closeness Centrality

**Definition**: Closeness centrality considering edge signs.

**Mathematical Definition**:
$$C_i = \frac{1}{\sum_{j \neq i} d_{ij}^{signed}}$$

where $d_{ij}^{signed}$ is the signed shortest path distance.

### 3. Signed Betweenness Centrality

**Definition**: Betweenness centrality considering edge signs.

**Algorithm**:
1. Calculate signed shortest paths
2. Count paths passing through each node
3. Normalize by total number of paths

### 4. Signed Eigenvector Centrality

**Definition**: Eigenvector centrality for signed networks.

**Mathematical Definition**:
$$x_i = \frac{1}{\lambda} \sum_j A_{ij} x_j$$

where $A_{ij}$ can be positive or negative.

## Community Detection

### 1. Signed Modularity

**Definition**: Extension of modularity to signed networks.

**Mathematical Definition**:
$$Q = \frac{1}{2m^+ + 2m^-} \sum_{ij} \left[ A_{ij} - \frac{k_i^+ k_j^+}{2m^+} + \frac{k_i^- k_j^-}{2m^-} \right] \delta(c_i, c_j)$$

where:
- $m^+$ and $m^-$ are the number of positive and negative edges
- $k_i^+$ and $k_i^-$ are positive and negative degrees

### 2. Signed Spectral Clustering

**Algorithm**:
1. Compute signed Laplacian matrix
2. Find eigenvectors of smallest eigenvalues
3. Apply k-means clustering to eigenvector coordinates

### 3. Balance-Based Clustering

**Algorithm**:
1. Find balanced subgraphs
2. Merge compatible balanced subgraphs
3. Assign remaining nodes to closest communities

## Link Prediction

### 1. Signed Link Prediction

**Problem**: Predict the sign of missing edges in signed networks.

**Features**:
- **Topological features**: Common neighbors, paths
- **Balance features**: Triangle balance
- **Structural features**: Degree, centrality

### 2. Balance-Based Prediction

**Principle**: Predict signs that maximize network balance.

**Algorithm**:
```python
def predict_sign_balance(graph, node1, node2):
    # Find common neighbors
    common_neighbors = set(graph.neighbors(node1)) & set(graph.neighbors(node2))
    
    # Count balanced and unbalanced triangles
    balanced = 0
    unbalanced = 0
    
    for neighbor in common_neighbors:
        sign1 = graph[node1][neighbor]['sign']
        sign2 = graph[node2][neighbor]['sign']
        if sign1 * sign2 == 1:  # Balanced
            balanced += 1
        else:  # Unbalanced
            unbalanced += 1
    
    # Predict based on balance
    if balanced > unbalanced:
        return 1  # Positive
    else:
        return -1  # Negative
```

## Applications

### 1. Social Networks

#### Online Social Networks
- **Positive edges**: Friends, followers, likes
- **Negative edges**: Blocks, unfriends, dislikes
- **Applications**: Recommendation systems, sentiment analysis

#### Political Networks
- **Positive edges**: Alliances, agreements
- **Negative edges**: Conflicts, disagreements
- **Applications**: Political analysis, conflict resolution

### 2. Economic Networks

#### Trade Networks
- **Positive edges**: Trade agreements, partnerships
- **Negative edges**: Trade disputes, sanctions
- **Applications**: Economic modeling, policy analysis

#### Financial Networks
- **Positive edges**: Investments, loans
- **Negative edges**: Defaults, conflicts
- **Applications**: Risk assessment, financial stability

### 3. Biological Networks

#### Protein Interaction Networks
- **Positive edges**: Activating interactions
- **Negative edges**: Inhibiting interactions
- **Applications**: Drug discovery, disease understanding

#### Gene Regulatory Networks
- **Positive edges**: Gene activation
- **Negative edges**: Gene repression
- **Applications**: Gene therapy, disease treatment

## Analysis Methods

### 1. Balance Analysis

```python
def analyze_balance(graph):
    triangles = list(nx.triangles(graph).keys())
    balanced_count = 0
    total_count = 0
    
    for triangle in triangles:
        edges = [(triangle[0], triangle[1]), 
                (triangle[1], triangle[2]), 
                (triangle[2], triangle[0])]
        
        signs = [graph[edge[0]][edge[1]]['sign'] for edge in edges]
        product = signs[0] * signs[1] * signs[2]
        
        if product == 1:
            balanced_count += 1
        total_count += 1
    
    balance_index = balanced_count / total_count
    return balance_index
```

### 2. Community Detection

```python
def signed_community_detection(graph):
    # Compute signed Laplacian
    laplacian = nx.laplacian_matrix(graph).toarray()
    
    # Find eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    
    # Use second smallest eigenvector for clustering
    second_eigenvector = eigenvectors[:, 1]
    
    # Apply k-means
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=2, random_state=42)
    communities = kmeans.fit_predict(second_eigenvector.reshape(-1, 1))
    
    return communities
```

### 3. Link Prediction

```python
def signed_link_prediction(graph, test_edges):
    predictions = []
    
    for edge in test_edges:
        node1, node2 = edge
        
        # Calculate features
        common_neighbors = len(set(graph.neighbors(node1)) & set(graph.neighbors(node2)))
        balance_score = calculate_balance_score(graph, node1, node2)
        
        # Simple prediction rule
        if balance_score > 0:
            predicted_sign = 1
        else:
            predicted_sign = -1
            
        predictions.append(predicted_sign)
    
    return predictions
```

## Challenges and Limitations

### 1. Data Quality

**Issues**:
- Missing negative edges
- Biased reporting of negative relationships
- Temporal changes in relationship signs

**Solutions**:
- Multiple data sources
- Validation techniques
- Temporal analysis

### 2. Computational Complexity

**Issues**:
- NP-hard problems (balance optimization)
- Large-scale networks
- Dynamic networks

**Solutions**:
- Approximation algorithms
- Sampling techniques
- Incremental updates

### 3. Interpretability

**Issues**:
- Complex balance patterns
- Context-dependent signs
- Cultural differences

**Solutions**:
- Domain-specific analysis
- Contextual information
- Cross-cultural studies

## Future Directions

### 1. Dynamic Signed Networks

**Research Areas**:
- Temporal evolution of signs
- Balance dynamics
- Prediction of sign changes

### 2. Machine Learning Integration

**Research Areas**:
- Deep learning for signed networks
- Graph neural networks
- Representation learning

### 3. Multilayer Signed Networks

**Research Areas**:
- Multiple relationship types
- Cross-layer balance
- Complex interactions

## References

- Harary, F. (1953). On the notion of balance of a signed graph. Michigan Mathematical Journal, 2(2), 143-146.
- Cartwright, D., & Harary, F. (1956). Structural balance: a generalization of Heider's theory. Psychological Review, 63(5), 277.
- Leskovec, J., Huttenlocher, D., & Kleinberg, J. (2010). Predicting positive and negative links in online social networks. Proceedings of the 19th International Conference on World Wide Web, 641-650.
- Yang, B., Cheung, W. K., & Liu, J. (2007). Community mining from signed social networks. IEEE Transactions on Knowledge and Data Engineering, 19(10), 1333-1348. 