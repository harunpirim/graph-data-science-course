# node2vec: Scalable Feature Learning for Networks

## A Comprehensive Tutorial with Examples

---

## Table of Contents

1. [Introduction and Motivation](#introduction-and-motivation)
2. [Background: From Word2Vec to node2vec](#background)
3. [Random Walks on Graphs](#random-walks-on-graphs)
4. [The Key Innovation: Biased Random Walks](#biased-random-walks)
5. [The node2vec Algorithm](#the-node2vec-algorithm)
6. [Mathematical Formulation](#mathematical-formulation)
7. [Complete Worked Example](#complete-worked-example)
8. [Implementation Guide](#implementation-guide)
9. [Comparison with Other Methods](#comparison)
10. [Applications and Use Cases](#applications)

---

## Introduction and Motivation

### What is node2vec?

**node2vec** is an algorithmic framework for learning continuous feature representations (embeddings) for nodes in networks. It was introduced by Grover & Leskovec in 2016 and extends the earlier DeepWalk algorithm.

### The Big Idea

**Analogy**: If words that appear in similar contexts should have similar embeddings (Word2Vec), then **nodes that appear in similar "graph contexts" should have similar embeddings**.

### Key Innovation

node2vec introduces **flexible, biased random walks** that can smoothly interpolate between:
- **Breadth-First Search (BFS)**: Local neighborhood exploration
- **Depth-First Search (DFS)**: Exploring farther nodes

This flexibility allows node2vec to capture both:
1. **Homophily**: Nodes in the same community (nearby nodes)
2. **Structural equivalence**: Nodes with similar roles (e.g., both are hubs)

---

## Background: From Word2Vec to node2vec

### Word2Vec: The Inspiration

**Skip-gram model** learns word embeddings by:
1. Taking a sentence: "the quick brown fox jumps"
2. Creating pairs: (quick, the), (quick, brown), (brown, quick), (brown, fox), ...
3. Learning embeddings where: P(context_word | center_word) is high

**Mathematical objective**:
```
maximize: Σ log P(w_context | w_center)

where: P(w_c | w_center) = exp(z_c^T · z_center) / Σ_w exp(z_w^T · z_center)
```

### Adapting to Graphs

**The mapping**:
- Sentences → **Random walks** on the graph
- Words → **Nodes**  
- Context window → **Nodes appearing together in walks**

**Example**: Walk: 1 → 3 → 5 → 7 → 9
- Creates pairs: (1,3), (1,5), (3,1), (3,5), (3,7), (5,3), (5,7), (5,9), ...

---

## Random Walks on Graphs

### What is a Random Walk?

A **random walk** is a path through the graph where at each step, we randomly choose one of the current node's neighbors.

**Example Graph**:
```
    1 --- 2
    |     |
    3 --- 4 --- 5
          |
          6
```

**Random walk starting from node 1**:
```
Step 0: At node 1
Step 1: Randomly pick neighbor → go to 2 or 3
Step 2: From there, randomly pick a neighbor
...
Continue for k steps
```

**Example walk**: 1 → 3 → 4 → 5 → 4 → 6

### Uniform Random Walk (DeepWalk)

In **DeepWalk**, the probability of moving to a neighbor is uniform:

```
P(next = v | current = u) = 1/degree(u)  if (u,v) is an edge
                            0             otherwise
```

**Problem**: Uniform walks may not capture all graph properties equally well.

---

## Biased Random Walks

### The Problem with Uniform Walks

Consider these two graph properties:

1. **Homophily** (Community): Nodes 1, 2, 3 form a tight cluster
2. **Structural Role**: Node 1 and Node 10 are both "hub" nodes

**Uniform random walks** might not capture both equally well!

### The node2vec Solution: Bias the Walk!

node2vec introduces **two parameters** to control the walk:

1. **p** (return parameter): Controls likelihood of revisiting a node
2. **q** (in-out parameter): Controls exploration vs. exploitation

### The Biased Random Walk Mechanism

**Setup**: Currently at node **v**, came from node **t**, deciding where to go next.

**Neighbors of v**:
- **t**: The previous node (where we came from)
- **x₁**: Neighbor at distance 1 from t (same distance as v)
- **x₂**: Neighbor at distance 2 from t (farther than v)

**Transition probabilities** (unnormalized):

```
α(t, x) = 1/p    if x = t        (returning to previous node)
          1      if d(t,x) = 1    (staying at same distance)
          1/q    if d(t,x) = 2    (moving away)
```

where `d(t,x)` is the shortest path distance between t and x.

### Visual Example

```
        t (previous)
        |
        v (current)
       /|\
      / | \
     /  |  \
    x₁  x₂  x₃

where:
- x₁ is also neighbor of t (distance 1 from t)
- x₂ is not neighbor of t (distance 2 from t)  
- x₃ = t (distance 0, returning)
```

**Transition probabilities**:
```
P(x₃ | v, came from t) ∝ 1/p  (return to t)
P(x₁ | v, came from t) ∝ 1    (explore at same level)
P(x₂ | v, came from t) ∝ 1/q  (go farther)
```

### Understanding Parameters p and q

#### Parameter **p** (Return Parameter)

**Controls**: Likelihood of immediately returning to the previous node

```
p < 1: High probability of returning → more "backtracking"
p = 1: No bias
p > 1: Low probability of returning → more "forward moving"
```

**Use case**: 
- **Low p**: Keeps walk local, good for community detection
- **High p**: Encourages exploration, prevents getting stuck

#### Parameter **q** (In-Out Parameter)

**Controls**: Balance between BFS and DFS

```
q < 1: DFS-like behavior (explore far)
       → Good for capturing structural equivalence
       → Nodes with similar "roles" get similar embeddings
       
q = 1: No bias (similar to unbiased walk)

q > 1: BFS-like behavior (stay local)
       → Good for capturing homophily
       → Nodes in same community get similar embeddings
```

### Example Settings

**Setting 1: Community Detection** (capture homophily)
```
p = 1
q = 2
```
→ BFS-like: Explores local neighborhood thoroughly

**Setting 2: Structural Equivalence** (capture roles)
```
p = 1
q = 0.5
```
→ DFS-like: Explores farther, finds nodes with similar structural patterns

**Setting 3: Balanced**
```
p = 1
q = 1
```
→ Unbiased (like DeepWalk)

---

## The node2vec Algorithm

### High-Level Algorithm

```
Algorithm: node2vec

Input:
  - Graph G = (V, E)
  - Dimensions d
  - Walk length l
  - Walks per node r
  - Context window size k
  - Parameters p, q

Output:
  - Embeddings Z ∈ R^(|V| × d)

Steps:
1. Precompute transition probabilities for biased walks
2. Generate walks:
   For each node u ∈ V:
     For i = 1 to r:
       Generate biased random walk of length l starting from u
3. Learn embeddings using Skip-gram:
   Optimize embeddings to predict context nodes from center nodes
4. Return learned embeddings Z
```

### Step-by-Step Detailed Algorithm

#### Step 1: Precompute Transition Probabilities

For computational efficiency, we precompute all transition probabilities:

```python
def precompute_transition_probs(G, p, q):
    """
    For each edge (t, v) in the graph, compute 
    probabilities for the next step from v.
    """
    transition_probs = {}
    
    for edge in G.edges():
        t, v = edge  # came from t, currently at v
        probs = {}
        
        for neighbor in G.neighbors(v):
            if neighbor == t:
                # Returning to previous node
                probs[neighbor] = 1/p
            elif G.has_edge(neighbor, t):
                # Distance 1 from t
                probs[neighbor] = 1
            else:
                # Distance 2 from t
                probs[neighbor] = 1/q
        
        # Normalize
        norm = sum(probs.values())
        for neighbor in probs:
            probs[neighbor] /= norm
            
        transition_probs[(t, v)] = probs
    
    return transition_probs
```

#### Step 2: Generate Biased Random Walks

```python
def generate_walk(G, start_node, walk_length, transition_probs):
    """
    Generate a single biased random walk.
    """
    walk = [start_node]
    
    for i in range(walk_length - 1):
        current = walk[-1]
        neighbors = list(G.neighbors(current))
        
        if len(neighbors) == 0:
            break
            
        if len(walk) == 1:
            # First step: uniform random choice
            next_node = random.choice(neighbors)
        else:
            # Use biased probabilities
            previous = walk[-2]
            probs = transition_probs[(previous, current)]
            next_node = random.choices(neighbors, 
                                      weights=[probs[n] for n in neighbors])[0]
        
        walk.append(next_node)
    
    return walk
```

#### Step 3: Train Skip-gram Model

```python
def train_skipgram(walks, d, window_size, num_epochs):
    """
    Learn embeddings using skip-gram objective.
    Uses negative sampling for efficiency.
    """
    # Initialize embeddings
    vocab_size = max(max(walk) for walk in walks) + 1
    Z = np.random.randn(vocab_size, d) * 0.01
    
    for epoch in range(num_epochs):
        for walk in walks:
            # Generate training pairs
            for i, center_node in enumerate(walk):
                # Context window
                start = max(0, i - window_size)
                end = min(len(walk), i + window_size + 1)
                
                for j in range(start, end):
                    if i != j:
                        context_node = walk[j]
                        
                        # Positive pair: (center_node, context_node)
                        # Negative samples: (center_node, random_node)
                        
                        # Update embeddings using gradient descent
                        # [Implementation details omitted for brevity]
    
    return Z
```

---

## Mathematical Formulation

### Objective Function

node2vec maximizes the log-probability of observing network neighborhoods:

```
maximize: Σ_{u ∈ V} log P(N_S(u) | z_u)
```

Where:
- `N_S(u)` is the network neighborhood of node u generated by strategy S
- `z_u` is the embedding of node u

### Skip-gram Formulation

Using the skip-gram assumption (independence):

```
P(N_S(u) | z_u) = Π_{v ∈ N_S(u)} P(v | z_u)
```

### Prediction Probability

```
P(v | z_u) = exp(z_u^T · z_v) / Σ_{w ∈ V} exp(z_u^T · z_w)
```

This is expensive to compute (O(|V|) for each pair)!

### Negative Sampling (Optimization Trick)

To make training efficient, node2vec uses **negative sampling**:

```
log P(v | z_u) ≈ log σ(z_u^T · z_v) + Σ_{i=1}^k E_{v_i ~ P_n} [log σ(-z_u^T · z_{v_i})]
```

Where:
- σ is the sigmoid function
- P_n is the negative sampling distribution (usually uniform)
- k is the number of negative samples (typically 5-20)

**Final Loss Function**:

```
L = Σ_{(u,v)∈D} [-log σ(z_u^T · z_v) - γ Σ_{v_n ~ P_n} log σ(-z_u^T · z_{v_n})]
```

Where D is the set of (center, context) pairs from random walks.

---

## Complete Worked Example

Let's work through node2vec on a small graph!

### The Graph

```
    1 --- 2 --- 3
    |     |     |
    4 --- 5 --- 6
```

**Adjacency**:
- 1: [2, 4]
- 2: [1, 3, 5]
- 3: [2, 6]
- 4: [1, 5]
- 5: [2, 4, 6]
- 6: [3, 5]

### Parameters

```
p = 1      (no return bias)
q = 0.5    (DFS-like: prefer exploring far)
l = 5      (walk length)
r = 3      (walks per node)
d = 3      (embedding dimension)
k = 2      (context window)
```

### Step 1: Compute Transition Probabilities

**Example**: Currently at node 5, came from node 2

Neighbors of 5: {2, 4, 6}

**Distances from node 2** (where we came from):
- Node 2: distance 0 (it's where we came from) → probability ∝ 1/p = 1
- Node 4: distance 2 (not neighbor of 2) → probability ∝ 1/q = 2
- Node 6: distance 2 (not neighbor of 2) → probability ∝ 1/q = 2

**Unnormalized weights**: {2: 1, 4: 2, 6: 2}

**Normalized probabilities**:
```
P(2 | at 5, from 2) = 1/5 = 0.20
P(4 | at 5, from 2) = 2/5 = 0.40
P(6 | at 5, from 2) = 2/5 = 0.40
```

**Interpretation**: With q = 0.5 (DFS-like), we're more likely to explore farther (nodes 4, 6) than return to 2.

### Step 2: Generate Random Walks

**Walk 1 from node 1**:
```
Start: 1
Step 1: 1 → 2 (random choice from neighbors)
Step 2: 2 → 5 (using transition probs based on coming from 1)
Step 3: 5 → 6 (using transition probs based on coming from 2)
Step 4: 6 → 3 (using transition probs based on coming from 5)

Walk: [1, 2, 5, 6, 3]
```

**Walk 2 from node 1**:
```
Walk: [1, 4, 5, 2, 3]
```

**Walk 3 from node 1**:
```
Walk: [1, 2, 3, 6, 5]
```

**Similar walks from all other nodes** (2, 3, 4, 5, 6)...

Total walks: 6 nodes × 3 walks/node = **18 walks**

### Step 3: Extract Training Pairs

From walk `[1, 2, 5, 6, 3]` with window size k=2:

**Center = 1** (position 0):
- Context: {2, 5}
- Pairs: (1,2), (1,5)

**Center = 2** (position 1):
- Context: {1, 5, 6}
- Pairs: (2,1), (2,5), (2,6)

**Center = 5** (position 2):
- Context: {1, 2, 6, 3}
- Pairs: (5,1), (5,2), (5,6), (5,3)

**Center = 6** (position 3):
- Context: {2, 5, 3}
- Pairs: (6,2), (6,5), (6,3)

**Center = 3** (position 4):
- Context: {5, 6}
- Pairs: (3,5), (3,6)

### Step 4: Train Skip-gram

Initialize embeddings randomly:
```
z_1 = [0.12, -0.34, 0.56]
z_2 = [-0.23, 0.45, 0.12]
z_3 = [0.34, 0.22, -0.11]
z_4 = [-0.45, 0.13, 0.28]
z_5 = [0.21, -0.12, 0.33]
z_6 = [0.33, 0.24, -0.15]
```

**Training pair example**: (5, 6) (node 5 predicts node 6)

**Forward pass**:
```
score = z_5^T · z_6 
      = 0.21×0.33 + (-0.12)×0.24 + 0.33×(-0.15)
      = 0.0693 + (-0.0288) + (-0.0495)
      = -0.009

P(6 | 5) = exp(-0.009) / Σ_w exp(z_5^T · z_w)
```

**Negative sampling**: Sample negative nodes (say nodes 1, 4)

**Loss**:
```
loss = -log σ(z_5^T · z_6) - log σ(-z_5^T · z_1) - log σ(-z_5^T · z_4)
```

**Gradient descent**: Update embeddings to minimize loss

**After training** (many iterations over all pairs):

```
Learned embeddings (example):
z_1 = [0.67, 0.45, -0.12]
z_2 = [0.71, 0.52, -0.08]  ← similar to z_1 (neighbors)
z_3 = [0.69, 0.48, -0.15]  ← similar to z_2, z_6
z_4 = [0.64, 0.43, -0.18]  ← similar to z_1 (neighbors)
z_5 = [0.70, 0.49, -0.10]  ← central node, similar to all
z_6 = [0.68, 0.47, -0.14]  ← similar to z_3, z_5
```

**Observation**: Nodes in the same cluster have similar embeddings!

---

## Implementation Guide

### Pseudocode

```python
import numpy as np
import random
from collections import defaultdict

class Node2Vec:
    def __init__(self, G, dimensions=128, walk_length=80, 
                 num_walks=10, p=1, q=1, window_size=10):
        self.G = G
        self.d = dimensions
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window_size = window_size
        
    def fit(self):
        # Step 1: Precompute probabilities
        self.transition_probs = self.precompute_transition_probs()
        
        # Step 2: Generate walks
        walks = self.generate_walks()
        
        # Step 3: Train skip-gram
        self.embeddings = self.train_skipgram(walks)
        
        return self.embeddings
    
    def precompute_transition_probs(self):
        # Implementation as shown earlier
        pass
    
    def generate_walks(self):
        walks = []
        nodes = list(self.G.nodes())
        
        for _ in range(self.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self.generate_walk(node)
                walks.append(walk)
        
        return walks
    
    def generate_walk(self, start_node):
        # Implementation as shown earlier
        pass
    
    def train_skipgram(self, walks):
        # Can use existing libraries like gensim.Word2Vec
        from gensim.models import Word2Vec
        
        # Convert walks to strings for gensim
        walks_str = [[str(node) for node in walk] for walk in walks]
        
        model = Word2Vec(walks_str, 
                        vector_size=self.d,
                        window=self.window_size,
                        min_count=0,
                        sg=1,  # skip-gram
                        workers=4,
                        epochs=5)
        
        # Extract embeddings
        embeddings = {}
        for node in self.G.nodes():
            embeddings[node] = model.wv[str(node)]
        
        return embeddings
```

### Using Existing Implementation

The easiest way is to use existing libraries:

```python
# Install: pip install node2vec

from node2vec import Node2Vec
import networkx as nx

# Create graph
G = nx.karate_club_graph()

# Generate walks and learn embeddings
node2vec = Node2Vec(G, 
                    dimensions=64,     # embedding dimension
                    walk_length=30,    # length of each walk
                    num_walks=200,     # walks per node
                    p=1,               # return parameter
                    q=1,               # in-out parameter
                    workers=4)

# Train
model = node2vec.fit(window=10,       # context window
                     min_count=1, 
                     batch_words=4)

# Get embeddings
embeddings = model.wv
```

---

## Comparison with Other Methods

### node2vec vs DeepWalk

| Aspect | DeepWalk | node2vec |
|--------|----------|----------|
| Random walks | Uniform | Biased (p, q parameters) |
| Flexibility | Less flexible | More flexible |
| Captures | Mixed properties | Can target specific properties |
| Speed | Faster | Slightly slower (precomputation) |

### node2vec vs Graph Factorization

| Aspect | Graph Factorization | node2vec |
|--------|---------------------|----------|
| Approach | Matrix factorization | Random walks |
| Similarity | Deterministic | Stochastic |
| Scalability | O(\|V\|²) | O(\|E\|) |
| Flexibility | Limited | High (via p, q) |

### When to Use node2vec?

**Use node2vec when**:
- You need flexible embeddings
- Graph is large (millions of nodes)
- You want to tune homophily vs structural equivalence
- You need state-of-the-art performance

**Consider alternatives when**:
- Graph is very small (< 100 nodes) → matrix methods may be simpler
- You need interpretable embeddings → spectral methods
- You have rich node features → GNNs

---

## Applications and Use Cases

### 1. Link Prediction

**Task**: Predict missing edges

**Setup**:
- Train node2vec on partial graph
- For candidate edge (u,v), compute similarity: z_u^T · z_v
- High similarity → likely edge

**Hyperparameters**: p=1, q=1 (balanced)

### 2. Node Classification

**Task**: Classify nodes into categories

**Setup**:
- Learn embeddings with node2vec
- Use embeddings as features in classifier (SVM, Logistic Regression)

**Hyperparameters**: p=1, q=0.5 (DFS-like, captures roles)

### 3. Community Detection

**Task**: Find tightly connected groups

**Setup**:
- Learn embeddings with BFS-like walks
- Cluster embeddings (K-means)

**Hyperparameters**: p=1, q=2 (BFS-like, captures homophily)

### 4. Recommendation Systems

**Task**: Recommend items to users

**Setup**:
- Build user-item graph
- Learn embeddings
- Recommend items with high similarity to user's embedding

### Real-World Examples

**Bioinformatics**: Protein-protein interaction networks
- p=1, q=0.5 (find proteins with similar functions)

**Social Networks**: Friend recommendation
- p=1, q=2 (find people in same community)

**Knowledge Graphs**: Entity linking
- p=0.5, q=2 (explore local neighborhood)

---

## Advantages and Limitations

### Advantages

1. **Scalable**: Linear in number of edges O(|E|)
2. **Flexible**: Parameters p, q control embedding properties
3. **General**: Works on any graph (directed, weighted, etc.)
4. **Proven**: Strong empirical performance
5. **Unsupervised**: No labels needed

### Limitations

1. **Transductive**: Can't embed new nodes without retraining
2. **No features**: Ignores node attributes
3. **Hyperparameters**: Need to tune p, q
4. **Memory**: Stores many random walks
5. **Static**: Doesn't handle dynamic graphs well

### Solutions to Limitations

**For new nodes**: 
- Retrain (expensive)
- Or use inductive methods (GraphSAGE)

**For node features**:
- Concatenate node2vec embeddings with features
- Or use GNNs

**For dynamic graphs**:
- Incremental training
- Or use temporal GNN methods

---

## Tips and Best Practices

### Hyperparameter Tuning

**Walk length** (l):
- Short walks (10-40): Faster, local structure
- Long walks (80-100): Slower, global structure
- **Typical**: 80

**Walks per node** (r):
- Few walks (10): Faster, less stable
- Many walks (100+): Slower, more stable
- **Typical**: 10-20

**Dimensions** (d):
- Low (32-64): Faster, less expressive
- High (128-256): Slower, more expressive
- **Typical**: 128

**Parameters p and q**:
- Start with p=1, q=1 (unbiased)
- For communities: increase q (q=2)
- For roles: decrease q (q=0.5)
- Grid search over {0.5, 1, 2, 4}

### Computational Tips

1. **Precompute transition probabilities** once
2. **Parallelize** walk generation
3. **Use existing Word2Vec implementations** (gensim)
4. **Sample fewer walks** for initial experiments
5. **Cache** random walks for multiple experiments

---

## Summary

### The node2vec Algorithm in Three Steps

1. **Generate biased random walks** using parameters p and q
2. **Extract (center, context) pairs** from walks
3. **Learn embeddings** using skip-gram with negative sampling

### Key Takeaways

- node2vec = **Flexible random walks** + **Skip-gram**
- Parameter **p** controls return probability
- Parameter **q** controls BFS vs DFS behavior
- Captures both **homophily** and **structural equivalence**
- **Scalable** to millions of nodes
- **State-of-the-art** performance on many tasks

### When You've Mastered node2vec

You understand:
- ✓ How random walks capture graph structure
- ✓ How p and q parameters control exploration
- ✓ How skip-gram learns from walks
- ✓ How to tune hyperparameters for your task
- ✓ When to use node2vec vs alternatives

---

## References

**Original Paper**:
- Grover, A., & Leskovec, J. (2016). node2vec: Scalable Feature Learning for Networks. KDD.

**Related Methods**:
- Perozzi, B., et al. (2014). DeepWalk: Online Learning of Social Representations. KDD.
- Mikolov, T., et al. (2013). Efficient Estimation of Word Representations. ICLR.

**Implementations**:
- Official: https://github.com/aditya-grover/node2vec
- Python library: https://github.com/eliorc/node2vec

**GRL Book**: Hamilton, W. L. (2020). Graph Representation Learning. Morgan & Claypool.

---

## Exercises

### Exercise 1: Parameter Effects
Try different (p, q) combinations on the Karate Club graph. How do embeddings change?

### Exercise 2: Implementation
Implement the biased random walk from scratch.

### Exercise 3: Application
Use node2vec for link prediction on a dataset of your choice.

### Exercise 4: Comparison
Compare node2vec with DeepWalk and graph factorization on the same task.

---

**🎓 Congratulations!** You now understand the node2vec algorithm and how to apply it to real-world graph problems!