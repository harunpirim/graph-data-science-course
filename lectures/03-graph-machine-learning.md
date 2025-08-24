# Graph Machine Learning

## Introduction

Graph Machine Learning (GML) is a rapidly growing field that combines graph theory with machine learning techniques to analyze and learn from graph-structured data. This field has applications in social networks, biological networks, recommendation systems, and many other domains.

## Graph Learning Tasks

### 1. Node Classification

**Definition**: Predicting the label of a node in a graph based on its features and the graph structure.

**Applications**:
- Social network analysis (predicting user interests)
- Biological networks (predicting protein functions)
- Citation networks (predicting paper categories)

**Approach**: Node features + information from edges + information from the graph structure.

### 2. Link Prediction

**Definition**: Predicting the existence of an edge between two nodes.

**Applications**:
- Social network friend suggestions
- Recommendation systems
- Biological network inference

**Methods**:
- Heuristic methods (Common Neighbors, Jaccard Coefficient)
- Embedding-based methods
- Graph Neural Networks

### 3. Graph Classification

**Definition**: Predicting the label of an entire graph.

**Applications**:
- Molecular property prediction
- Social network analysis
- Document classification

## Node Representation Learning

### DeepWalk Architecture

DeepWalk is a two-step approach for learning node representations:

1. **Random Walk Generation**: Generate random walks from the graph (similar to sequences of words)
2. **SkipGram Training**: Use SkipGram to learn node representations

### Word2Vec and SkipGram

**SkipGram Model**: Predicts context words given a target word.

**Mathematical Formulation**:
For a vocabulary of $N$ words: $w_1, w_2, ..., w_N$

The goal is to maximize the probability of context words given the target word:

$$ P(w_{i+j}|w_i) = \frac{exp(v_{w_{i+j}}^T v_{w_i})}{\sum_{w=1}^N exp(v_w^T v_{w_i})} $$

where $v_w$ is the vector representation of word $w$.

**Objective Function**:
$$\frac{1}{N}\sum_{n=1}^{N} \sum_{-c\leq j \leq c, j\ne 0} \log p(W_{n+j} | W_n)$$

where $c$ is the context window size.

### Node2Vec

**Extension of DeepWalk** with biased random walks:

- **p parameter**: Controls return to previous node
- **q parameter**: Controls exploration vs exploitation

**Random Walk Strategy**:
- **BFS-like**: Explores local neighborhood (homophily)
- **DFS-like**: Explores distant nodes (structural roles)

## Graph Neural Networks (GNNs)

### Message Passing Framework

**Core Idea**: Update node representations by aggregating information from neighbors.

**Mathematical Formulation**:
$$h_v^{(l+1)} = \sigma(W^{(l)} \cdot \text{AGGREGATE}^{(l)}(\{h_u^{(l)} : u \in \mathcal{N}(v)\}))$$

where:
- $h_v^{(l)}$ is the representation of node $v$ at layer $l$
- $\mathcal{N}(v)$ is the neighborhood of node $v$
- $\text{AGGREGATE}$ is an aggregation function
- $\sigma$ is an activation function

### Types of GNNs

#### 1. Graph Convolutional Networks (GCN)

**Aggregation Function**:
$h_v^{(l+1)} = \sigma(W^{(l)} \cdot \frac{1}{\sqrt{|\mathcal{N}(v)|}} \sum_{u \in \mathcal{N}(v)} h_u^{(l)})$

**Normalization**: Uses symmetric normalization for stability.

#### 2. GraphSAGE

**Aggregation Functions**:
- Mean
- Max
- LSTM
- Pooling

**Inductive Learning**: Can generalize to unseen nodes.

#### 3. Graph Attention Networks (GAT)

**Attention Mechanism**:
$\alpha_{ij} = \frac{exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_j]))}{\sum_{k \in \mathcal{N}_i} exp(\text{LeakyReLU}(a^T[Wh_i \| Wh_k]))}$

**Weighted Aggregation**:
$h_i^{(l+1)} = \sigma \ \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \, W^{(l)} h_j^{(l)} \right)$

## Graph Pooling

### Purpose
Reduce graph size while preserving important structural information.

### Methods

#### 1. Top-K Pooling
Select top-k nodes based on learnable scores.

#### 2. DiffPool
Learn hierarchical clustering of nodes.

#### 3. SAGPool
Use self-attention for node selection.

## Training GNNs

### Loss Functions

**Node Classification**:
$$\mathcal{L} = -\sum_{v \in \mathcal{V}_L} y_v \log(\hat{y}_v)$$

**Link Prediction**:
$\mathcal{L} = -\sum_{(u,v) \in \mathcal{E}} \Big[ y_{uv} \log(\hat{y}_{uv}) + (1 - y_{uv}) \log(1 - \hat{y}_{uv}) \Big]$

**Graph Classification**:
$$\mathcal{L} = -\sum_{G \in \mathcal{G}} y_G \log(\hat{y}_G)$$

### Regularization Techniques

1. **Dropout**: Randomly zero node features during training
2. **Weight Decay**: L2 regularization on model parameters
3. **Early Stopping**: Monitor validation performance
4. **Graph Augmentation**: Add/remove edges, mask features

## Applications

### 1. Social Networks
- **Node Classification**: User interest prediction, fake account detection
- **Link Prediction**: Friend suggestions, influence prediction
- **Community Detection**: Identifying cohesive groups

### 2. Biological Networks
- **Protein Function Prediction**: Predicting protein roles
- **Drug Discovery**: Molecular property prediction
- **Disease Prediction**: Identifying disease-related genes

### 3. Computer Vision
- **Scene Understanding**: Modeling object relationships
- **Point Cloud Analysis**: 3D shape classification
- **Image Segmentation**: Pixel-level predictions

### 4. Natural Language Processing
- **Document Classification**: Hierarchical text classification
- **Knowledge Graphs**: Entity and relation extraction
- **Semantic Parsing**: Converting text to structured representations

## Challenges and Limitations

### 1. Scalability
- **Computational Complexity**: O(|V|²) for dense graphs
- **Memory Requirements**: Large graphs may not fit in memory
- **Training Time**: Slow convergence for large networks

### 2. Over-smoothing
- **Problem**: Node representations become indistinguishable after many layers
- **Solutions**: Residual connections, normalization, attention mechanisms

### 3. Heterogeneous Graphs
- **Multiple Node Types**: Different types of entities
- **Multiple Edge Types**: Different types of relationships
- **Solutions**: Heterogeneous GNNs, meta-paths

### 4. Dynamic Graphs
- **Temporal Evolution**: Graphs change over time
- **Solutions**: Temporal GNNs, recurrent architectures

## Evaluation Metrics

### Node Classification
- **Accuracy**: Overall correct predictions
- **F1-Score**: Balance between precision and recall
- **AUC-ROC**: Area under the receiver operating characteristic curve

### Link Prediction
- **AUC**: Area under the curve
- **Precision@k**: Precision at top-k predictions
- **MRR**: Mean Reciprocal Rank

### Graph Classification
- **Accuracy**: Overall correct predictions
- **Cross-validation**: K-fold cross-validation
- **Statistical Significance**: Paired t-tests, Wilcoxon tests

## Future Directions

### 1. Theoretical Understanding
- **Expressiveness**: What functions can GNNs represent?
- **Convergence**: When and how fast do GNNs converge?
- **Generalization**: How well do GNNs generalize?

### 2. Scalability
- **Sampling Methods**: Efficient neighbor sampling
- **Distributed Training**: Multi-GPU, multi-machine training
- **Approximation**: Fast approximate algorithms

### 3. Interpretability
- **Attention Visualization**: Understanding what GNNs attend to
- **Feature Attribution**: Which features contribute to predictions?
- **Graph Explanations**: Explaining graph-level predictions

## References

- Hamilton, W. L. (2020). Graph representation learning. Synthesis Lectures on Artificial Intelligence and Machine Learning, 14(3), 1-159.
- Kipf, T. N., & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. ICLR.
- Veličković, P., et al. (2018). Graph attention networks. ICLR.
- Zhou, J., et al. (2020). Graph neural networks: A review of methods and applications. AI Open, 1, 57-81. 
