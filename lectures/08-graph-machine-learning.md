# Graph Machine Learning
1-6.png
Title: hypergraph
Width: 50%

1-5.png
Title: knowledge graph
Width: 50%

## Introduction

# Machine Learning as Optimization: A Simple Summary

Machine learning is fundamentally an **optimization problem** where we search for the best possible model to perform a specific task.

## The Core Process

**Goal**: Find a mathematical model that achieves optimal performance on a given task

**Key Components**:

1. **Performance Metric** (Loss Function/Cost Function)
   - Quantifies how well the model is performing
   - Lower loss = better performance

2. **Data-Driven Learning**
   - Algorithm receives data (often large amounts)
   - Uses this data to make iterative improvements

3. **The Learning Cycle** (Training):
   ```
   Make Decision/Prediction
          ↓
   Evaluate with Loss Function
          ↓
   Calculate Error
          ↓
   Update Model Parameters
          ↓
   (Repeat until performance is satisfactory)
   ```

**The Essence**: At each iteration, the model makes predictions, measures how wrong it is, and adjusts its internal parameters to do better next time. Through repeated cycles, the model progressively improves its performance.

This iterative process of **learning from mistakes** is what we call **training** - it's how machines get "smarter" at their tasks over time.

SL.jpg

+++

Learning.jpg

## Graph Learning Tasks

encode.jpg

[Graph Machine Learning levels](https://claude.ai/public/artifacts/5c8f26be-d383-4c7a-8114-8df2d4a86864)


## Node Representation Learning

nodeE.jpg

$$\text{ENC} : \mathcal{V} \rightarrow \mathbb{R}^{d},$$


decode.jpg

$$\text{DEC} : \mathbb{R}^{d} \times \mathbb{R}^{d} \rightarrow \mathbb{R}^{+}.$$


$$\text{DEC}(\text{ENC}(u), \text{ENC}(v)) = \text{DEC}(\mathbf{z}_u, \mathbf{z}_v) \approx \mathbf{S}[u, v].$$


To achieve the reconstruction this objective, the standard practice is to minimize an empirical reconstruction loss Lover a set of training node pairs D:

$$\mathcal{L} = \sum_{(u,v) \in \mathcal{D}} \ell\big(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v), \mathbf{S}[u, v]\big),$$


| **Method**         | **Decoder** | **Similarity measure** | **Loss function** |
|--------------------|-------------|------------------------|-------------------|
| Lap. Eigenmaps     | $\|\mathbf{z}_u - \mathbf{z}_v\|_2^2$ | general | $\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) \cdot \mathbf{S}[u,v]$ |
| Graph Fact.        | $\mathbf{z}_u^\top \mathbf{z}_v$ | $\mathbf{A}[u,v]$ | $\|\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) - \mathbf{S}[u,v]\|_2^2$ |
| GraRep             | $\mathbf{z}_u^\top \mathbf{z}_v$ | $\mathbf{A}[u,v], \ldots, \mathbf{A}^k[u,v]$ | $\|\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) - \mathbf{S}[u,v]\|_2^2$ |
| HOPE               | $\mathbf{z}_u^\top \mathbf{z}_v$ | general | $\|\text{DEC}(\mathbf{z}_u, \mathbf{z}_v) - \mathbf{S}[u,v]\|_2^2$ |
| DeepWalk           | $\dfrac{e^{\mathbf{z}_u^\top \mathbf{z}_v}}{\sum_{k \in \mathcal{V}} e^{\mathbf{z}_u^\top \mathbf{z}_k}}$ | $p_{\mathcal{G}}(v|u)$ | $-\mathbf{S}[u,v]\log(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v))$ |
| node2vec           | $\dfrac{e^{\mathbf{z}_u^\top \mathbf{z}_v}}{\sum_{k \in \mathcal{V}} e^{\mathbf{z}_u^\top \mathbf{z}_k}}$ | $p_{\mathcal{G}}(v|u)$ (biased) | $-\mathbf{S}[u,v]\log(\text{DEC}(\mathbf{z}_u, \mathbf{z}_v))$ |


Shallow embedding approaches face three major drawbacks:

	1.	No Parameter Sharing:
	Each node has its own independent embedding vector, leading to poor statistical efficiency and high computational cost. The number of parameters grows linearly with the number of nodes O(|V|), making the method unscalable for large graphs.
	
	2.	No Use of Node Features:
	These methods ignore rich node attribute information that could improve the quality and generalizability of learned representations.
	
	3.	Transductive Limitation:
	Shallow embeddings can only represent nodes seen during training. They cannot generalize to unseen nodes without retraining, preventing their use in inductive settings.

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

node2vec_tutorial.md

 [illustration here](https://claude.ai/public/artifacts/38bca534-40f2-4b6c-abca-ec283ded304a)


	The node embedding approaches we discussed used a shallow embedding approach to generate representations of nodes, where we simply optimized a unique embedding vector for each node.

	  The key idea is that we want togenerate representations of nodes that actually depend on the structure of the graph, as well as any feature information we might have.


$$f(\mathbf{PAP}^\top) = f(\mathbf{A})$$

$$f(\mathbf{PAP}^\top) = \mathbf{P} f(\mathbf{A})$$

The shallow encoders  are an example of permutation equivariant functions.) 

## Graph Neural Networks (GNNs)

 Regardless of the motivation, the defining feature of a GNN is that it uses a form of neural message passing in which vector messages are exchanged between nodes and updated using neural networks [Gilmer et al., 2017].
 
We will describe how we can take an input graph 
$\mathcal{G} = (\mathcal{V}, \mathcal{E})$, along with a set of node features 
$\mathbf{X} \in \mathbb{R}^{d \times |\mathcal{V}|}$, and use this information to generate node embeddings $\mathbf{z}_u, \ \forall u \in \mathcal{V}$. 

### Message Passing Framework

**Core Idea**: Update node representations by aggregating information from neighbors.

message.jpg

$$\mathbf{h}_u^{(k+1)} = \text{UPDATE}^{(k)} \Big( 
\mathbf{h}_u^{(k)}, 
\text{AGGREGATE}^{(k)} \big( \{ \mathbf{h}_v^{(k)}, \forall v \in \mathcal{N}(u) \} \big) 
\Big)$$

$$= \text{UPDATE}^{(k)} \Big( 
\mathbf{h}_u^{(k)}, 
\mathbf{m}_{\mathcal{N}(u)}^{(k)} 
\Big),$$

where $\text{UPDATE}$ and $\text{AGGREGATE}$ are arbitrary differentiable functions (i.e., neural networks), and $\mathbf{m}_{\mathcal{N}(u)}$ is the “message” that is 
aggregated from $u$’s graph neighborhood $\mathcal{N}(u)$. 
We use superscripts to distinguish the embeddings and functions at different iterations of message passing.\footnote{For example, $\mathbf{h}_u^{(k)}$ and 
$\text{UPDATE}^{(k)}$ refer to the embedding and update function at the $k$-th iteration.}




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
- Stamile, Claudio, et al. Graph Machine Learning : Take Graph Data to the Next Level by Applying Machine Learning Techniques and Algorithms, Packt Publishing,
- Keita Broadwater and Namid Stillman. Graph Neural Networks in Action. Manning Publications, 2025. ISBN: 9781617299056.
