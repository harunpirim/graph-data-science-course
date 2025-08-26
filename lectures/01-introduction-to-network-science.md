# Introduction to Network Science

## Overview

Network Science is the study of complex systems represented as networks (graphs) where entities are nodes and relationships are edges. This field combines elements from mathematics, computer science, physics, and social sciences to understand how interconnected systems behave.
Thanks.......

## Learning Objectives

By the end of this lecture, you will be able to:
- Understand basic network science concepts and terminology
- Identify different types of networks and their properties
- Analyze fundamental network metrics
- Recognize real-world applications of network science
- Follow a systematic approach to network analysis

## 1. What is Network Science?

### Definition
Network Science is the study of complex systems as networks, where:
- **Nodes** (vertices) represent entities (people, computers, proteins, etc.)
- **Edges** (links) represent relationships or interactions between entities

### Key Concepts
- **Graph Theory**: Mathematical foundation for network analysis
- **Complex Systems**: Systems with many interacting components
- **Emergent Properties**: Behaviors that arise from interactions, not individual components

### Historical Context
- **1736**: Euler's solution to the Königsberg bridge problem
- **1950s-60s**: Development of graph theory in mathematics
- **1990s**: Emergence of "small-world" and "scale-free" networks
- **2000s**: Network Science as a distinct interdisciplinary field

## 2. Types of Networks

### Undirected Networks
- Edges have no direction (e.g., friendship networks)
- Symmetric relationships
- Example: Facebook friendships

### Directed Networks
- Edges have direction (e.g., following on social media)
- Asymmetric relationships
- Example: Twitter followers

### Weighted Networks
- Edges have weights (e.g., communication frequency)
- Quantitative relationships
- Example: Email communication volume

### Bipartite Networks
- Two types of nodes
- Edges only between different types
- Example: Users and movies in recommendation systems

### Temporal Networks
- Networks that change over time
- Dynamic relationships
- Example: Communication networks over time

## 3. Basic Network Properties

### Size Metrics
- **Number of nodes (N)**: Total entities in the network
- **Number of edges (E)**: Total relationships
- **Density**: Ratio of actual edges to possible edges
  - Formula: `density = 2E / (N × (N-1))` for undirected networks

### Connectivity
- **Connected component**: Subset of nodes where any two nodes are connected by a path
- **Giant component**: Largest connected component
- **Isolated nodes**: Nodes with no connections

### Degree Distribution
- **Degree**: Number of connections a node has
- **Average degree**: Mean number of connections per node
- **Degree distribution**: Distribution of degrees across all nodes

### Path Properties
- **Path**: Sequence of edges connecting two nodes
- **Shortest path**: Minimum number of edges between two nodes
- **Diameter**: Maximum shortest path length in the network
- **Average path length**: Mean shortest path length between all pairs

## 4. Network Models

### Random Networks (Erdős-Rényi)
- Each pair of nodes connected with probability p
- Poisson degree distribution
- Low clustering coefficient

### Small-World Networks (Watts-Strogatz)
- High clustering like regular networks
- Short path lengths like random networks
- Created by rewiring regular networks

### Scale-Free Networks (Barabási-Albert)
- Power-law degree distribution
- Preferential attachment mechanism
- Few highly connected hubs, many poorly connected nodes

## 5. Real-World Network Examples

### Social Networks
- **Facebook**: Friendship networks
- **Twitter**: Follower networks
- **LinkedIn**: Professional networks
- **Academic collaborations**: Co-authorship networks

### Biological Networks
- **Protein-protein interactions**: How proteins interact
- **Gene regulatory networks**: How genes control each other
- **Metabolic networks**: Biochemical reactions
- **Neural networks**: Brain connectivity

### Technological Networks
- **Internet**: Router connections
- **Power grids**: Electrical transmission
- **Transportation**: Roads, flights, public transit
- **Communication**: Phone calls, emails

### Information Networks
- **World Wide Web**: Web pages and links
- **Citation networks**: Academic paper references
- **Wikipedia**: Articles and internal links
- **Software dependencies**: Package relationships

## 6. Network Analysis Workflow

### Step 1: Data Collection
- Identify entities (nodes) and relationships (edges)
- Collect relevant attributes
- Handle missing data and errors
- Validate data quality

### Step 2: Network Construction
- Choose appropriate network type
- Define node and edge attributes
- Handle multiple relationships
- Validate network structure

### Step 3: Basic Analysis
- Calculate network properties
- Identify key nodes and edges
- Analyze network structure
- Detect anomalies

### Step 4: Advanced Analysis
- Community detection
- Centrality analysis
- Network dynamics
- Predictive modeling

### Step 5: Visualization and Interpretation
- Create effective visualizations
- Interpret results in context
- Draw conclusions
- Communicate findings

## 7. Key Network Metrics

### Centrality Measures
- **Degree centrality**: Number of connections
- **Closeness centrality**: Average distance to other nodes
- **Betweenness centrality**: Role as bridge between others
- **Eigenvector centrality**: Importance based on neighbors' importance

### Clustering
- **Clustering coefficient**: How connected neighbors are
- **Local clustering**: Clustering around specific nodes
- **Global clustering**: Overall network clustering

### Assortativity
- **Degree assortativity**: Do high-degree nodes connect to other high-degree nodes?
- **Attribute assortativity**: Do similar nodes connect to each other?

## 8. Applications and Impact

### Social Sciences
- Understanding social influence and diffusion
- Analyzing organizational structures
- Studying political networks

### Biology and Medicine
- Drug discovery and protein interactions
- Disease spread modeling
- Brain connectivity analysis

### Technology and Engineering
- Internet and communication networks
- Power grid resilience
- Transportation optimization

### Economics and Finance
- Supply chain analysis
- Financial contagion modeling
- Market structure analysis

## 9. Tools and Software

### Python Libraries
- **NetworkX**: Comprehensive network analysis
- **igraph**: Fast network algorithms
- **PyTorch Geometric**: Graph neural networks
- **Plotly**: Interactive visualizations

### Specialized Software
- **Gephi**: Interactive network visualization
- **Cytoscape**: Biological network analysis
- **Pajek**: Large network analysis
- **UCINET**: Social network analysis

### Visualization Tools
- **D3.js**: Web-based visualizations
- **Gephi**: Desktop visualization
- **Matplotlib/Seaborn**: Python plotting
- **Plotly**: Interactive plots

## 10. Challenges and Future Directions

### Data Challenges
- **Scale**: Handling massive networks
- **Quality**: Dealing with noisy data
- **Privacy**: Protecting sensitive information
- **Dynamic**: Analyzing evolving networks

### Methodological Challenges
- **Causality**: Distinguishing correlation from causation
- **Temporal dynamics**: Understanding network evolution
- **Multilayer networks**: Analyzing multiple relationship types
- **Non-stationarity**: Handling changing network properties

### Emerging Areas
- **Graph Neural Networks**: Machine learning on networks
- **Temporal networks**: Time-evolving networks
- **Multilayer networks**: Multiple relationship types
- **Network medicine**: Medical applications

## Summary

Network Science provides powerful tools for understanding complex systems across many domains. By representing systems as networks, we can:

1. **Identify patterns** in complex interactions
2. **Predict behavior** based on network structure
3. **Optimize systems** by understanding connectivity
4. **Understand emergence** of collective behavior

The field continues to grow with new applications in machine learning, medicine, and technology. Understanding network science fundamentals provides a foundation for advanced analysis and research.

## Key Takeaways

- Networks represent complex systems as nodes and edges
- Different network types capture different relationship types
- Network properties reveal system behavior and structure
- Network analysis follows a systematic workflow
- Applications span social, biological, and technological domains
- Tools and methods continue to evolve rapidly

## Next Steps

- Learn about network connectivity and paths
- Explore centrality measures in detail
- Study community detection methods
- Understand network dynamics and evolution
- Practice with real-world datasets
- Explore advanced topics like graph machine learning


# Graph Theory: Understanding Nodes and Arcs

## Introduction to Graph Theory

Graph theory is a branch of mathematics that studies the relationships between objects. Think of it as a way to map connections - like how cities connect through roads, how people know each other in social networks, or how web pages link to one another.

A **graph** consists of two fundamental components:
- **Nodes** (also called vertices): The objects or entities we're studying
- **Arcs** (also called edges): The connections or relationships between nodes

## Basic Graph Components

### Nodes (Vertices)
Nodes represent the entities in your system. In different contexts, they might represent:
- Cities in a transportation network
- People in a social network
- Web pages on the internet
- Computers in a network

We typically represent nodes as circles or dots and label them with letters or names.

### Arcs (Edges)
Arcs represent the relationships or connections between nodes. They can be:
- **Undirected**: The connection works both ways (like friendship)
- **Directed**: The connection has a specific direction (like following someone on social media)

## Simple Graph Examples

### Example 1: A Basic Undirected Graph
```
    A ——— B
    |     |
    |     |
    D ——— C
```

**Nodes:** A, B, C, D
**Arcs:** A-B, B-C, C-D, D-A

This creates a square where each vertex connects to its neighbors. Notice that in an undirected graph, we can travel from any connected node to another in either direction.

### Example 2: A Directed Graph (Digraph)
```
    A ——→ B
    ↑     ↓
    |     |
    D ←—— C
```

**Nodes:** A, B, C, D
**Directed Arcs:** A→B, B→C, C→D, D→A

Here, the arrows show direction. You can go from A to B, but not necessarily from B to A without following the complete cycle.

## More Complex Graph Structures

### Example 3: A Social Network
```
      Alice
     /  |  \
    /   |   \
   Bob  |    Charlie
    \   |   /   |
     \  |  /    |
      David ——— Eve
```

**Nodes:** Alice, Bob, Charlie, David, Eve
**Relationships:**
- Alice knows: Bob, Charlie, David
- Bob knows: Alice, David
- Charlie knows: Alice, David, Eve
- David knows: Alice, Bob, Charlie, Eve
- Eve knows: Charlie, David

This represents a friendship network where Alice is highly connected (a hub), while Eve is more peripheral.

### Example 4: A Transportation Network
```
    Airport A ←——→ Airport B
        ↓              ↓
        ↓              ↓
    Airport C ←——→ Airport D
        ↓              ↑
        ↓              |
    Airport E ——————→ Airport F
```

**Flight Routes:**
- A ↔ B (bidirectional flights)
- A → C (one-way route)
- B → D (one-way route)
- C ↔ D (bidirectional flights)
- C → E (one-way route)
- E → F (one-way route)
- F → D (one-way route)

## Graph Properties and Terminology

### Degree of a Node
The **degree** of a node is the number of arcs connected to it.

In our social network example:
- Alice has degree 3 (connected to Bob, Charlie, David)
- David has degree 4 (connected to Alice, Bob, Charlie, Eve)
- Eve has degree 2 (connected to Charlie, David)

### Paths and Cycles
A **path** is a sequence of nodes connected by arcs. A **cycle** is a path that starts and ends at the same node.

Path example: Alice → David → Eve
Cycle example: A → B → C → D → A

### Connected vs. Disconnected Graphs
```
Connected Graph:       Disconnected Graph:
    A ——— B               A ——— B    E ——— F
    |     |               |     |       
    |     |               |     |       
    D ——— C               D ——— C    
```

## Special Types of Graphs

### Tree Structure
A tree is a connected graph with no cycles - like a family tree or file system.

```
        Root
       /  |  \
      /   |   \
     A    B    C
    / \       / \
   D   E     F   G
```

### Complete Graph
A complete graph connects every node to every other node.

```
Complete Graph with 4 nodes:
    A ——— B
    |\   /|
    | \ / |
    |  X  |
    | / \ |
    |/   \|
    D ——— C
```

### Weighted Graphs
Sometimes arcs have values (weights) representing cost, distance, or strength of connection.

```
    A ——5—— B
    |       |
    3       7
    |       |
    D ——2—— C
```

The numbers represent distances, costs, or other measurable relationships.

## Real-World Applications

Understanding nodes and arcs helps us model and solve problems in many fields:

**Computer Science:** Network routing, data structures, algorithms
**Social Sciences:** Analyzing relationships, influence patterns, community detection
**Transportation:** Route optimization, traffic flow analysis
**Biology:** Modeling neural networks, food webs, protein interactions
**Business:** Supply chains, organizational structures, customer relationships

## Key Insights for Learning

When you encounter any system with relationships, try to identify:
1. What are the entities? (These become your nodes)
2. How do they connect? (These become your arcs)
3. Do the connections have direction? (Directed vs. undirected)
4. Do the connections have different strengths or costs? (Weighted vs. unweighted)

Graph theory provides a powerful language for describing and analyzing these relationships, making complex systems easier to understand and optimize.

## Practice Exercise

Try modeling your own social network or the websites you visit regularly as a graph. Identify the nodes and arcs, and consider what insights you might gain from analyzing the structure. What patterns do you notice? Who or what serves as important connection points?

HEIF Image.jpg