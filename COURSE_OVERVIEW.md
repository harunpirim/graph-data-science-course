# Graph Data Science Course - Overview

## Course Structure

This repository contains a comprehensive course on Graph Data Science, organized into 7 main topics with both theoretical lectures and practical coding examples.

## 📚 Course Modules

### Module 1: Centrality in Networks
- **Lecture**: `lectures/01-centrality.md`
- **Notebook**: `notebooks/01-centrality.ipynb`
- **Topics Covered**:
  - Degree, Closeness, Betweenness Centrality
  - Eigenvector Centrality and PageRank
  - Real-world applications with Les Miserables network
  - Comparison of centrality measures

### Module 2: Network Connectivity
- **Lecture**: `lectures/02-connectivity.md`
- **Topics Covered**:
  - Walks, paths, and connected components
  - Node and edge connectivity
  - Giant component analysis
  - Assortativity and degree correlations
  - Network robustness measures

### Module 3: Graph Machine Learning
- **Lecture**: `lectures/03-graph-machine-learning.md`
- **Topics Covered**:
  - Node classification, link prediction, graph classification
  - DeepWalk and Node2Vec
  - Graph Neural Networks (GCN, GraphSAGE, GAT)
  - Training and evaluation of GNNs
  - Applications in various domains

### Module 4: Graph Visualization
- **Lecture**: `lectures/04-graph-visualization.md`
- **Topics Covered**:
  - Layout algorithms (force-directed, hierarchical, geometric)
  - NetworkX and Gephi visualization
  - Interactive visualizations with Plotly and D3.js
  - Large network visualization techniques
  - Best practices for effective visualization

### Module 5: Network Robustness
- **Lecture**: `lectures/05-network-robustness.md`
- **Topics Covered**:
  - Types of network failures (random, targeted, cascading)
  - Percolation theory
  - Attack strategies and defense mechanisms
  - Robustness measures and analysis methods
  - Applications in infrastructure and social networks

### Module 6: Multilayer Networks
- **Lecture**: `lectures/06-multilayer-networks.md`
- **Topics Covered**:
  - Types of multilayer networks (temporal, multiplex, interdependent)
  - Mathematical representation (supra-adjacency, tensors)
  - Centrality measures in multilayer networks
  - Community detection and random walk models
  - Applications in social, transportation, and biological networks

### Module 7: Signed Networks
- **Lecture**: `lectures/07-signed-networks.md`
- **Topics Covered**:
  - Balance theory and structural balance
  - Signed centrality measures
  - Community detection in signed networks
  - Link prediction with sign prediction
  - Applications in social and economic networks

## 🛠️ Practical Components

### Jupyter Notebooks
- `notebooks/01-centrality.ipynb` - Interactive centrality analysis
- `notebooks/pd2graphml.ipynb` - Pandas to GraphML conversion
- `notebooks/py_multinet.ipynb` - Multilayer network analysis

### Sample Data
- `data/sample_network.csv` - Example network dataset for practice

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Basic knowledge of Python programming
- Familiarity with linear algebra and statistics

### Installation
```bash
# Clone the repository
git clone https://github.com/harunpirim/graph-data-science-course.git
cd graph-data-science-course

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Learning Path
1. **Start with Module 1** (Centrality) for fundamental concepts
2. **Continue with Module 2** (Connectivity) for network structure
3. **Move to Module 4** (Visualization) for practical skills
4. **Study Module 5** (Robustness) for advanced analysis
6. **Explore Module 6** (Multilayer) for complex networks
7. **Finish with Module 7** (Signed Networks) for specialized applications
8. **Apply Module 3** (Graph ML) for machine learning applications

## 📖 Learning Resources

### Primary References
- Newman, M. E. J. (2010). Networks: An Introduction. Oxford University Press.
- Hamilton, W. L. (2020). Graph representation learning. Synthesis Lectures on AI and ML.
- Kivelä, M., et al. (2014). Multilayer networks. Journal of Complex Networks.

### Online Resources
- [NetworkX Documentation](https://networkx.org/)
- [PyTorch Geometric Tutorials](https://pytorch-geometric.readthedocs.io/)
- [Gephi Tutorials](https://gephi.org/tutorials/)

### Video Resources
- [Eigenvector Centrality](https://www.youtube.com/watch?v=Fr-KK8Ks5vg)
- [PageRank Algorithm](https://www.youtube.com/watch?v=-u02pxg4w8U)
- [Word2Vec Explanation](https://youtu.be/viZrOnJclY0)

## 🎯 Learning Objectives

By the end of this course, students will be able to:

1. **Understand** fundamental concepts in network analysis
2. **Apply** centrality measures to identify important nodes
3. **Analyze** network connectivity and robustness
4. **Visualize** networks effectively using various tools
5. **Implement** graph machine learning algorithms
6. **Model** complex systems using multilayer networks
7. **Analyze** signed networks and balance theory
8. **Apply** network analysis to real-world problems

## 💻 Technical Skills Developed

- **Python Programming**: NetworkX, PyTorch Geometric, Matplotlib
- **Data Analysis**: Network metrics, statistical analysis
- **Machine Learning**: Graph neural networks, link prediction
- **Visualization**: Interactive plots, network layouts
- **Research Methods**: Network modeling, hypothesis testing

## 🔬 Applications Covered

- **Social Networks**: Friend recommendations, influence analysis
- **Biological Networks**: Protein interactions, gene regulation
- **Transportation Networks**: Route optimization, accessibility
- **Economic Networks**: Trade relationships, financial contagion
- **Infrastructure Networks**: Power grids, communication systems

## 📊 Assessment and Projects

### Suggested Projects
1. **Social Network Analysis**: Analyze a real social network dataset
2. **Network Visualization**: Create an interactive visualization tool
3. **Link Prediction**: Implement and evaluate link prediction algorithms
4. **Community Detection**: Compare different community detection methods
5. **Robustness Analysis**: Study network resilience under various attacks

### Evaluation Criteria
- Understanding of theoretical concepts
- Implementation of algorithms
- Quality of analysis and interpretation
- Effective communication of results

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Submit bug reports
- Suggest new topics or examples
- Improve existing content
- Add new applications or case studies

## 📄 License

This course is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍🏫 Instructor

**Harun Pirim** - Professor and researcher in network science and data analytics.

---

*This course provides a comprehensive introduction to graph data science, combining theoretical foundations with practical applications. Each module builds upon previous knowledge, creating a structured learning experience for students and researchers interested in network analysis.* 