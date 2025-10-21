$$\begin{align*}
\text{Step 1:} \quad 
& \mathbf{h}^{(1)} = \mathbf{A}[1] \\[4pt]
\text{Step 2:} \quad 
& \mathbf{h}^{(2)} = \mathbf{A}[1] \oplus \mathbf{A}[2] \\[4pt]
\text{Step 3:} \quad 
& \mathbf{h}^{(3)} = \mathbf{A}[1] \oplus \mathbf{A}[2] \oplus \mathbf{A}[3] \\[4pt]
\text{Step 4:} \quad 
& \mathbf{h}^{(4)} = \mathbf{A}[1] \oplus \mathbf{A}[2] \oplus \mathbf{A}[3] \oplus \mathbf{A}[4] \\[4pt]
& \;\vdots \\[4pt]
\text{Step } n: \quad 
& \mathbf{h}^{(n)} = \mathbf{A}[1] \oplus \mathbf{A}[2] \oplus \dots \oplus \mathbf{A}[n] \\[6pt]
\text{Final:} \quad 
& \mathbf{z_G} = \text{MLP}(\mathbf{h}^{(|\mathcal{V}|)}) \\[4pt]
& = \text{MLP}(\mathbf{A}[1] \oplus \mathbf{A}[2] \oplus \dots \oplus \mathbf{A}[|\mathcal{V}|])
\end{align*}$$

## Node Features (not really but similar)

$$\mathbf{A}[1], \mathbf{A}[2], \dots, \mathbf{A}[|\mathcal{V}|]$$

## Aggregate them
### Apply an operation like summation:

$$\mathbf{h_G} = \mathbf{A}[1] \oplus \mathbf{A}[2] \oplus \dots \oplus \mathbf{A}[|\mathcal{V}|]$$ 

$$If \oplus = \sum, then:
\mathbf{h_G} = \sum_{i=1}^{|\mathcal{V}|} \mathbf{A}[I]$$

## Feed the aggregated representation into an MLP

$$\mathbf{z_G} = \text{MLP}(\mathbf{h_G})$$

# Example

Suppose your graph has 3 nodes with features:
$$\mathbf{A}[1] = [1, 0], \quad \mathbf{A}[2] = [0, 1], \quad \mathbf{A}[3] = [1, 1]$$

Aggregation by summation:

$$\mathbf{h_G} = [1, 0] + [0, 1] + [1, 1] = [2, 2]$$

$$\mathbf{z_G} = \text{MLP}([2, 2])$$

$$\mathbf{z_G} = \sigma(\mathbf{W}_2 \, \text{ReLU}(\mathbf{W}_1 [2, 2]^\top + \mathbf{b}_1) + \mathbf{b}_2)$$

If each node feature vector $\mathbf{A}[i]$ has dimension d, and your graph has $|\mathcal{V}|$ nodes, then concatenating them gives:

$$\mathbf{h_G} = \mathbf{A}[1] \oplus \mathbf{A}[2] \oplus \dots \oplus \mathbf{A}[|\mathcal{V}|]
\Rightarrow \mathbf{h_G} \in \mathbb{R}^{d \times |\mathcal{V}|}$$
—or as a single long vector of size $d|\mathcal{V}|$.

So if each $\mathbf{A}[i] = [a₁ᵢ, a₂ᵢ, a₃ᵢ]$ (3 features per node),
and you have 4 nodes,
then:

$$\mathbf{h_G} = [a_{11}, a_{21}, a_{31}, a_{12}, a_{22}, a_{32}, a_{13}, a_{23}, a_{33}, a_{14}, a_{24}, a_{34}]$$

That is permutation variant!

``` python
A = torch.tensor([[1,2,3],
                  [4,5,6],
                  [7,8,9],
                  [10,11,12]])
h_G = torch.cat([A[0], A[1], A[2], A[3]])  # [1,2,3,4,5,6,7,8,9,10,11,12]
z_G = mlp(h_G)
``` 

Shallow encoders are equavariant functions:

Adjacency (path 1–2–3):

$$\mathbf A=\begin{bmatrix}
0&1&0\\
1&0&1\\
0&1&0
\end{bmatrix},\qquad
\mathbf X=\begin{bmatrix}2\\-1\\3\end{bmatrix}$$

Shallow encoder: 

$$f(\mathbf A,\mathbf X)=\mathbf A\mathbf X.$$

Permutation (relabel to order 2,3,1):

$$\mathbf P=\begin{bmatrix}
0&1&0\\
0&0&1\\
1&0&0
\end{bmatrix}$$

Original output

$$\mathbf A\mathbf X=
\begin{bmatrix}
-1\\
5\\
-1
\end{bmatrix}$$

Permuted inputs

$$\mathbf X’=\mathbf P\mathbf X=
\begin{bmatrix}-1\\3\\2\end{bmatrix},\qquad
\mathbf A’=\mathbf P\mathbf A\mathbf P^{\top}=
\begin{bmatrix}
0&1&1\\
1&0&0\\
1&0&0
\end{bmatrix}$$

Output after permutation

$$\mathbf A’\mathbf X’=
\begin{bmatrix}
5\\
-1\\
-1
\end{bmatrix}$$

Compare with permuted original output

$$\mathbf P(\mathbf A\mathbf X)=
\mathbf P
\begin{bmatrix}
-1\\
5\\
-1
\end{bmatrix}$$

$$\begin{bmatrix}
5\\
-1\\
-1
\end{bmatrix}$$

They match, so

$f(\mathbf P\mathbf A\mathbf P^{\top},\,\mathbf P\mathbf X)=\mathbf P\,f(\mathbf A,\mathbf X)$,
which demonstrates permutation equivariance for the shallow encoder $f(\mathbf A,\mathbf X)=\mathbf A\mathbf X$.

# Message Passing

Neural message passing in GNNs: nodes exchange vector “messages” with neighbors and update their states using neural networks. 

message.jpg

At each iteration k:

$$\mathbf{h}_u^{(k+1)} = \text{UPDATE}^{(k)}\big(\mathbf{h}_u^{(k)}, \text{AGGREGATE}^{(k)}(\{\,\mathbf{h}_v^{(k)} \mid v \in \mathcal{N}(u)\,\})\big)$$
$•	\mathbf{h}_u^{(k)}$ — current embedding (feature) of node u 
	
$•	\mathcal{N}(u)$ — neighbors of node u
	
$•	\text{AGGREGATE}$ — combines messages from neighbors (permutation-invariant)
	
$•	\text{UPDATE}$ — updates node u’s representation using its previous embedding and the aggregated message

At the end (after K iterations):

$\mathbf{z}_u = \mathbf{h}_u^{(K)}    \forall{u}$

# Example

1 — 2 — 3

Adjacency list:
	•	1’s neighbors: {2}
	•	2’s neighbors: {1, 3}
	•	3’s neighbors: {2}

⸻

Step 1: Initialize features

At iteration k = 0:

$\mathbf{h}_1^{(0)} = 1, \quad \mathbf{h}_2^{(0)} = 2, \quad \mathbf{h}_3^{(0)} = 3$

⸻

Step 2: Define AGGREGATE and UPDATE

For simplicity:
$\text{AGGREGATE}^{(k)}(\{\,\mathbf{h}v^{(k)}\,\}) = \sum_{v \in \mathcal{N}(u)} \mathbf{h}_v^{(k)}$

$\text{UPDATE}^{(k)}(x, y) = x + y$

⸻

Step 3: Compute iteration k=1

Node 1:

$\text{AGGREGATE} = \mathbf{h}_2^{(0)} = 2$

$\text{UPDATE}: \mathbf{h}_1^{(1)} = 1 + 2 = 3$

Node 2:

$\text{AGGREGATE} = \mathbf{h}_1^{(0)} + \mathbf{h}_3^{(0)} = 1 + 3 = 4$

$\text{UPDATE}: \mathbf{h}_2^{(1)} = 2 + 4 = 6$

Node 3:

$\text{AGGREGATE} = \mathbf{h}_2^{(0)} = 2$

$\text{UPDATE}: \mathbf{h}_3^{(1)} = 3 + 2 = 5$

⸻

Step 4: Output after one iteration

$\mathbf{h}_1^{(1)} = 3, \quad \mathbf{h}_2^{(1)} = 6, \quad \mathbf{h}_3^{(1)} = 5$

⸻

Step 5: Next iteration k=2

Node 1:

$\text{AGGREGATE} = \mathbf{h}_2^{(1)} = 6$

$\text{UPDATE}: \mathbf{h}_1^{(2)} = 3 + 6 = 9$

Node 2:

$\text{AGGREGATE} = \mathbf{h}_1^{(1)} + \mathbf{h}_3^{(1)} = 3 + 5 = 8$

$\text{UPDATE}: \mathbf{h}_2^{(2)} = 6 + 8 = 14$

Node 3:

$\text{AGGREGATE} = \mathbf{h}_2^{(1)} = 6$

$\text{UPDATE}: \mathbf{h}_3^{(2)} = 5 + 6 = 11$

⸻

Step 6: Final embeddings (after K=2)

$\mathbf{z}_1 = 9, \quad \mathbf{z}_2 = 14, \quad \mathbf{z}_3 = 11$


gnn.png

	•	After the first iteration (k = 1), each node’s embedding contains information from its 1-hop neighborhood — i.e., its immediate neighbors.
	•	After the second iteration (k = 2), each node’s embedding includes information from its 2-hop neighborhood (neighbors of neighbors).
	•	In general, after k iterations, every node embedding encodes information from its k-hop neighborhood.

The information captured by GNNs comes in two main forms:

	1.	Structural information – the topology of the graph, such as node degrees or local motifs (e.g., molecular substructures like benzene rings).
	2.	Feature-based information – aggregated attributes or features of neighboring nodes, similar to how CNNs gather information from spatially nearby pixels.