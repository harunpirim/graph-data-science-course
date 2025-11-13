# Graph autoencoders

Convolutional GNNs and attention GNNs are both discriminative models, meaning they’re designed to tell apart different data points, like figuring out if a picture is of a cat or a dog.

Generative models are like trying to understand the whole picture, instead of just focusing on the edges. Discriminative models, on the other hand, are more about spotting the differences between things. For instance, a generative model would pick up on how to create images of cats and dogs, not just the specific features that make them different, like a cat’s pointed ears or a spaniel’s long ears.


The aim here is to extend generative architectures to act on graph-structured data, leading to graph autoencoders (GAEs) and variational graph autoencoders (VGAEs).

Discriminative --> We want our models to return the probability of some target, Y, given an instance of data, X. We can write this as P(Y|X)

$\max_{w,b} \prod_{n=1}^N P(y_n \mid x_n)$

Generative --> they model the joint probability between data and targets, P(X,Y)

$\max_{\theta} \prod_{n=1}^N P(x_n, y_n)$

Both models output a probability P(Y|X).
The difference is what they learn to get there.

gen.jpg


Generative models can be a great way to handle data labeling when it’s a bit pricey, but creating datasets is a breeze.

Supports data augmentation.

Also, discriminative models are frequently used after generative models. This is because generative models usually learn without labels, using a “self-supervised” approach. They figure out how to compress or encode complex, high-dimensional data into simpler forms.


## Graph autoencoders for link prediction

	1.Define the model:
	Create both an encoder and decoder.
	Use the encoder to create a latent space to sample from.
	2.Define the training and testing loop by including a loss suitable for constructing a generative model.
	3.Prepare the data as a graph, with edge lists and node features.
	4.Train the model, passing the edge data to compute the loss.
	5.Test the model using the test dataset.


The GAE model is similar to a typical autoencoder. The only difference is that each individual layer of our network is a GNN, such as a GCN or GraphSAGE.

encoder.jpg


## Building a variational graph autoencoder

VAE.jpg



## Reference

Graph Neural Networks in Action, K. Broadwater, 2025

---
Annotations: 0,2337 SHA-256 b5fca5d38a1dc70e57f7  
&Writing Tools: 22 64,4 92,92 227,196 445,16 469,3 499,71 585 589,19 1206,75 1299,10 1339,4 1371,21 1446,32 1480 1502,29 1561 1585,2 1590,12  
@harun <HP>: 206,2 609,22 776,21 918,3 961,17 1044,3 1086,3 1177,12 1309,30 1604,6 1648,6 1671 1708 1766,3 1872,3 1939,3 2002,3 2043,3 2198,20 2258,79  
...
