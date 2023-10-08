Here we present a novel approach to the classification of Japanese adverbs. 

Traditional methods for categorizing Japanese adverbs have faced challenges, with little advancement since the 1930s. In the age of large language models, we aim to provide a data-driven framework for grouping adverbs that relies on quantitative evidence rather than purely theoretical classification. Our method combines the power of cutting-edge language models, such as BERT and RoBERTa, with fuzzy set theory to create empirical groupings of Japanese adverbs based on their semantic relationships. Here's a brief overview of our approach:

Embedding Generation: We start by generating multi-dimensional embeddings for a list of Japanese adverbs using pretrained BERT or RoBERTa models trained on Japanese text. This step captures the rich semantic nuances of the adverbs.

Dimensionality Reduction: To streamline the data and reduce noise, we utilize Principal Component Analysis (PCA) to reduce the dimensionality of each embedding.

Semantic Positioning: We plot the relative positions of adverbs in a 3D space using K-means clustering, initially with a cluster count of n=3, to group adverbs based on their semantic similarities.

Optimal Cluster Count: We employ silhouette analysis to determine the optimal cluster count. Surprisingly, our analysis suggests that Japanese adverbs optimally cluster into n=4 groups instead of n=3.

Intuitive Visualization: In addition to 3D positioning, we create 2D semantic position plots, providing an intuitive way to visualize the relationships among adverbs.

Quantitative Clustering: A centroid distance matrix is generated, offering a quantitative "fingerprint" for Japanese adverbs and revealing how they cluster together.

Our innovative approach offers several advantages over traditional adverb classification methods. It harnesses the capabilities of modern computational techniques to uncover empirical patterns in the semantic relationships among Japanese adverbs, making the results more interpretable and data-driven.
