Here we present a novel approach to the classification of Japanese adverbs. 

Traditional methods for categorizing Japanese adverbs have faced challenges, with little advancement since the 1930s. In the age of large language models, we aim to provide a data-driven framework for grouping adverbs that relies on quantitative evidence rather than purely theoretical classification. Our method combines the power of cutting-edge language models, such as BERT and RoBERTa, with fuzzy set theory to create empirical groupings of Japanese adverbs based on their semantic relationships. Here's a brief overview of our approach:

Embedding Generation: We start by generating multi-dimensional embeddings for a list of Japanese adverbs using pretrained BERT or RoBERTa models trained on Japanese text. This step captures the rich semantic nuances of the adverbs.

Dimensionality Reduction: To streamline the data and reduce noise, we utilize Principal Component Analysis (PCA) to reduce the dimensionality of each embedding.

Semantic Positioning: We plot the relative positions of adverbs in a 3D space using K-means clustering, initially with a cluster count of n=3, to group adverbs based on their semantic similarities.

Optimal Cluster Count: We employ silhouette analysis to determine the optimal cluster count. Surprisingly, our analysis suggests that Japanese adverbs optimally cluster into n=4 groups instead of n=3.

Intuitive Visualization: In addition to 3D positioning, we create 2D semantic position plots, providing an intuitive way to visualize the relationships among adverbs.

Quantitative Clustering: A centroid distance matrix is generated, offering a quantitative "fingerprint" for Japanese adverbs and revealing how they cluster together.

Our innovative approach offers several advantages over traditional adverb classification methods. It harnesses the capabilities of modern computational techniques to uncover empirical patterns in the semantic relationships among Japanese adverbs, making the results more interpretable and data-driven.


# Japanese Adverb Classification

Welcome to our Japanese Adverb Classification repository, where we introduce a novel approach to categorizing Japanese adverbs. Traditional adverb classification methods have faced challenges, and progress has been limited since the 1930s. In the era of large language models, our goal is to provide a data-driven framework for adverb grouping that relies on quantitative evidence instead of purely theoretical categorization.

## Approach Overview

Our approach combines advanced language models, such as BERT and RoBERTa, with fuzzy set theory to empirically group Japanese adverbs based on their semantic relationships. Here's a brief overview of our methodology:

1. **Embedding Generation**: We start by generating multi-dimensional embeddings for a list of Japanese adverbs using pretrained BERT or RoBERTa models specifically trained on Japanese text. This step captures the nuanced semantics of the adverbs.

2. **Dimensionality Reduction**: To streamline the data and reduce noise, we utilize Principal Component Analysis (PCA) to reduce the dimensionality of each embedding.

3. **Semantic Positioning**: Adverbs are plotted in a 3D space using K-means clustering, initially with a cluster count of n=3, to group them based on semantic similarities.

4. **Optimal Cluster Count**: Silhouette analysis helps us determine the optimal cluster count. Interestingly, our analysis suggests that Japanese adverbs optimally cluster into n=4 groups rather than n=3.

5. **Intuitive Visualization**: In addition to 3D positioning, we create 2D semantic position plots, providing an intuitive way to visualize adverb relationships.

6. **Quantitative Clustering**: We generate a centroid distance matrix, offering a quantitative "fingerprint" for Japanese adverbs, revealing how they cluster together.

## Advantages

Our innovative approach offers several advantages over traditional adverb classification methods:

- Empirical and data-driven
- Intuitive visualization of semantic relationships
- Quantitative "fingerprint" for adverb clustering

Join us on this linguistic journey as we explore new horizons in Japanese adverb classification using state-of-the-art techniques.

## Getting Started

To get started with our adverb classification approach, please refer to the [Installation](#installation) and [Usage](#usage) sections in this repository.
