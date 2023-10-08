# Japanese Adverb Classification

Here, we introduce a novel approach to categorizing Japanese adverbs. Traditional adverb classification methods have faced challenges, and progress has been limited since the 1930s. In the era of large language models, our goal is to provide a data-driven framework for adverb grouping that relies on quantitative evidence instead of purely theoretical categorization.

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

### Files

1. **Japanese_adverbs.csv**: This CSV file contains a comprehensive list of Japanese adverbs used in our analysis, along with their English translations. It serves as the foundational dataset for our project.

2. **tohoku_BERT_analysis.ipynb**: This Jupyter Notebook explores the analysis of Japanese adverbs using the BERT model pretrained on Japanese text. It covers the steps for generating embeddings, dimensionality reduction, semantic positioning, and clustering. You can use this notebook to delve into the details of our BERT-based analysis.

3. **waseda_RoBERTa_analysis.ipynb**: In this Jupyter Notebook, we delve into the analysis of Japanese adverbs using the RoBERTa model pretrained on Japanese text. The notebook guides you through the process of generating embeddings, reducing dimensionality, performing semantic positioning, and clustering. It provides insights into our RoBERTa-based analysis.

## Research Paper

Our research paper, titled "Semantic Positioning of Japanese Adverbs Using Advanced Language Models," provides a comprehensive overview of the methodology, results, and insights derived from this repository. You can access the full paper by following this link: [Research Paper - Semantic Positioning of Japanese Adverbs](https://example.com/your-research-paper-link).

### Citation

If you use our research or findings in your work, we kindly request that you cite our paper using the following citation format:

## License

This project is open-source and is released under the [MIT License](LICENSE). Feel free to use and build upon our work while giving appropriate credit.
