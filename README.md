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


## Prerequisite

Install [Python3](https://www.python.org/downloads/) on your computer.

Enter this into your computer's command line interface (terminal, control panel, etc.) to check the version:

  ```sh
  python --version
  ```

If the first number is not a 3, update to Python3.

## Setup

Here is an easy way to use our GitHub repository.

### Step 1: Clone the repository


Open the command line interface and run:
  ```sh
  git clone https://github.com/ericodle/Semantic-Positioning-Model-for-Japanese-Adverbs.git
  ```

You have now downloaded the entire project, including all its sub-directories (folders) and files.
(We will avoid using Git commands.)

### Step 2: Navigate to the project directory
Find where your computer saved the project, then enter:

  ```sh
  cd /path/to/project/directory
  ```

If performed correctly, your command line interface should resemble

```
user@user:~/Semantic-Positioning-Model-for-Japanese-Adverbs-main$
```

### Step 3: Create a virtual environment: 
Use a **virtual environment** so library versions on your computer match the versions used during development and testing.


```sh
python3 -m venv adverbs-env
```

A virtual environment named "adverbs-env" has been created. 
Enter the environment by using the following command:


```sh
source adverbs-env/bin/activate
```

When performed correctly, your command line interface prompt should look like 

```
(adverbs-env) user@user:~//Semantic-Positioning-Model-for-Japanese-Adverbs-main$
```

### Step 3: Install requirements.txt

Avoid dependency hell by installing specific software versions known to work well together.

  ```sh
pip install -r requirements.txt
  ```

### Step 4: run_analysis.py

This script analyzes adverb embeddings using pre-trained language models by clustering them with K-Means, visualizing the clusters in both 3D and 2D scatter plots, and conducting silhouette analysis to determine an optimal number of clusters. It then saves the results, including cluster assignments and distances between centroids, to files for further analysis.

```sh
python3 ./run_analysis.py ./adverbs.csv ./test_output bert bert-base-japanese
```

## Citing Our Research

Our research paper provides a comprehensive overview of the methodology, results, and insights derived from this repository. You can access the full paper by following this link: [Semantic Positioning Model Incorporating BERT/RoBERTa and Fuzzy Theory Achieves More Nuanced Japanese Adverb Clustering](https://www.mdpi.com/2079-9292/12/19/4185/pdf).

If you find our research and code useful in your work, we kindly request that you cite our associated research paper in your publications. You can find the paper through the following citation:

Odle, E.; Hsueh, Y.-J.; Lin,
P.-C. Semantic Positioning Model
Incorporating BERT/RoBERTa and
Fuzzy Theory Achieves More
Nuanced Japanese Adverb
Clustering. Electronics 2023, 12, 4185.
https://doi.org/10.3390/
electronics12194185

## License

This project is open-source and is released under the [MIT License](LICENSE). Feel free to use and build upon our work while giving appropriate credit.
