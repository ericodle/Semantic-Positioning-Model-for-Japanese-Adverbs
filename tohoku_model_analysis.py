import sys
import os
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler
from scipy.spatial.distance import cdist


def main(csv_file_path, output_dir, model_type):
    # Load the BERT tokenizer and model based on the specified model type
    if model_type == "bert-large-japanese":
        tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-large-japanese")
        model = BertModel.from_pretrained("cl-tohoku/bert-large-japanese")
    elif model_type == "bert-base-japanese":
        tokenizer = BertTokenizer.from_pretrained("cl-tohoku/bert-base-japanese")
        model = BertModel.from_pretrained("cl-tohoku/bert-base-japanese")
    else:
        print("Invalid BERT model type specified. Please choose 'bert-large-japanese' or 'bert-base-japanese'.")
        return

    # Read the CSV file using pandas
    data = pd.read_csv(csv_file_path, header=None, encoding="utf-8")

    # Get the adverbs from the first column
    adverbs = data.iloc[:, 0].tolist()
    true_labels = data.iloc[:, 1].tolist()

    # Calculate embeddings and store them in a list
    embeddings = []
    for adverb in adverbs:
        # Tokenize and convert to BERT input format
        inputs = tokenizer(adverb, return_tensors="pt")

        # Get the embeddings from the BERT model
        with torch.no_grad():
            output = model(**inputs)
            embedding = output.last_hidden_state[:, 0, :].numpy()  # Using [CLS] token embedding

        embeddings.append(embedding)

    # Convert embeddings list to a numpy array
    embedding_array = np.array(embeddings)

    # Reshape the embedding array for PCA
    num_adverbs, num_tokens, embedding_dim = embedding_array.shape

    # Flatten the normalized embedding array before applying Min-Max Normalization
    flattened_embedding_array = embedding_array.reshape(-1, embedding_dim)

    # Apply Min-Max Normalization
    min_max_scaler = MinMaxScaler()
    min_max_normalized_embedding_array = min_max_scaler.fit_transform(flattened_embedding_array)

    # Reshape the normalized embedding array back to its original shape
    min_max_normalized_embedding_array = min_max_normalized_embedding_array.reshape(num_adverbs, num_tokens, embedding_dim)

    # Flatten the 3D array to a 2D array
    flattened_normalized_embedding_array = min_max_normalized_embedding_array.reshape(-1, embedding_dim)

    # Reduce dimensionality using PCA
    num_dimensions = 3  # Using 3 components for PCA
    pca = PCA(n_components=num_dimensions)
    reduced_embedding_array = pca.fit_transform(flattened_normalized_embedding_array)

    # Use K-Means clustering with 3 clusters
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_embedding_array)

    # Generate a 3D scatter plot with cluster centroids
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cluster_names = ['Cluster A', 'Cluster B', 'Cluster C']
    for i, cluster_name in enumerate(cluster_names):
        ax.scatter(
            reduced_embedding_array[cluster_labels == i, 0],
            reduced_embedding_array[cluster_labels == i, 1],
            reduced_embedding_array[cluster_labels == i, 2],  # Use the third component
            label=cluster_name
        )
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        kmeans.cluster_centers_[:, 2],  # Use the third component
        marker='X',
        color='black',
        label='Centroids'
    )

    # Label the points with adverbs and true labels
    for i, (adverb, true_label) in enumerate(zip(adverbs, true_labels)):
        ax.text(
            reduced_embedding_array[i, 0],
            reduced_embedding_array[i, 1],
            reduced_embedding_array[i, 2],  # Use the third component
            f'{adverb} ({true_label})',
            fontsize=8,
            ha='center'
        )

    ax.set_title('3D Cluster Plot with Centroids')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

    # Save the plot as a PNG image with 600 DPI
    fig_path = os.path.join(output_dir, 'pre_analysis_3d_scatterplot.png')
    plt.savefig(fig_path, dpi=600)
    plt.close()

    # Calculate the distance matrix between cluster centroids
    centroid_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)

    # Save the distance matrix to a CSV file
    dist_path = os.path.join(output_dir, 'pre_dist.csv')
    np.savetxt(dist_path, centroid_distances, delimiter=",")

    # Silhouette Score
    sil_scores = []
    for n_clusters in range(2, 50):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(flattened_normalized_embedding_array)
        sil_score = silhouette_score(flattened_normalized_embedding_array, cluster_labels)
        sil_scores.append(sil_score)

    # Save silhouette analysis plot
    silhouette_path = os.path.join(output_dir, 'silhouette_analysis.png')
    plt.figure()
    plt.plot(range(2, 50), sil_scores)
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(silhouette_path, dpi=600)
    plt.close()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script_name.py csv_file_path output_directory bert_model_type")
        sys.exit(1)
    csv_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    bert_model_type = sys.argv[3]
    main(csv_file_path, output_dir, bert_model_type)
