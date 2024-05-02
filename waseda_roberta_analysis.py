import sys
import os
import csv
from transformers import AutoTokenizer, AutoModelForMaskedLM
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

def main():
    # Check if correct number of arguments are passed
    if len(sys.argv) != 4:
        print("Usage: python script.py <csv_file_path> <output_dir> <model_name>")
        sys.exit(1)

    # Extract command-line arguments
    csv_file_path = sys.argv[1]
    output_dir = sys.argv[2]
    model_name = sys.argv[3]

    # Prepend "nlp-waseda/" to the model name
    model_type = f"nlp-waseda/{model_name}"

    # Load BERT tokenizer and model based on the specified model type
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForMaskedLM.from_pretrained(model_type)

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Read the CSV file with adverbs
    data = pd.read_csv(csv_file_path, header=None, encoding="utf-8")
    adverbs = data.iloc[:, 0].tolist()
    true_labels = data.iloc[:, 1].tolist()

    # Calculate embeddings
    embeddings = []
    for adverb in adverbs:
        inputs = tokenizer.encode_plus(adverb, return_tensors="pt", add_special_tokens=True)
        with torch.no_grad():
            output = model(**inputs)
            hidden_states = output.logits

        embedding = hidden_states.mean(dim=1).cpu().numpy()
        embeddings.append(embedding)

    embedding_array = np.array(embeddings)
    num_adverbs, num_tokens, embedding_dim = embedding_array.shape
    flattened_embedding_array = embedding_array.reshape(-1, embedding_dim)

    # Normalize embeddings
    min_max_scaler = MinMaxScaler()
    min_max_normalized_embedding_array = min_max_scaler.fit_transform(flattened_embedding_array)
    min_max_normalized_embedding_array = min_max_normalized_embedding_array.reshape(num_adverbs, num_tokens, embedding_dim)
    flattened_normalized_embedding_array = min_max_normalized_embedding_array.reshape(-1, embedding_dim)

    # Reduce dimensionality using PCA
    num_dimensions = 3
    pca = PCA(n_components=num_dimensions)
    reduced_embedding_array = pca.fit_transform(flattened_normalized_embedding_array)

    # K-Means clustering
    num_clusters = 3
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_embedding_array)

    # Generate 3D scatter plot with cluster centroids
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    cluster_names = ['Cluster A', 'Cluster B', 'Cluster C']
    for i, cluster_name in enumerate(cluster_names):
        ax.scatter(
            reduced_embedding_array[cluster_labels == i, 0],
            reduced_embedding_array[cluster_labels == i, 1],
            reduced_embedding_array[cluster_labels == i, 2],
            label=cluster_name
        )
    ax.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        kmeans.cluster_centers_[:, 2],
        marker='X',
        color='black',
        label='Centroids'
    )

    for i, (adverb, true_label) in enumerate(zip(adverbs, true_labels)):
        ax.text(
            reduced_embedding_array[i, 0],
            reduced_embedding_array[i, 1],
            reduced_embedding_array[i, 2],
            f'{adverb} ({true_label})',
            fontsize=8,
            ha='center'
        )

    ax.set_title('3D Cluster Plot with Centroids')
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    ax.legend()

    # Save 3D scatter plot
    plt.savefig(os.path.join(output_dir, 'pre_analysis_3d_scatterplot.png'), dpi=600)
    plt.close()

    # Calculate distance matrix between cluster centroids
    centroid_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
    np.savetxt(os.path.join(output_dir, 'pre_dist.csv'), centroid_distances, delimiter=",")

    # Silhouette analysis
    sil_scores = []
    for n_clusters in range(2, 50):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        cluster_labels = kmeans.fit_predict(flattened_normalized_embedding_array)
        sil_score = silhouette_score(flattened_normalized_embedding_array, cluster_labels)
        sil_scores.append(sil_score)
        print(f"Number of clusters: {n_clusters}, Silhouette Score: {sil_score}")

    plt.figure()
    plt.plot(range(2, 50), sil_scores)
    plt.title('Silhouette Score')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette Score')
    plt.savefig(os.path.join(output_dir, 'silhouette_analysis.png'), dpi=600)
    plt.close()

    # Re-analyze adverb embeddings with optimized number of clusters
    num_dimensions = 2
    pca = PCA(n_components=num_dimensions)
    reduced_embedding_array = pca.fit_transform(flattened_normalized_embedding_array)

    num_clusters = 4
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    cluster_labels = kmeans.fit_predict(reduced_embedding_array)

    adverb_label_dict = {adverb: true_label for adverb, true_label in zip(adverbs, true_labels)}
    adverb_cluster_dict = {adverb: cluster_label for adverb, cluster_label in zip(adverbs, cluster_labels)}
    adverb_true_label_to_cluster = {adverb: {"TrueLabel": adverb_label_dict[adverb], "ClusterLabel": adverb_cluster_dict[adverb]} for adverb in adverbs}

    output_csv_file_path = os.path.join(output_dir, "post_analysis.csv")
    with open(output_csv_file_path, mode='w', newline='', encoding='utf-8') as csv_file:
        fieldnames = ['Adverb', 'TrueLabel', 'ClusterLabel']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for adverb, labels in adverb_true_label_to_cluster.items():
            writer.writerow({'Adverb': adverb, 'TrueLabel': labels['TrueLabel'], 'ClusterLabel': labels['ClusterLabel']})

    print(f"CSV file '{output_csv_file_path}' has been created.")

    plt.figure(figsize=(10, 8))
    cluster_names = ['Cluster A', 'Cluster B', 'Cluster C', 'Cluster D']
    for i, cluster_name in enumerate(cluster_names):
        plt.scatter(
            reduced_embedding_array[cluster_labels == i, 0],
            reduced_embedding_array[cluster_labels == i, 1],
            label=cluster_name
        )
    plt.scatter(
        kmeans.cluster_centers_[:, 0],
        kmeans.cluster_centers_[:, 1],
        marker='X',
        color='black',
        label='Centroids'
    )

    for i, (adverb, true_label) in enumerate(zip(adverbs, true_labels)):
        plt.annotate(f'{adverb} ({true_label})', (reduced_embedding_array[i, 0], reduced_embedding_array[i, 1]))

    plt.title('Cluster Plot with Centroids')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend()

    # Save scatter plot
    plt.savefig(os.path.join(output_dir, 'post_analysis_scatterplot.png'), dpi=600)
    plt.close()

    centroid_distances = cdist(kmeans.cluster_centers_, kmeans.cluster_centers_)
    np.savetxt(os.path.join(output_dir, 'post_dist.csv'), centroid_distances, delimiter=",")

if __name__ == "__main__":
    main()
