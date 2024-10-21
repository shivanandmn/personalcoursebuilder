from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


def find_optimal_clusters(embeddings, max_clusters=10):
    """
    Find the optimal number of clusters using the silhouette score.

    Parameters:
    embeddings (ndarray): The embeddings of the input skills.
    max_clusters (int): The maximum number of clusters to test.

    Returns:
    int: The optimal number of clusters based on the silhouette score.
    """
    silhouette_scores = []

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=0)
        cluster_labels = kmeans.fit_predict(embeddings)
        silhouette_avg = silhouette_score(embeddings, cluster_labels)
        silhouette_scores.append(silhouette_avg)

    # Find the number of clusters with the highest Silhouette Score
    optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
    return optimal_clusters


def cluster_skills(skills, model_name="all-MiniLM-L6-v2", max_clusters=10):
    """
    Cluster the given skills using KMeans and SentenceTransformer embeddings.

    Parameters:
    skills (list): A list of skills to be clustered.
    model_name (str): The name of the pre-trained SentenceTransformer model.
    max_clusters (int): The maximum number of clusters to test.

    Returns:
    dict: A dictionary with cluster labels as keys and lists of skills as values.
    """
    # Load the pre-trained model and generate embeddings
    model = SentenceTransformer(model_name)
    embeddings = model.encode(skills)

    # Find the optimal number of clusters
    optimal_clusters = find_optimal_clusters(embeddings, max_clusters)

    # Run KMeans with the optimal number of clusters
    kmeans_optimal = KMeans(n_clusters=optimal_clusters, random_state=0)
    kmeans_optimal.fit(embeddings)
    clusters = kmeans_optimal.labels_

    # Organize skills by their cluster labels
    cluster_dict = {}
    for skill, cluster_label in zip(skills, clusters):
        if cluster_label not in cluster_dict:
            cluster_dict[cluster_label] = []
        cluster_dict[cluster_label].append(skill)

    return cluster_dict


def main():
    # List of skills to be clustered
    skills = [
        "Probability",
        "Statistics",
        "Linear Algebra",
        "Programming",
        "Machine Learning",
        "NLP tasks",
        "Topic modelling",
        "Entity Extraction",
        "Summarization",
        "Sentiment analysis",
        "Object detection",
        "Image segmentation",
        "Image classification",
        "AWS",
        "Azure",
        "Google Cloud",
        "Cloud AI services",
        "Cloud AI tools",
        "Python",
        "R",
        "TensorFlow",
        "PyTorch",
        "scikit-learn",
        "MLOps",
    ]

    # Cluster the skills and display the results
    clustered_skills = cluster_skills(skills)
    print("Optimal Clusters and Corresponding Skills:")
    for cluster, skills in clustered_skills.items():
        print(f"Cluster {cluster}: {skills}")


if __name__ == "__main__":
    main()
