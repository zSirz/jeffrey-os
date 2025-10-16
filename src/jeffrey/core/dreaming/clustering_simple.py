import numpy as np
from typing import List, Dict, Tuple

def cluster_by_similarity(embeddings: List[np.ndarray], threshold: float = 0.7) -> List[List[int]]:
    """
    Clustering simple par similarité cosine sans sklearn
    Complexité O(n²) mais acceptable pour <1000 mémoires

    Args:
        embeddings: Liste des embeddings numpy
        threshold: Seuil de similarité cosine (0.0-1.0)

    Returns:
        Liste de clusters, chaque cluster est une liste d'indices
    """
    n = len(embeddings)
    if n < 2:
        return [[0]] if n == 1 else []

    # Normaliser les embeddings pour calcul cosine efficace
    normalized_embeddings = []
    for emb in embeddings:
        norm = np.linalg.norm(emb)
        if norm > 0:
            normalized_embeddings.append(emb / norm)
        else:
            normalized_embeddings.append(emb)

    # Clustering par seuil
    clusters = []
    assigned = set()

    for i in range(n):
        if i in assigned:
            continue

        cluster = [i]
        assigned.add(i)

        for j in range(i+1, n):
            if j in assigned:
                continue

            # Similarité cosine avec embeddings normalisés
            similarity = np.dot(normalized_embeddings[i], normalized_embeddings[j])
            if similarity >= threshold:
                cluster.append(j)
                assigned.add(j)

        clusters.append(cluster)

    return clusters

def analyze_clusters(clusters: List[List[int]], memories: List[Dict]) -> Dict:
    """
    Analyse des clusters générés

    Args:
        clusters: Résultat de cluster_by_similarity
        memories: Liste des mémoires correspondantes

    Returns:
        Statistiques sur les clusters
    """
    if not clusters or not memories:
        return {"status": "no_data"}

    cluster_stats = []
    emotion_distributions = {}

    for i, cluster_indices in enumerate(clusters):
        cluster_memories = [memories[idx] for idx in cluster_indices]

        # Analyse des émotions dans ce cluster
        emotions = [mem.get('emotion', 'neutral') for mem in cluster_memories]
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

        # Émotion dominante
        dominant_emotion = max(emotion_counts.items(), key=lambda x: x[1])[0] if emotion_counts else 'neutral'

        cluster_stat = {
            "cluster_id": i,
            "size": len(cluster_indices),
            "dominant_emotion": dominant_emotion,
            "emotion_distribution": emotion_counts,
            "indices": cluster_indices
        }

        cluster_stats.append(cluster_stat)

        # Accumulation globale
        for emotion, count in emotion_counts.items():
            emotion_distributions[emotion] = emotion_distributions.get(emotion, 0) + count

    # Statistiques globales
    total_memories = len(memories)
    clustered_memories = sum(len(cluster) for cluster in clusters)

    return {
        "status": "analyzed",
        "total_clusters": len(clusters),
        "total_memories": total_memories,
        "clustered_memories": clustered_memories,
        "clustering_ratio": clustered_memories / total_memories if total_memories > 0 else 0,
        "clusters": cluster_stats,
        "global_emotion_distribution": emotion_distributions,
        "average_cluster_size": clustered_memories / len(clusters) if clusters else 0
    }

def simple_centroid_clustering(embeddings: List[np.ndarray], max_clusters: int = 5) -> List[List[int]]:
    """
    Clustering simple par centroïdes (alternative k-means sans sklearn)

    Args:
        embeddings: Liste des embeddings
        max_clusters: Nombre maximum de clusters

    Returns:
        Liste de clusters
    """
    n = len(embeddings)
    if n <= max_clusters:
        return [[i] for i in range(n)]

    # Initialisation: premiers points comme centroïdes
    centroids = embeddings[:max_clusters].copy()
    clusters = [[] for _ in range(max_clusters)]

    # Itérations simples (max 10 pour éviter les boucles infinies)
    for iteration in range(10):
        # Réinitialiser clusters
        new_clusters = [[] for _ in range(max_clusters)]

        # Assigner chaque point au centroïde le plus proche
        for i, emb in enumerate(embeddings):
            similarities = [np.dot(emb, centroid) for centroid in centroids]
            closest_cluster = np.argmax(similarities)
            new_clusters[closest_cluster].append(i)

        # Éviter les clusters vides
        for i, cluster in enumerate(new_clusters):
            if not cluster:
                new_clusters[i] = [i % n]  # Assign at least one point

        # Mettre à jour centroïdes
        new_centroids = []
        for cluster_indices in new_clusters:
            if cluster_indices:
                cluster_embeddings = [embeddings[idx] for idx in cluster_indices]
                centroid = np.mean(cluster_embeddings, axis=0)
                new_centroids.append(centroid)
            else:
                new_centroids.append(centroids[len(new_centroids)])

        # Vérifier convergence
        if np.allclose(centroids, new_centroids, rtol=1e-3):
            break

        centroids = new_centroids
        clusters = new_clusters

    # Filtrer les clusters vides
    return [cluster for cluster in clusters if cluster]