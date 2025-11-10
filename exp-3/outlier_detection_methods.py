"""
Different outlier detection methods for comparing 4 embedding vectors.

This module provides multiple algorithms to identify which of 4 embeddings
is the semantic outlier. Use this to experiment and find the best approach.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from typing import Literal


def find_outlier_centroid(embeddings: np.ndarray, normalize: bool = False) -> int:
    """
    Current method: Compare each embedding to centroid of the other 3.

    For each option i:
      - Calculate centroid of options {j : j != i}
      - Compute cosine similarity between option i and this centroid
      - Return option with LOWEST similarity

    Pros: Intuitive, fast
    Cons: Centroid can be skewed if one of the "others" is also somewhat different

    Args:
        embeddings: (4, dim) array of embeddings
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Index (0-3) of the predicted outlier
    """
    if len(embeddings) != 4 or embeddings.ndim != 2:
        return -1

    # Optional normalization
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    similarity_scores = []
    for i in range(4):
        others = np.delete(embeddings, i, axis=0)
        centroid = np.mean(others, axis=0, keepdims=True)
        current = embeddings[i].reshape(1, -1)
        similarity = cosine_similarity(current, centroid)[0][0]
        similarity_scores.append(similarity)

    return int(np.argmin(similarity_scores))


def find_outlier_pairwise_avg(embeddings: np.ndarray, normalize: bool = False) -> int:
    """
    Improved method: Average pairwise similarity.

    For each option i:
      - Calculate average cosine similarity to ALL other options
      - Return option with LOWEST average similarity

    Pros: More robust, symmetric, considers all relationships
    Cons: Slightly more computation (negligible for 4 items)

    This is theoretically better because:
    - Each option is compared to all others equally
    - Not dependent on centroid calculation
    - More aligned with intuition: "which is least similar to the group?"

    Args:
        embeddings: (4, dim) array of embeddings
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Index (0-3) of the predicted outlier
    """
    if len(embeddings) != 4 or embeddings.ndim != 2:
        return -1

    # Optional normalization
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute full pairwise similarity matrix
    sim_matrix = cosine_similarity(embeddings)

    # For each option, calculate average similarity to others (excluding self)
    avg_similarities = []
    for i in range(4):
        # Get similarities to all others (exclude diagonal which is self-similarity = 1.0)
        other_sims = [sim_matrix[i][j] for j in range(4) if j != i]
        avg_sim = np.mean(other_sims)
        avg_similarities.append(avg_sim)

    # Return index with lowest average similarity
    return int(np.argmin(avg_similarities))


def find_outlier_max_distance_sum(embeddings: np.ndarray, normalize: bool = False) -> int:
    """
    Alternative method: Maximum distance sum.

    For each option i:
      - Sum the cosine distances to all other options
      - Return option with MAXIMUM total distance

    Note: Cosine distance = 1 - cosine similarity

    Args:
        embeddings: (4, dim) array of embeddings
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Index (0-3) of the predicted outlier
    """
    if len(embeddings) != 4 or embeddings.ndim != 2:
        return -1

    # Optional normalization
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute pairwise similarities
    sim_matrix = cosine_similarity(embeddings)

    # Convert to distances and sum
    distance_sums = []
    for i in range(4):
        # Cosine distance = 1 - similarity
        distances = [1 - sim_matrix[i][j] for j in range(4) if j != i]
        total_distance = np.sum(distances)
        distance_sums.append(total_distance)

    # Return index with maximum total distance
    return int(np.argmax(distance_sums))


def find_outlier_euclidean(embeddings: np.ndarray, normalize: bool = False) -> int:
    """
    Alternative metric: Use Euclidean distance instead of cosine similarity.

    For each option i:
      - Calculate average Euclidean distance to all other options
      - Return option with MAXIMUM average distance

    Args:
        embeddings: (4, dim) array of embeddings
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Index (0-3) of the predicted outlier
    """
    if len(embeddings) != 4 or embeddings.ndim != 2:
        return -1

    # Optional normalization
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    # Compute pairwise Euclidean distances
    dist_matrix = euclidean_distances(embeddings)

    # Calculate average distance for each option
    avg_distances = []
    for i in range(4):
        other_dists = [dist_matrix[i][j] for j in range(4) if j != i]
        avg_dist = np.mean(other_dists)
        avg_distances.append(avg_dist)

    # Return index with maximum average distance
    return int(np.argmax(avg_distances))


def find_outlier_isolation_score(embeddings: np.ndarray, normalize: bool = False) -> int:
    """
    Advanced method: Isolation score.

    For each option i:
      - Calculate how much more similar the other 3 are to each other
        compared to their similarity to option i
      - Higher isolation score = more likely to be outlier

    Args:
        embeddings: (4, dim) array of embeddings
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Index (0-3) of the predicted outlier
    """
    if len(embeddings) != 4 or embeddings.ndim != 2:
        return -1

    # Optional normalization
    if normalize:
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    sim_matrix = cosine_similarity(embeddings)

    isolation_scores = []
    for i in range(4):
        # Get indices of the other 3 options
        others = [j for j in range(4) if j != i]

        # Average similarity among the other 3 (internal cohesion)
        internal_sims = [sim_matrix[others[a]][others[b]]
                        for a in range(3) for b in range(3) if a < b]
        avg_internal = np.mean(internal_sims) if internal_sims else 0

        # Average similarity from option i to the other 3
        external_sims = [sim_matrix[i][j] for j in others]
        avg_external = np.mean(external_sims)

        # Isolation score: how much more cohesive are the others compared to i?
        isolation = avg_internal - avg_external
        isolation_scores.append(isolation)

    # Return index with highest isolation score
    return int(np.argmax(isolation_scores))


# Factory function to get any method by name
def find_outlier(
    embeddings: np.ndarray,
    method: Literal[
        'centroid',
        'pairwise_avg',
        'max_distance_sum',
        'euclidean',
        'isolation'
    ] = 'pairwise_avg',
    normalize: bool = False
) -> int:
    """
    Unified interface for all outlier detection methods.

    Args:
        embeddings: (4, dim) array of embeddings
        method: Which detection method to use
        normalize: Whether to L2-normalize embeddings first

    Returns:
        Index (0-3) of the predicted outlier, or -1 if failed
    """
    methods = {
        'centroid': find_outlier_centroid,
        'pairwise_avg': find_outlier_pairwise_avg,
        'max_distance_sum': find_outlier_max_distance_sum,
        'euclidean': find_outlier_euclidean,
        'isolation': find_outlier_isolation_score,
    }

    if method not in methods:
        raise ValueError(f"Unknown method: {method}. Choose from {list(methods.keys())}")

    return methods[method](embeddings, normalize=normalize)


if __name__ == "__main__":
    # Example usage and comparison
    print("Testing different outlier detection methods...")
    print("="*60)

    # Create a simple test case: 3 similar vectors and 1 outlier
    # Vectors 0, 1, 2 are similar (around [1, 0, 0])
    # Vector 3 is different (around [0, 1, 0])
    test_embeddings = np.array([
        [1.0, 0.1, 0.0],  # Similar to group
        [0.9, 0.2, 0.1],  # Similar to group
        [1.1, 0.0, 0.1],  # Similar to group
        [0.2, 0.9, 0.1],  # OUTLIER - different direction
    ])

    print("Test embeddings (option 3 should be the outlier):\n")
    for i, emb in enumerate(test_embeddings):
        print(f"  Option {i}: {emb}")

    print("\n" + "="*60)
    print("Results from different methods:")
    print("="*60)

    methods = ['centroid', 'pairwise_avg', 'max_distance_sum', 'euclidean', 'isolation']

    for method in methods:
        outlier_idx = find_outlier(test_embeddings, method=method)
        status = "✓ CORRECT" if outlier_idx == 3 else "✗ WRONG"
        print(f"{method:20s} → Outlier: {outlier_idx}  {status}")

    print("\n" + "="*60)
    print("Test with normalization:")
    print("="*60)

    for method in methods:
        outlier_idx = find_outlier(test_embeddings, method=method, normalize=True)
        status = "✓ CORRECT" if outlier_idx == 3 else "✗ WRONG"
        print(f"{method:20s} → Outlier: {outlier_idx}  {status}")
