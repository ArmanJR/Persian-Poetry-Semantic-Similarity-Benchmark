"""
Compare different outlier detection methods on cached embeddings.

This script loads previously cached embeddings and tests all outlier detection
methods to see which performs best on Persian poetry data.
"""

import hashlib
import json
import logging
import sys
import numpy as np
from pathlib import Path
from outlier_detection_methods import find_outlier
import pandas as pd

LOGGER = logging.getLogger(__name__)


def load_cached_embeddings(cache_dir: str, model: str) -> dict:
    """Load all cached embeddings for a specific model."""
    cache_path = Path(cache_dir)
    model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]

    embeddings_cache = {}

    if not cache_path.exists():
        return embeddings_cache

    for cache_file in cache_path.glob(f"{model_hash}_*.json"):
        try:
            with open(cache_file, "r") as f:
                data = json.load(f)
                if "embedding" in data:
                    # Extract hash from filename
                    hash_key = cache_file.stem.replace(f"{model_hash}_", "")
                    embeddings_cache[hash_key] = data["embedding"]
        except Exception as e:
            LOGGER.warning("Failed to load %s: %s", cache_file, e)

    return embeddings_cache


def compare_methods_on_dataset(dataset_path: str, cache_dir: str, model: str):
    """
    Compare all outlier detection methods using cached embeddings.

    Args:
        dataset_path: Path to benchmark_dataset.json
        cache_dir: Path to openrouter_cache directory
        model: Model name to test (must have cached embeddings)
    """
    # Load dataset
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    # Load cached embeddings
    cache = load_cached_embeddings(cache_dir, model)

    if not cache:
        LOGGER.error(
            "No cached embeddings found for %s; run openrouter_benchmark.py first",
            model,
        )
        return

    LOGGER.info("Loaded %d cached embeddings for %s", len(cache), model)

    # Methods to test
    methods = [
        ("centroid", False),
        ("centroid", True),
        ("pairwise_avg", False),
        ("pairwise_avg", True),
        ("max_distance_sum", False),
        ("max_distance_sum", True),
        ("euclidean", False),
        ("euclidean", True),
        ("isolation", False),
        ("isolation", True),
    ]

    method_results = {}

    for method_name, normalize in methods:
        key = f"{method_name}{'_norm' if normalize else ''}"
        method_results[key] = {
            "correct": 0,
            "total": 0,
            "method": method_name,
            "normalize": normalize,
        }

    # Process each question
    for item in dataset:
        if "options" not in item or len(item["options"]) != 4:
            continue
        if "correct_answer_index" not in item:
            continue

        options = item["options"]
        true_outlier = item["correct_answer_index"]

        # Try to load embeddings for all 4 options
        embeddings = []
        for option_text in options:
            text_hash = hashlib.sha256(option_text.encode("utf-8")).hexdigest()
            if text_hash in cache:
                embeddings.append(cache[text_hash])
            else:
                embeddings = None
                break

        if embeddings is None or len(embeddings) != 4:
            continue

        embeddings = np.array(embeddings)

        # Test each method
        for method_name, normalize in methods:
            key = f"{method_name}{'_norm' if normalize else ''}"

            try:
                predicted = find_outlier(
                    embeddings, method=method_name, normalize=normalize
                )

                method_results[key]["total"] += 1
                if predicted == true_outlier:
                    method_results[key]["correct"] += 1

            except Exception as e:
                LOGGER.exception("Outlier detection failed with %s: %s", key, e)

    # Calculate accuracies and display results
    LOGGER.info("Outlier detection method comparison for %s", model)

    results_data = []
    for key, data in method_results.items():
        if data["total"] > 0:
            accuracy = (data["correct"] / data["total"]) * 100
            results_data.append(
                {
                    "Method": data["method"],
                    "Normalized": "Yes" if data["normalize"] else "No",
                    "Correct": data["correct"],
                    "Total": data["total"],
                    "Accuracy (%)": accuracy,
                }
            )

    if not results_data:
        LOGGER.error("No results; the embedding cache for %s may be incomplete", model)
        return

    df = pd.DataFrame(results_data)
    df = df.sort_values(by="Accuracy (%)", ascending=False).reset_index(drop=True)

    LOGGER.info("Method comparison results:\n%s", df.to_string(index=False))

    # Highlight best method
    if len(df) > 0:
        best = df.iloc[0]
        LOGGER.info(
            "Best method: %s (normalized=%s), accuracy=%.2f%%, correct=%d/%d",
            best["Method"],
            best["Normalized"],
            best["Accuracy (%)"],
            best["Correct"],
            best["Total"],
        )

    return df


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    script_dir = Path(__file__).resolve().parent
    dataset_path = script_dir.parent / "preprocess-data" / "benchmark_dataset.json"
    cache_dir = script_dir / "openrouter_cache"

    # Default model to test - change this to the model you've cached
    default_model = "openai/text-embedding-3-small"

    # Allow model to be specified via command line
    selected_model = sys.argv[1] if len(sys.argv) > 1 else default_model

    compare_methods_on_dataset(dataset_path, cache_dir, selected_model)
