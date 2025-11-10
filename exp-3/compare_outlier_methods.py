"""
Compare different outlier detection methods on cached embeddings.

This script loads previously cached embeddings and tests all outlier detection
methods to see which performs best on Persian poetry data.
"""

import json
import numpy as np
from pathlib import Path
from outlier_detection_methods import find_outlier
import pandas as pd


def load_cached_embeddings(cache_dir: str, model: str) -> dict:
    """Load all cached embeddings for a specific model."""
    cache_path = Path(cache_dir)
    model_safe = model.replace('/', '_')

    embeddings_cache = {}

    if not cache_path.exists():
        return embeddings_cache

    for cache_file in cache_path.glob(f"{model_safe}_*.json"):
        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                if 'embedding' in data:
                    # Extract hash from filename
                    hash_key = cache_file.stem.replace(f"{model_safe}_", "")
                    embeddings_cache[hash_key] = data['embedding']
        except Exception as e:
            print(f"Warning: Failed to load {cache_file}: {e}")

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
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Load cached embeddings
    cache = load_cached_embeddings(cache_dir, model)

    if not cache:
        print(f"No cached embeddings found for model: {model}")
        print(f"Run openrouter_benchmark.py first to generate cache.")
        return

    print(f"Loaded {len(cache)} cached embeddings for {model}")

    # Methods to test
    methods = [
        ('centroid', False),
        ('centroid', True),
        ('pairwise_avg', False),
        ('pairwise_avg', True),
        ('max_distance_sum', False),
        ('max_distance_sum', True),
        ('euclidean', False),
        ('euclidean', True),
        ('isolation', False),
        ('isolation', True),
    ]

    method_results = {}

    for method_name, normalize in methods:
        key = f"{method_name}{'_norm' if normalize else ''}"
        method_results[key] = {
            'correct': 0,
            'total': 0,
            'method': method_name,
            'normalize': normalize
        }

    # Process each question
    import hashlib

    for item in dataset:
        if 'options' not in item or len(item['options']) != 4:
            continue
        if 'correct_answer_index' not in item:
            continue

        options = item['options']
        true_outlier = item['correct_answer_index']

        # Try to load embeddings for all 4 options
        embeddings = []
        for option_text in options:
            text_hash = hashlib.md5(option_text.encode('utf-8')).hexdigest()
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
                predicted = find_outlier(embeddings, method=method_name, normalize=normalize)

                method_results[key]['total'] += 1
                if predicted == true_outlier:
                    method_results[key]['correct'] += 1

            except Exception as e:
                print(f"Error with {key}: {e}")

    # Calculate accuracies and display results
    print("\n" + "="*80)
    print(f"Outlier Detection Method Comparison - Model: {model}")
    print("="*80)

    results_data = []
    for key, data in method_results.items():
        if data['total'] > 0:
            accuracy = (data['correct'] / data['total']) * 100
            results_data.append({
                'Method': data['method'],
                'Normalized': 'Yes' if data['normalize'] else 'No',
                'Correct': data['correct'],
                'Total': data['total'],
                'Accuracy (%)': accuracy
            })

    if not results_data:
        print("No results - cache may be incomplete. Run benchmark first.")
        return

    df = pd.DataFrame(results_data)
    df = df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

    print("\n" + df.to_string(index=False))

    # Highlight best method
    if len(df) > 0:
        best = df.iloc[0]
        print("\n" + "="*80)
        print(f"Best Method: {best['Method']} (Normalized: {best['Normalized']})")
        print(f"Accuracy: {best['Accuracy (%)']:.2f}%")
        print(f"Correct: {best['Correct']}/{best['Total']}")
        print("="*80)

    return df


if __name__ == "__main__":
    import sys

    DATASET_PATH = '../preprocess-data/benchmark_dataset.json'
    CACHE_DIR = 'openrouter_cache'

    # Default model to test - change this to the model you've cached
    DEFAULT_MODEL = 'openai/text-embedding-3-small'

    # Allow model to be specified via command line
    model = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_MODEL

    print("="*80)
    print("Outlier Detection Method Comparison")
    print("="*80)
    print(f"\nDataset: {DATASET_PATH}")
    print(f"Cache: {CACHE_DIR}")
    print(f"Model: {model}")
    print("\nThis compares different algorithms using cached embeddings.")
    print("="*80)

    compare_methods_on_dataset(DATASET_PATH, CACHE_DIR, model)
