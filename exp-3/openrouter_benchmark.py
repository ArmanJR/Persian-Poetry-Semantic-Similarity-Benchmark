"""
OpenRouter Embedding Models Benchmark for Persian Poetry Semantic Similarity

This script benchmarks various embedding models available through OpenRouter API
on their ability to identify semantic outliers in Persian poetry couplets.
"""

import json
import os
import time
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Literal
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dotenv import load_dotenv
from outlier_detection_methods import find_outlier as find_outlier_advanced

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
BENCHMARK_DATA_PATH = '../preprocess-data/benchmark_dataset.json'
CACHE_DIR = 'openrouter_cache'
RESULTS_CSV_PATH = 'openrouter_results.csv'

# Outlier detection configuration
OUTLIER_METHOD = 'pairwise_avg'  # Options: 'centroid', 'pairwise_avg', 'max_distance_sum', 'euclidean', 'isolation'
NORMALIZE_EMBEDDINGS = False  # Whether to L2-normalize embeddings before comparison

# Embedding models to test (OpenRouter model IDs)
EMBEDDING_MODELS = [
    'qwen/qwen3-embedding-0.6b',
    'mistralai/mistral-embed-2312',
    'google/gemini-embedding-001',
    'openai/text-embedding-ada-002',
    'mistralai/codestral-embed-2505',
    'openai/text-embedding-3-large',
    'openai/text-embedding-3-small',
    'qwen/qwen3-embedding-8b',
    'qwen/qwen3-embedding-4b'
]

# Rate limiting configuration
RATE_LIMIT_DELAY = 0.1  # seconds between requests
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds


class OpenRouterEmbeddingClient:
    """Client for OpenRouter Embeddings API with caching and rate limiting."""

    def __init__(self, api_key: str, cache_dir: str = CACHE_DIR):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not found. Please set it in .env file")

        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.last_request_time = 0

    def _get_cache_key(self, model: str, text: str) -> str:
        """Generate a cache key for the given model and text."""
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model.replace('/', '_')}_{text_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        """Get the cache file path for a cache key."""
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
        """Load embedding from cache if available."""
        cache_path = self._get_cache_path(cache_key)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data['embedding']
            except Exception as e:
                print(f"Warning: Failed to load from cache: {e}")
        return None

    def _save_to_cache(self, cache_key: str, embedding: List[float]):
        """Save embedding to cache."""
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'embedding': embedding}, f)
        except Exception as e:
            print(f"Warning: Failed to save to cache: {e}")

    def _rate_limit(self):
        """Implement rate limiting between requests."""
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last_request)
        self.last_request_time = time.time()

    def get_embedding(self, text: str, model: str, use_cache: bool = True) -> Optional[List[float]]:
        """
        Get embedding for a text using OpenRouter API.

        Args:
            text: The text to embed
            model: The model ID to use
            use_cache: Whether to use caching

        Returns:
            List of floats representing the embedding, or None if failed
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(model, text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        # Make API request with retries
        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()

                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }

                payload = {
                    "input": text,
                    "model": model
                }

                response = requests.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=60  # Increased timeout to 60 seconds
                )

                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        embedding = data['data'][0]['embedding']

                        # Handle base64 encoded embeddings
                        if isinstance(embedding, str):
                            import base64
                            decoded = base64.b64decode(embedding)
                            embedding = np.frombuffer(decoded, dtype=np.float32).tolist()

                        # Save to cache
                        if use_cache:
                            self._save_to_cache(cache_key, embedding)

                        return embedding
                    else:
                        print(f"Warning: Unexpected response format for model {model}")
                        return None

                elif response.status_code == 429:  # Rate limit
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue

                elif response.status_code == 402:  # Payment required
                    print(f"Error: Insufficient credits for model {model}")
                    return None

                else:
                    print(f"Error: API returned status {response.status_code}: {response.text}")
                    if attempt < MAX_RETRIES - 1:
                        time.sleep(RETRY_DELAY)
                        continue
                    return None

            except requests.exceptions.Timeout:
                print(f"Timeout on attempt {attempt + 1}/{MAX_RETRIES}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

            except Exception as e:
                print(f"Error getting embedding: {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
                    continue
                return None

        return None

    def get_embeddings_batch(self, texts: List[str], model: str, use_cache: bool = True) -> Optional[np.ndarray]:
        """
        Get embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            model: The model ID to use
            use_cache: Whether to use caching

        Returns:
            NumPy array of embeddings, or None if any failed
        """
        embeddings = []
        embedding_dim = None

        for idx, text in enumerate(texts):
            print(f".", end='', flush=True)  # Progress indicator
            embedding = self.get_embedding(text, model, use_cache)
            if embedding is None:
                print(f"\n  Failed to get embedding for option {idx + 1}", flush=True)
                return None

            # Validate dimension consistency
            if embedding_dim is None:
                embedding_dim = len(embedding)
            elif len(embedding) != embedding_dim:
                print(f"\n  Error: Inconsistent embedding dimensions ({len(embedding)} vs {embedding_dim})", flush=True)
                return None

            embeddings.append(embedding)

        return np.array(embeddings)


def find_outlier_index(embeddings: np.ndarray) -> int:
    """
    Identifies the index of the outlier embedding based on lowest similarity
    to the centroid of the other embeddings.

    Args:
        embeddings: A numpy array of 4 embedding vectors.

    Returns:
        The index (0-3) of the predicted outlier embedding, or -1 if prediction fails.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    if len(embeddings) != 4 or embeddings.ndim != 2:
        print(f"Warning: Invalid input shape {embeddings.shape}")
        return -1

    similarity_scores = []

    for i in range(4):
        # Get all other embeddings
        others = np.delete(embeddings, i, axis=0)
        if others.size == 0:
            similarity_scores.append(-np.inf)
            continue

        # Calculate centroid of others
        centroid = np.mean(others, axis=0, keepdims=True)
        current = embeddings[i].reshape(1, -1)

        # Calculate cosine similarity
        similarity = cosine_similarity(current, centroid)[0][0]
        similarity_scores.append(similarity)

    # Check for valid similarities
    if all(score == -np.inf for score in similarity_scores):
        print("Warning: No valid similarities calculated")
        return -1

    # Return index with minimum similarity (most dissimilar = outlier)
    return int(np.argmin(similarity_scores))


def load_benchmark_dataset(path: str) -> List[Dict]:
    """Load the benchmark dataset from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark data file not found at '{path}'")

    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    if not dataset:
        raise ValueError("Loaded dataset is empty")

    return dataset


def benchmark_model(
    client: OpenRouterEmbeddingClient,
    model: str,
    dataset: List[Dict],
    verbose: bool = True
) -> Tuple[float, int, int, Optional[int]]:
    """
    Benchmark a single model on the dataset.

    Args:
        client: OpenRouter client instance
        model: Model ID to test
        dataset: Benchmark dataset
        verbose: Whether to print progress

    Returns:
        Tuple of (accuracy, correct_predictions, total_valid, embedding_dimension)
    """
    if verbose:
        print(f"\n{'='*60}")
        print(f"Testing Model: {model}")
        print(f"Method: {OUTLIER_METHOD}" + (f" (normalized)" if NORMALIZE_EMBEDDINGS else ""))
        print(f"{'='*60}")

    correct_predictions = 0
    prediction_failures = 0
    total_questions = len(dataset)
    embedding_dimension = None  # Track embedding dimension

    for i, item in enumerate(dataset):
        question_id = item.get('id', f'index_{i}')

        if verbose:
            print(f"Processing question {i + 1}/{total_questions} (ID: {question_id})...", end='', flush=True)

        # Validate question data
        if 'options' not in item or len(item['options']) != 4:
            prediction_failures += 1
            if verbose:
                print(" SKIPPED (invalid options)")
            continue
        if 'correct_answer_index' not in item:
            prediction_failures += 1
            if verbose:
                print(" SKIPPED (missing answer)")
            continue

        options_text = item['options']
        true_outlier_index = item['correct_answer_index']

        # Get embeddings for all options
        try:
            option_embeddings = client.get_embeddings_batch(options_text, model)

            if option_embeddings is None:
                prediction_failures += 1
                if verbose:
                    print(f" FAILED (could not get embeddings)")
                continue

            # Validate embedding shape
            if option_embeddings.shape[0] != 4:
                prediction_failures += 1
                if verbose:
                    print(f" FAILED (invalid shape)")
                continue

            # Track embedding dimension (log on first successful question)
            if embedding_dimension is None and option_embeddings.shape[1] > 0:
                embedding_dimension = option_embeddings.shape[1]
                if verbose:
                    print(f" [dim={embedding_dimension}]", end='', flush=True)

        except Exception as e:
            if verbose:
                print(f" ERROR: {str(e)[:100]}")
            prediction_failures += 1
            continue

        # Predict the outlier using configured method
        try:
            predicted_outlier_index = find_outlier_advanced(
                option_embeddings,
                method=OUTLIER_METHOD,
                normalize=NORMALIZE_EMBEDDINGS
            )
        except Exception as e:
            if verbose:
                print(f" ERROR in outlier detection: {str(e)[:50]}")
            prediction_failures += 1
            continue

        if predicted_outlier_index == -1:
            prediction_failures += 1
            if verbose:
                print(" FAILED (outlier detection)")
        elif predicted_outlier_index == true_outlier_index:
            correct_predictions += 1
            if verbose:
                print(" ✓ CORRECT")
        else:
            if verbose:
                print(" ✗ INCORRECT")

    # Calculate accuracy
    effective_total = total_questions - prediction_failures
    accuracy = (correct_predictions / effective_total * 100) if effective_total > 0 else 0.0

    if verbose:
        print(f"\nResults for {model}:")
        if embedding_dimension:
            print(f"  Embedding dimension: {embedding_dimension}")
        print(f"  Correct: {correct_predictions}/{effective_total}")
        print(f"  Accuracy: {accuracy:.2f}%")
        if prediction_failures > 0:
            print(f"  Failed/Skipped: {prediction_failures}")

    return accuracy, correct_predictions, effective_total, embedding_dimension


def main():
    """Main benchmark execution function."""
    print("="*80)
    print("OpenRouter Embedding Models Benchmark")
    print("Persian Poetry Semantic Similarity")
    print("="*80)
    print("\nTip: Press Ctrl+C to interrupt and save partial results")

    # Validate API key
    if not OPENROUTER_API_KEY:
        print("\nError: OPENROUTER_API_KEY not found!")
        print("Please create a .env file with your OpenRouter API key:")
        print("  OPENROUTER_API_KEY=your_api_key_here")
        print("\nYou can get an API key from: https://openrouter.ai/keys")
        return

    # Load dataset
    print(f"\nLoading benchmark dataset from: {BENCHMARK_DATA_PATH}")
    try:
        dataset = load_benchmark_dataset(BENCHMARK_DATA_PATH)
        print(f"Successfully loaded {len(dataset)} questions")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Initialize client
    print(f"\nInitializing OpenRouter client...")
    print(f"Cache directory: {CACHE_DIR}")
    client = OpenRouterEmbeddingClient(OPENROUTER_API_KEY, CACHE_DIR)

    # Run benchmarks
    results = {}

    print(f"\n{'='*80}")
    print(f"Starting benchmark for {len(EMBEDDING_MODELS)} models")
    print(f"{'='*80}")

    try:
        for idx, model in enumerate(EMBEDDING_MODELS, 1):
            print(f"\n[{idx}/{len(EMBEDDING_MODELS)}] Testing: {model}")

            try:
                accuracy, correct, total, dimension = benchmark_model(client, model, dataset, verbose=True)
                results[model] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'dimension': dimension
                }
            except KeyboardInterrupt:
                print("\n\n⚠️  Interrupted by user!")
                raise
            except Exception as e:
                print(f"Error benchmarking {model}: {e}")
                results[model] = {
                    'accuracy': 0.0,
                    'correct': 0,
                    'total': 0,
                    'dimension': None
                }
    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Benchmark interrupted by user")
        print("Saving partial results...")
        print("="*80)

    # Process and save results
    print("\n" + "="*80)
    print("Benchmark Complete - Results Summary")
    print("="*80)

    if not results:
        print("No results generated.")
        return

    # Create DataFrame
    results_data = []
    for model, data in results.items():
        results_data.append({
            'Model Name': model,
            'Accuracy (%)': data['accuracy'],
            'Correct': data['correct'],
            'Total': data['total'],
            'Embedding Dim': data.get('dimension', 'N/A')
        })

    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

    # Print results table
    print("\n" + results_df.to_string(index=False))

    # Save to CSV
    try:
        results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8')
        print(f"\nResults saved to: {RESULTS_CSV_PATH}")
    except Exception as e:
        print(f"Error saving results to CSV: {e}")

    # Print cost information
    print("\n" + "="*80)
    print("Note: Check your OpenRouter dashboard for cost information:")
    print("https://openrouter.ai/activity")
    print("="*80)


if __name__ == "__main__":
    main()
