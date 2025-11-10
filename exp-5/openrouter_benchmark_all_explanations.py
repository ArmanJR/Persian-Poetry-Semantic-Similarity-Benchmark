"""
OpenRouter Embedding Models Benchmark - ALL Explanation Sources

This script benchmarks embedding models across ALL LLM explanation sources,
creating a 2D matrix: Embedding Models × Explanation Source LLMs

This allows us to determine:
1. Which embedding models perform best overall
2. Which LLM explanations are most effective for embeddings
3. Which combinations work best together
"""

import json
import os
import time
import numpy as np
import requests
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Set
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from dotenv import load_dotenv
from outlier_detection_methods import find_outlier as find_outlier_advanced
import signal

# Load environment variables
load_dotenv()

# Configuration
OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
BENCHMARK_DATA_PATH = '../exp-4/benchmark_dataset_with_explanations.json'
CACHE_DIR = 'openrouter_cache'
RESULTS_CSV_PATH = 'openrouter_results_all_explanations.csv'
SUMMARY_CSV_PATH = 'summary_by_embedding_model.csv'
SUMMARY_BY_LLM_PATH = 'summary_by_explanation_llm.csv'

# Outlier detection configuration
OUTLIER_METHOD = 'pairwise_avg'
NORMALIZE_EMBEDDINGS = False

# Embedding models to test
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
RATE_LIMIT_DELAY = 0.1
MAX_RETRIES = 5
RETRY_DELAY = 2

# Timeout configuration
EMBEDDING_TIMEOUT = 45  # seconds per individual embedding request
QUESTION_TIMEOUT = 250  # seconds per question (4 embeddings)


class TimeoutException(Exception):
    """Exception raised when operation times out."""
    pass


def timeout_handler(signum, frame):
    """Signal handler for timeout."""
    raise TimeoutException("Operation timed out")


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
        import hashlib
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
        return f"{model.replace('/', '_')}_{text_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> Optional[List[float]]:
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
        cache_path = self._get_cache_path(cache_key)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({'embedding': embedding}, f)
        except Exception as e:
            print(f"Warning: Failed to save to cache: {e}")

    def _rate_limit(self):
        current_time = time.time()
        time_since_last_request = current_time - self.last_request_time
        if time_since_last_request < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - time_since_last_request)
        self.last_request_time = time.time()

    def get_embedding(self, text: str, model: str, use_cache: bool = True) -> Optional[List[float]]:
        if use_cache:
            cache_key = self._get_cache_key(model, text)
            cached_embedding = self._load_from_cache(cache_key)
            if cached_embedding is not None:
                return cached_embedding

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                headers = {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                }
                payload = {"input": text, "model": model}
                response = requests.post(OPENROUTER_API_URL, headers=headers, json=payload, timeout=EMBEDDING_TIMEOUT)

                if response.status_code == 200:
                    data = response.json()
                    if 'data' in data and len(data['data']) > 0:
                        embedding = data['data'][0]['embedding']
                        if isinstance(embedding, str):
                            import base64
                            decoded = base64.b64decode(embedding)
                            embedding = np.frombuffer(decoded, dtype=np.float32).tolist()
                        if use_cache:
                            self._save_to_cache(cache_key, embedding)
                        return embedding
                    else:
                        print(f"Warning: Unexpected response format for model {model}")
                        return None
                elif response.status_code == 429:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"Rate limited. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                    continue
                elif response.status_code == 402:
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
        embeddings = []
        embedding_dim = None
        for idx, text in enumerate(texts):
            print(f".", end='', flush=True)
            embedding = self.get_embedding(text, model, use_cache)
            if embedding is None:
                print(f"\n  Failed to get embedding for option {idx + 1}", flush=True)
                return None
            if embedding_dim is None:
                embedding_dim = len(embedding)
            elif len(embedding) != embedding_dim:
                print(f"\n  Error: Inconsistent embedding dimensions ({len(embedding)} vs {embedding_dim})", flush=True)
                return None
            embeddings.append(embedding)
        return np.array(embeddings)


def discover_explanation_models(dataset: List[Dict]) -> Set[str]:
    """Discover all available LLM explanation models in the dataset."""
    llm_models = set()
    for item in dataset:
        if 'explanations' in item:
            for exp_item in item['explanations']:
                llm_explanations = exp_item.get('llm_explanations', {})
                llm_models.update(llm_explanations.keys())
    return llm_models


def extract_explanations_from_item(item: Dict, explanation_model: str) -> Optional[List[str]]:
    """Extract explanations for all 4 options from a question item."""
    if 'explanations' not in item:
        return None
    explanations_list = []
    for option_idx in range(4):
        option_explanation = None
        for exp_item in item['explanations']:
            if exp_item.get('option_index') == option_idx:
                llm_explanations = exp_item.get('llm_explanations', {})
                if explanation_model in llm_explanations:
                    model_data = llm_explanations[explanation_model]
                    # Check if model_data is a dict before calling .get()
                    if isinstance(model_data, dict):
                        option_explanation = model_data.get('explanation')
                break
        if option_explanation is None:
            return None
        explanations_list.append(option_explanation)
    if len(explanations_list) != 4:
        return None
    return explanations_list


def load_benchmark_dataset(path: str) -> List[Dict]:
    """Load the benchmark dataset from JSON file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Benchmark data file not found at '{path}'")
    with open(path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    if not dataset:
        raise ValueError("Loaded dataset is empty")
    return dataset


def benchmark_combination(
    client: OpenRouterEmbeddingClient,
    embedding_model: str,
    explanation_model: str,
    dataset: List[Dict],
    verbose: bool = False
) -> Tuple[float, int, int]:
    """Benchmark a single combination of embedding model + explanation LLM."""
    correct_predictions = 0
    prediction_failures = 0
    total_questions = len(dataset)

    for i, item in enumerate(dataset):
        if verbose:
            question_id = item.get('id', f'index_{i}')
            print(f"  Q{i + 1}/{total_questions} (ID: {question_id})...", end='', flush=True)

        if 'correct_answer_index' not in item:
            prediction_failures += 1
            if verbose:
                print(" SKIP")
            continue

        explanations = extract_explanations_from_item(item, explanation_model)
        if explanations is None:
            prediction_failures += 1
            if verbose:
                print(" SKIP")
            continue

        true_outlier_index = item['correct_answer_index']

        # Set timeout for this question
        try:
            # Set alarm for question timeout
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(QUESTION_TIMEOUT)

            try:
                option_embeddings = client.get_embeddings_batch(explanations, embedding_model)
                if option_embeddings is None or option_embeddings.shape[0] != 4:
                    prediction_failures += 1
                    if verbose:
                        print(" FAIL")
                    signal.alarm(0)  # Cancel alarm
                    continue
            finally:
                signal.alarm(0)  # Cancel alarm

        except TimeoutException:
            prediction_failures += 1
            if verbose:
                print(" TIMEOUT")
            continue
        except Exception as e:
            prediction_failures += 1
            if verbose:
                print(f" ERR")
            continue

        try:
            predicted_outlier_index = find_outlier_advanced(
                option_embeddings, method=OUTLIER_METHOD, normalize=NORMALIZE_EMBEDDINGS
            )
        except Exception as e:
            prediction_failures += 1
            if verbose:
                print(" FAIL")
            continue

        if predicted_outlier_index == -1:
            prediction_failures += 1
            if verbose:
                print(" FAIL")
        elif predicted_outlier_index == true_outlier_index:
            correct_predictions += 1
            if verbose:
                print(" ✓")
        else:
            if verbose:
                print(" ✗")

    effective_total = total_questions - prediction_failures
    accuracy = (correct_predictions / effective_total * 100) if effective_total > 0 else 0.0
    return accuracy, correct_predictions, effective_total


def main():
    """Main benchmark execution function."""
    print("="*80)
    print("OpenRouter Embedding Models Benchmark - ALL EXPLANATION SOURCES")
    print("2D Matrix: Embedding Models × Explanation LLMs")
    print("="*80)

    if not OPENROUTER_API_KEY:
        print("\nError: OPENROUTER_API_KEY not found!")
        return

    # Load dataset
    print(f"\nLoading benchmark dataset from: {BENCHMARK_DATA_PATH}")
    try:
        dataset = load_benchmark_dataset(BENCHMARK_DATA_PATH)
        print(f"✓ Successfully loaded {len(dataset)} questions")
    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return

    # Discover available explanation models
    print("\nDiscovering explanation models...")
    explanation_models = sorted(list(discover_explanation_models(dataset)))
    print(f"✓ Found {len(explanation_models)} explanation models:")
    for model in explanation_models:
        print(f"  - {model}")

    # Initialize client
    print(f"\nInitializing OpenRouter client...")
    client = OpenRouterEmbeddingClient(OPENROUTER_API_KEY, CACHE_DIR)

    # Run benchmarks for all combinations
    results = []
    total_combinations = len(EMBEDDING_MODELS) * len(explanation_models)
    current_combination = 0

    print(f"\n{'='*80}")
    print(f"Starting benchmark: {len(EMBEDDING_MODELS)} embedding models × {len(explanation_models)} explanation models")
    print(f"Total combinations: {total_combinations}")
    print(f"{'='*80}\n")

    try:
        for emb_idx, embedding_model in enumerate(EMBEDDING_MODELS, 1):
            for exp_idx, explanation_model in enumerate(explanation_models, 1):
                current_combination += 1
                print(f"\n[{current_combination}/{total_combinations}] Testing:")
                print(f"  Embedding: {embedding_model}")
                print(f"  Explanation: {explanation_model}")

                try:
                    start_time = time.time()
                    accuracy, correct, total = benchmark_combination(
                        client, embedding_model, explanation_model, dataset, verbose=False
                    )
                    elapsed_time = time.time() - start_time
                    results.append({
                        'Embedding Model': embedding_model,
                        'Explanation Model': explanation_model,
                        'Accuracy (%)': accuracy,
                        'Correct': correct,
                        'Total': total
                    })
                    print(f"  Result: {accuracy:.2f}% ({correct}/{total}) [{elapsed_time:.1f}s]")
                except KeyboardInterrupt:
                    print("\n\n⚠️  Interrupted by user!")
                    raise
                except Exception as e:
                    print(f"  Error: {e}")
                    results.append({
                        'Embedding Model': embedding_model,
                        'Explanation Model': explanation_model,
                        'Accuracy (%)': 0.0,
                        'Correct': 0,
                        'Total': 0
                    })

    except KeyboardInterrupt:
        print("\n\n" + "="*80)
        print("Benchmark interrupted by user")
        print("Saving partial results...")
        print("="*80)

    # Save detailed results
    print("\n" + "="*80)
    print("Saving Results")
    print("="*80)

    if not results:
        print("No results generated.")
        return

    # Create detailed results DataFrame
    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8')
    print(f"\n✓ Detailed results saved to: {RESULTS_CSV_PATH}")

    # Create summary by embedding model
    summary_by_embedding = results_df.groupby('Embedding Model').agg({
        'Accuracy (%)': ['mean', 'std', 'min', 'max'],
        'Correct': 'sum',
        'Total': 'sum'
    }).reset_index()
    summary_by_embedding.columns = ['Embedding Model', 'Mean Accuracy', 'Std Accuracy', 'Min Accuracy', 'Max Accuracy', 'Total Correct', 'Total Questions']
    summary_by_embedding = summary_by_embedding.sort_values('Mean Accuracy', ascending=False)
    summary_by_embedding.to_csv(SUMMARY_CSV_PATH, index=False, encoding='utf-8')
    print(f"✓ Summary by embedding model saved to: {SUMMARY_CSV_PATH}")

    # Create summary by explanation model
    summary_by_llm = results_df.groupby('Explanation Model').agg({
        'Accuracy (%)': ['mean', 'std', 'min', 'max'],
        'Correct': 'sum',
        'Total': 'sum'
    }).reset_index()
    summary_by_llm.columns = ['Explanation Model', 'Mean Accuracy', 'Std Accuracy', 'Min Accuracy', 'Max Accuracy', 'Total Correct', 'Total Questions']
    summary_by_llm = summary_by_llm.sort_values('Mean Accuracy', ascending=False)
    summary_by_llm.to_csv(SUMMARY_BY_LLM_PATH, index=False, encoding='utf-8')
    print(f"✓ Summary by explanation model saved to: {SUMMARY_BY_LLM_PATH}")

    # Display summaries
    print("\n" + "="*80)
    print("SUMMARY: Top Embedding Models (by mean accuracy across all explanations)")
    print("="*80)
    print(summary_by_embedding.head(5).to_string(index=False))

    print("\n" + "="*80)
    print("SUMMARY: Top Explanation Models (by mean accuracy across all embeddings)")
    print("="*80)
    print(summary_by_llm.head(5).to_string(index=False))

    # Find best combination
    best_idx = results_df['Accuracy (%)'].idxmax()
    best_result = results_df.loc[best_idx]
    print("\n" + "="*80)
    print("BEST COMBINATION")
    print("="*80)
    print(f"Embedding Model: {best_result['Embedding Model']}")
    print(f"Explanation Model: {best_result['Explanation Model']}")
    print(f"Accuracy: {best_result['Accuracy (%)']:.2f}%")
    print(f"Score: {best_result['Correct']}/{best_result['Total']}")

    print("\n" + "="*80)
    print("Benchmark Complete!")
    print("="*80)


if __name__ == "__main__":
    main()
