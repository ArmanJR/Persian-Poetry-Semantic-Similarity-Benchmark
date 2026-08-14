"""Benchmark OpenRouter embedding models on Persian poetry similarity."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from outlier_detection_methods import find_outlier

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
load_dotenv(PROJECT_ROOT / ".env")

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/embeddings"
BENCHMARK_DATA_PATH = PROJECT_ROOT / "preprocess-data" / "benchmark_dataset.json"
CACHE_DIR = SCRIPT_DIR / "openrouter_cache"
RESULTS_CSV_PATH = SCRIPT_DIR / "openrouter_results.csv"

OUTLIER_METHOD = "pairwise_avg"
NORMALIZE_EMBEDDINGS = False

EXISTING_EMBEDDING_MODELS = (
    "qwen/qwen3-embedding-0.6b",
    "mistralai/mistral-embed-2312",
    "google/gemini-embedding-001",
    "openai/text-embedding-ada-002",
    "mistralai/codestral-embed-2505",
    "openai/text-embedding-3-large",
    "openai/text-embedding-3-small",
    "qwen/qwen3-embedding-8b",
    "qwen/qwen3-embedding-4b",
)

NEW_EMBEDDING_MODELS = (
    "voyageai/voyage-4-lite",
    "voyageai/voyage-4",
    "voyageai/voyage-4-large",
    "nvidia/nemotron-3-embed-1b:free",
    "google/gemini-embedding-2",
    "perplexity/pplx-embed-v1-4b",
    "perplexity/pplx-embed-v1-0.6b",
    "sentence-transformers/paraphrase-minilm-l6-v2",
    "sentence-transformers/all-minilm-l12-v2",
)

EMBEDDING_MODELS = EXISTING_EMBEDDING_MODELS + NEW_EMBEDDING_MODELS

RATE_LIMIT_DELAY = 0.1
MAX_RETRIES = 5
RETRY_DELAY = 2.0
REQUEST_TIMEOUT = 60
PREFETCH_BATCH_SIZE = 32
RETRYABLE_STATUS_CODES = frozenset({408, 409, 429, 500, 502, 503, 524, 529})
RESULT_COLUMNS = (
    "Model Name",
    "Accuracy (%)",
    "Correct",
    "Total",
    "Embedding Dim",
)


class OpenRouterEmbeddingClient:
    """OpenRouter embeddings client with local caching and retry handling."""

    def __init__(
        self,
        api_key: str,
        cache_dir: Path = CACHE_DIR,
        session: requests.Session | None = None,
    ) -> None:
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY is required")

        self.api_key = api_key
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.last_request_time = 0.0
        self.unavailable_models: set[str] = set()

    @staticmethod
    def _get_cache_key(model: str, text: str) -> str:
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        model_hash = hashlib.sha256(model.encode("utf-8")).hexdigest()[:16]
        return f"{model_hash}_{text_hash}"

    def _get_cache_path(self, cache_key: str) -> Path:
        return self.cache_dir / f"{cache_key}.json"

    def _load_from_cache(self, cache_key: str) -> list[float] | None:
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return None

        try:
            with cache_path.open("r", encoding="utf-8") as cache_file:
                data = json.load(cache_file)
            embedding = data["embedding"]
            if not isinstance(embedding, list) or not embedding:
                raise ValueError("cached embedding is not a non-empty list")
            return embedding
        except (
            OSError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            LOGGER.warning("Could not read embedding cache %s: %s", cache_path, error)
            return None

    def _save_to_cache(self, cache_key: str, embedding: list[float]) -> None:
        cache_path = self._get_cache_path(cache_key)
        try:
            with cache_path.open("w", encoding="utf-8") as cache_file:
                json.dump({"embedding": embedding}, cache_file)
        except OSError as error:
            LOGGER.warning("Could not write embedding cache %s: %s", cache_path, error)

    def _rate_limit(self) -> None:
        elapsed = time.monotonic() - self.last_request_time
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        self.last_request_time = time.monotonic()

    @staticmethod
    def _decode_embedding(value: Any) -> list[float]:
        if isinstance(value, str):
            decoded = base64.b64decode(value, validate=True)
            return np.frombuffer(decoded, dtype=np.float32).tolist()
        if not isinstance(value, list) or not value:
            raise ValueError("embedding is not a non-empty list or base64 string")
        return value

    @staticmethod
    def _retry_wait_seconds(response: requests.Response, attempt: int) -> float:
        retry_after = response.headers.get("Retry-After")
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), 60.0)
            except ValueError:
                LOGGER.debug("Ignoring non-numeric Retry-After header: %s", retry_after)
        return min(RETRY_DELAY * (2**attempt), 60.0)

    def _request_embeddings(
        self, texts: Sequence[str], model: str
    ) -> list[list[float]] | None:
        if model in self.unavailable_models:
            LOGGER.debug("Skipping request for unavailable model %s", model)
            return None

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "input": list(texts),
            "model": model,
            "encoding_format": "float",
        }

        for attempt in range(MAX_RETRIES):
            try:
                self._rate_limit()
                LOGGER.debug(
                    "Requesting %d embedding(s) from %s (attempt %d/%d)",
                    len(texts),
                    model,
                    attempt + 1,
                    MAX_RETRIES,
                )
                response = self.session.post(
                    OPENROUTER_API_URL,
                    headers=headers,
                    json=payload,
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.RequestException as error:
                LOGGER.warning(
                    "Embedding request for %s failed on attempt %d/%d: %s",
                    model,
                    attempt + 1,
                    MAX_RETRIES,
                    error,
                )
                if attempt == MAX_RETRIES - 1:
                    return None
                wait_seconds = min(RETRY_DELAY * (2**attempt), 60.0)
                LOGGER.info("Retrying %s in %.1f seconds", model, wait_seconds)
                time.sleep(wait_seconds)
                continue

            if response.status_code == 200:
                try:
                    response_data = response.json()["data"]
                    if len(response_data) != len(texts):
                        raise ValueError(
                            f"expected {len(texts)} embeddings, received {len(response_data)}"
                        )
                    ordered_data = sorted(
                        enumerate(response_data),
                        key=lambda pair: pair[1].get("index", pair[0]),
                    )
                    embeddings = [
                        self._decode_embedding(item["embedding"])
                        for _, item in ordered_data
                    ]
                    if len({len(embedding) for embedding in embeddings}) != 1:
                        raise ValueError(
                            "response contains inconsistent embedding dimensions"
                        )
                    return embeddings
                except (
                    KeyError,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ) as error:
                    LOGGER.error("Invalid embedding response from %s: %s", model, error)
                    return None

            response_excerpt = response.text.replace("\n", " ")[:1_000]
            if response.status_code in RETRYABLE_STATUS_CODES:
                if attempt == MAX_RETRIES - 1:
                    LOGGER.error(
                        "OpenRouter returned status %d for %s after %d attempts. "
                        "Response: %s",
                        response.status_code,
                        model,
                        MAX_RETRIES,
                        response_excerpt,
                    )
                    return None
                wait_seconds = self._retry_wait_seconds(response, attempt)
                LOGGER.warning(
                    "OpenRouter returned retryable status %d for %s on attempt "
                    "%d/%d; retrying in %.1f seconds. Response: %s",
                    response.status_code,
                    model,
                    attempt + 1,
                    MAX_RETRIES,
                    wait_seconds,
                    response_excerpt,
                )
                time.sleep(wait_seconds)
                continue

            LOGGER.error(
                "OpenRouter rejected model %s with status %d. Response: %s",
                model,
                response.status_code,
                response_excerpt,
            )
            self.unavailable_models.add(model)
            return None

        return None

    def get_embeddings_batch(
        self,
        texts: Sequence[str],
        model: str,
        use_cache: bool = True,
    ) -> np.ndarray | None:
        """Return embeddings in input order, requesting all cache misses together."""
        if not texts:
            LOGGER.error("Cannot request an empty embedding batch for %s", model)
            return None

        cache_keys = [self._get_cache_key(model, text) for text in texts]
        embeddings: list[list[float] | None] = [None] * len(texts)
        missing_indices: list[int] = []

        for index, cache_key in enumerate(cache_keys):
            cached_embedding = self._load_from_cache(cache_key) if use_cache else None
            if cached_embedding is None:
                missing_indices.append(index)
            else:
                embeddings[index] = cached_embedding

        if missing_indices:
            missing_texts = [texts[index] for index in missing_indices]
            requested_embeddings = self._request_embeddings(missing_texts, model)
            if requested_embeddings is None:
                return None

            for index, embedding in zip(
                missing_indices, requested_embeddings, strict=True
            ):
                embeddings[index] = embedding
                if use_cache:
                    self._save_to_cache(cache_keys[index], embedding)

        if any(embedding is None for embedding in embeddings):
            LOGGER.error("Failed to assemble a complete embedding batch for %s", model)
            return None

        embedding_dimensions = {
            len(embedding) for embedding in embeddings if embedding is not None
        }
        if len(embedding_dimensions) != 1:
            LOGGER.error(
                "Inconsistent cached embedding dimensions for %s: %s",
                model,
                sorted(embedding_dimensions),
            )
            return None

        embedding_array = np.asarray(embeddings, dtype=np.float64)
        if embedding_array.ndim != 2 or embedding_array.shape[0] != len(texts):
            LOGGER.error(
                "Invalid embedding array shape for %s: %s",
                model,
                embedding_array.shape,
            )
            return None
        return embedding_array

    def close(self) -> None:
        self.session.close()


def load_benchmark_dataset(path: Path) -> list[dict[str, Any]]:
    """Load and validate the benchmark dataset."""
    if not path.exists():
        raise FileNotFoundError(f"Benchmark data file not found at {path}")

    with path.open("r", encoding="utf-8") as data_file:
        dataset = json.load(data_file)

    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Benchmark dataset must be a non-empty list")

    invalid_items: list[Any] = []
    for index, item in enumerate(dataset):
        if not isinstance(item, dict):
            invalid_items.append(index)
            continue
        if (
            not isinstance(item.get("options"), list)
            or len(item["options"]) != 4
            or item.get("correct_answer_index") not in range(4)
        ):
            invalid_items.append(item.get("id", index))
    if invalid_items:
        raise ValueError(f"Invalid benchmark items: {invalid_items}")

    LOGGER.info("Loaded %d benchmark questions from %s", len(dataset), path)
    return dataset


def benchmark_model(
    client: OpenRouterEmbeddingClient,
    model: str,
    dataset: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Benchmark one model and return a CSV-compatible result row."""
    LOGGER.info(
        "Starting model %s with method=%s normalize=%s questions=%d",
        model,
        OUTLIER_METHOD,
        NORMALIZE_EMBEDDINGS,
        len(dataset),
    )
    prefetch_model_embeddings(client, model, dataset)
    correct_predictions = 0
    prediction_failures = 0
    embedding_dimension: int | None = None

    for question_number, item in enumerate(dataset, start=1):
        question_id = item.get("id", question_number - 1)
        option_embeddings = client.get_embeddings_batch(item["options"], model)
        if option_embeddings is None:
            prediction_failures += 1
            LOGGER.error(
                "Model %s failed question %d/%d (id=%s): embeddings unavailable",
                model,
                question_number,
                len(dataset),
                question_id,
            )
            continue

        if embedding_dimension is None:
            embedding_dimension = int(option_embeddings.shape[1])
            LOGGER.info("Model %s embedding dimension: %d", model, embedding_dimension)

        try:
            predicted_index = find_outlier(
                option_embeddings,
                method=OUTLIER_METHOD,
                normalize=NORMALIZE_EMBEDDINGS,
            )
        except (TypeError, ValueError) as error:
            prediction_failures += 1
            LOGGER.exception(
                "Outlier detection failed for model %s question id=%s: %s",
                model,
                question_id,
                error,
            )
            continue

        true_index = item["correct_answer_index"]
        is_correct = predicted_index == true_index
        correct_predictions += int(is_correct)
        LOGGER.info(
            "Model %s question %d/%d id=%s predicted=%d expected=%d result=%s",
            model,
            question_number,
            len(dataset),
            question_id,
            predicted_index,
            true_index,
            "correct" if is_correct else "incorrect",
        )

    effective_total = len(dataset) - prediction_failures
    accuracy = correct_predictions / effective_total * 100 if effective_total else 0.0
    LOGGER.info(
        "Completed model %s: accuracy=%.2f%% correct=%d total=%d failures=%d",
        model,
        accuracy,
        correct_predictions,
        effective_total,
        prediction_failures,
    )
    return {
        "Model Name": model,
        "Accuracy (%)": accuracy,
        "Correct": correct_predictions,
        "Total": effective_total,
        "Embedding Dim": embedding_dimension,
    }


def prefetch_model_embeddings(
    client: OpenRouterEmbeddingClient,
    model: str,
    dataset: Sequence[dict[str, Any]],
    batch_size: int = PREFETCH_BATCH_SIZE,
) -> None:
    """Populate the cache in bounded batches to minimize provider round trips."""
    if batch_size < 1:
        raise ValueError("Prefetch batch size must be positive")

    texts = [option for item in dataset for option in item["options"]]
    batch_count = (len(texts) + batch_size - 1) // batch_size
    LOGGER.info(
        "Prefetching %d texts for %s in %d batch(es) of at most %d",
        len(texts),
        model,
        batch_count,
        batch_size,
    )
    for batch_number, start in enumerate(range(0, len(texts), batch_size), start=1):
        batch = texts[start : start + batch_size]
        embeddings = client.get_embeddings_batch(batch, model)
        if embeddings is None:
            LOGGER.warning(
                "Prefetch batch %d/%d failed for %s; question-level requests will retry",
                batch_number,
                batch_count,
                model,
            )
        else:
            LOGGER.info(
                "Prefetch batch %d/%d complete for %s (%d texts)",
                batch_number,
                batch_count,
                model,
                len(batch),
            )


def save_results(
    new_results: Sequence[dict[str, Any]],
    results_path: Path,
    merge_existing: bool = True,
) -> pd.DataFrame:
    """Save results, replacing rows for rerun models while preserving other rows."""
    results_frame = pd.DataFrame(new_results, columns=RESULT_COLUMNS)
    if merge_existing and results_path.exists():
        existing_frame = pd.read_csv(results_path, encoding="utf-8")
        missing_columns = set(RESULT_COLUMNS) - set(existing_frame.columns)
        if missing_columns:
            raise ValueError(
                f"Existing results file is missing columns: {sorted(missing_columns)}"
            )
        results_frame = pd.concat(
            [existing_frame.loc[:, RESULT_COLUMNS], results_frame],
            ignore_index=True,
        )

    results_frame = (
        results_frame.drop_duplicates(subset=["Model Name"], keep="last")
        .sort_values(
            by=["Accuracy (%)", "Model Name"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    results_frame.to_csv(results_path, index=False, encoding="utf-8")
    LOGGER.info("Saved %d model results to %s", len(results_frame), results_path)
    return results_frame


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        default=list(EMBEDDING_MODELS),
        help="OpenRouter model IDs to benchmark (default: all configured models)",
    )
    parser.add_argument(
        "--replace-results",
        action="store_true",
        help="Replace the results CSV instead of merging rerun model rows",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    configure_logging(args.log_level)

    if not OPENROUTER_API_KEY:
        LOGGER.error(
            "OPENROUTER_API_KEY is missing; add it to %s",
            PROJECT_ROOT / ".env",
        )
        return 2

    try:
        dataset = load_benchmark_dataset(BENCHMARK_DATA_PATH)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        LOGGER.exception("Could not load benchmark dataset: %s", error)
        return 2

    LOGGER.info(
        "Starting Experiment #3 for %d model(s); cache=%s",
        len(args.models),
        CACHE_DIR,
    )
    client = OpenRouterEmbeddingClient(OPENROUTER_API_KEY, CACHE_DIR)
    results: list[dict[str, Any]] = []
    interrupted = False

    try:
        for model_number, model in enumerate(args.models, start=1):
            LOGGER.info(
                "Benchmarking model %d/%d: %s",
                model_number,
                len(args.models),
                model,
            )
            results.append(benchmark_model(client, model, dataset))
    except KeyboardInterrupt:
        interrupted = True
        LOGGER.warning("Benchmark interrupted; saving completed model results")
    finally:
        client.close()

    if not results:
        LOGGER.error("No completed model results to save")
        return 130 if interrupted else 1

    try:
        results_frame = save_results(
            results,
            RESULTS_CSV_PATH,
            merge_existing=not args.replace_results,
        )
    except (OSError, ValueError, pd.errors.ParserError) as error:
        LOGGER.exception("Could not save benchmark results: %s", error)
        return 1

    LOGGER.info("Benchmark results:\n%s", results_frame.to_string(index=False))
    incomplete_models = [
        result["Model Name"] for result in results if result["Total"] != len(dataset)
    ]
    if incomplete_models:
        LOGGER.error("Models with incomplete evaluations: %s", incomplete_models)
        return 1
    return 130 if interrupted else 0


if __name__ == "__main__":
    sys.exit(main())
