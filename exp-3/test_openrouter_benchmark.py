"""Unit tests for the Experiment #3 OpenRouter benchmark runner."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import requests

import openrouter_benchmark as benchmark


class OpenRouterEmbeddingClientTests(unittest.TestCase):
    def test_batch_response_is_ordered_and_cached(self) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 200
        response.json.return_value = {
            "data": [
                {"index": 1, "embedding": [0.0, 1.0]},
                {"index": 0, "embedding": [1.0, 0.0]},
            ]
        }
        session = Mock(spec=requests.Session)
        session.post.return_value = response

        with tempfile.TemporaryDirectory() as temporary_directory:
            client = benchmark.OpenRouterEmbeddingClient(
                "test-key",
                Path(temporary_directory),
                session=session,
            )
            first_result = client.get_embeddings_batch(
                ["first", "second"], "test/model"
            )
            second_result = client.get_embeddings_batch(
                ["first", "second"], "test/model"
            )

            self.assertEqual(first_result.tolist(), [[1.0, 0.0], [0.0, 1.0]])
            self.assertEqual(second_result.tolist(), first_result.tolist())
            self.assertEqual(session.post.call_count, 1)
            self.assertEqual(len(list(Path(temporary_directory).glob("*.json"))), 2)

    def test_permanent_api_error_blocks_follow_up_requests(self) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 404
        response.text = "model unavailable"
        session = Mock(spec=requests.Session)
        session.post.return_value = response

        with tempfile.TemporaryDirectory() as temporary_directory:
            client = benchmark.OpenRouterEmbeddingClient(
                "test-key",
                Path(temporary_directory),
                session=session,
            )
            self.assertIsNone(client.get_embeddings_batch(["first"], "bad/model"))
            self.assertIsNone(client.get_embeddings_batch(["second"], "bad/model"))

        self.assertEqual(session.post.call_count, 1)

    def test_retryable_error_stops_at_max_attempts_without_final_sleep(self) -> None:
        response = Mock(spec=requests.Response)
        response.status_code = 429
        response.text = "rate limited"
        response.headers = {}
        session = Mock(spec=requests.Session)
        session.post.return_value = response

        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            patch.object(benchmark, "MAX_RETRIES", 1),
            patch.object(benchmark.time, "sleep") as sleep,
        ):
            client = benchmark.OpenRouterEmbeddingClient(
                "test-key",
                Path(temporary_directory),
                session=session,
            )
            result = client.get_embeddings_batch(["first"], "limited/model")

        self.assertIsNone(result)
        self.assertEqual(session.post.call_count, 1)
        sleep.assert_not_called()


class SaveResultsTests(unittest.TestCase):
    def test_save_results_replaces_rerun_models_and_preserves_other_rows(self) -> None:
        existing_rows = [
            {
                "Model Name": "existing/model",
                "Accuracy (%)": 50.0,
                "Correct": 2,
                "Total": 4,
                "Embedding Dim": 2,
            },
            {
                "Model Name": "rerun/model",
                "Accuracy (%)": 0.0,
                "Correct": 0,
                "Total": 4,
                "Embedding Dim": 2,
            },
        ]
        rerun_rows = [
            {
                "Model Name": "rerun/model",
                "Accuracy (%)": 50.0,
                "Correct": 2,
                "Total": 4,
                "Embedding Dim": 3,
            }
        ]

        with tempfile.TemporaryDirectory() as temporary_directory:
            results_path = Path(temporary_directory) / "results.csv"
            pd.DataFrame(existing_rows).to_csv(results_path, index=False)
            saved_frame = benchmark.save_results(rerun_rows, results_path)

        self.assertEqual(
            saved_frame["Model Name"].tolist(), ["existing/model", "rerun/model"]
        )
        rerun_result = saved_frame.loc[saved_frame["Model Name"] == "rerun/model"].iloc[
            0
        ]
        self.assertEqual(rerun_result["Correct"], 2)
        self.assertEqual(rerun_result["Embedding Dim"], 3)


class PrefetchTests(unittest.TestCase):
    def test_prefetch_splits_options_into_bounded_batches(self) -> None:
        client = Mock(spec=benchmark.OpenRouterEmbeddingClient)
        client.get_embeddings_batch.return_value = Mock()
        dataset = [
            {"options": ["a", "b", "c", "d"]},
            {"options": ["e", "f", "g", "h"]},
        ]

        benchmark.prefetch_model_embeddings(client, "test/model", dataset, batch_size=5)

        requested_batches = [
            call.args[0] for call in client.get_embeddings_batch.call_args_list
        ]
        self.assertEqual(
            requested_batches, [["a", "b", "c", "d", "e"], ["f", "g", "h"]]
        )


if __name__ == "__main__":
    unittest.main()
