"""
Test script for explanation generation - processes only the first question.

This script is useful for testing the setup and API connectivity before running
the full experiment.
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import List, Dict, Optional
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError, as_completed

# Load environment variables
load_dotenv(dotenv_path="../.env")

# Configuration
BENCHMARK_DATA_PATH = '../preprocess-data/benchmark_dataset.json'
OUTPUT_PATH = 'test_output.json'
LOG_FILE_PATH = 'test.log'
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Timeout configuration
REQUEST_TIMEOUT = 150  # Maximum seconds to wait for a single API request
MAX_CONCURRENT_REQUESTS = 6  # Number of simultaneous API requests

# Use fewer models for testing
MODELS_TO_USE = [
    "moonshotai/kimi-linear-48b-a3b-instruct",
    "moonshotai/kimi-k2-thinking",
    "minimax/minimax-m2",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "z-ai/glm-4.6",
    "deepseek/deepseek-chat-v3.1",
    "openai/gpt-oss-120b",
    "x-ai/grok-4"
]

# Number of questions to process for testing
NUM_QUESTIONS_TO_TEST = 1

# System prompt for explanation generation
SYSTEM_PROMPT = """You are an expert in Persian classical poetry. When given a poetry couplet (beit),
provide a clear and concise Persian interpretation of its meaning in a few sentences.
Focus on the conceptual meaning and main message of the verse.
ALWAYS answer in Persian.
DO NOT use any introductory or meta phrases such as 'این بیت به این معنی است که' or 'شاعر می‌گوید'.
Start directly with the interpretation itself. Keep the tone natural and literary.
Only output the interpretation text — no explanations, formatting, or additional commentary."""

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)


def load_benchmark_dataset(path: str, limit: int = None) -> List[Dict]:
    """Load the benchmark dataset from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        if limit:
            dataset = dataset[:limit]

        logging.info(f"Successfully loaded {len(dataset)} questions from {path}")
        return dataset
    except Exception as e:
        logging.error(f"Error loading dataset: {e}")
        raise


def _get_explanation_internal(client: OpenAI, model_id: str, poetry_text: str) -> Optional[Dict]:
    """
    Get an explanation for a poetry couplet from an LLM.

    Returns a dict with 'explanation' and optionally 'reasoning' (for thinking models).
    """
    prompt = f"Please provide a simple interpretation of this Persian poetry couplet:\n\n{poetry_text}"

    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0.5,
            max_tokens=12000,
        )

        if completion.choices and len(completion.choices) > 0:
            message = completion.choices[0].message
            content = message.content

            if content:
                result = {
                    "explanation": content.strip()
                }

                # Check if this is a thinking model with reasoning
                if hasattr(message, 'reasoning') and message.reasoning:
                    result["reasoning"] = message.reasoning.strip()
                    logging.info(f"  (with reasoning: {len(message.reasoning)} chars)")

                return result

        logging.warning(f"Received unexpected response from {model_id}")
        return None

    except RateLimitError as e:
        logging.warning(f"Rate limit hit for model {model_id}: {e}")
        return None
    except APIError as e:
        logging.error(f"API Error for model {model_id}: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error calling model {model_id}: {e}")
        return None


def get_explanation(client: OpenAI, model_id: str, poetry_text: str) -> Optional[Dict]:
    """
    Get an explanation with timeout protection.

    Wraps _get_explanation_internal with a timeout to prevent hanging requests.
    """
    executor = ThreadPoolExecutor(max_workers=1)
    try:
        future = executor.submit(_get_explanation_internal, client, model_id, poetry_text)
        result = future.result(timeout=REQUEST_TIMEOUT)
        return result
    except FutureTimeoutError:
        logging.error(f"⏱️  Request timeout ({REQUEST_TIMEOUT}s exceeded) for model {model_id}")
        print(f" ⏱️  TIMEOUT ({REQUEST_TIMEOUT}s)")
        future.cancel()
        return None
    except Exception as e:
        logging.error(f"Error in timeout wrapper for {model_id}: {e}")
        return None
    finally:
        executor.shutdown(wait=False)


def get_explanations_concurrent(
    client: OpenAI,
    model_ids: List[str],
    poetry_text: str,
    max_workers: int = MAX_CONCURRENT_REQUESTS
) -> Dict[str, Optional[Dict]]:
    """
    Get explanations from multiple models concurrently.

    Args:
        client: OpenAI client instance
        model_ids: List of model IDs to query
        poetry_text: The poetry text to explain
        max_workers: Maximum number of concurrent requests

    Returns:
        Dictionary mapping model_id to result (or None if failed)
    """
    results = {}
    total_models = len(model_ids)
    completed_count = 0

    print(f"  Requesting from {total_models} models concurrently (max {max_workers} simultaneous)...")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all model requests
        future_to_model = {
            executor.submit(get_explanation, client, model_id, poetry_text): model_id
            for model_id in model_ids
        }

        # Collect results as they complete
        for future in as_completed(future_to_model):
            model_id = future_to_model[future]
            completed_count += 1

            try:
                result = future.result()
                results[model_id] = result

                if result:
                    explanation_text = result.get('explanation', '')
                    has_reasoning = 'reasoning' in result
                    print(
                        f"  ✓ [{completed_count}/{total_models}] {model_id}: "
                        f"{len(explanation_text)} chars{', +reasoning' if has_reasoning else ''}"
                    )
                    print(f"    → {explanation_text[:100]}...")
                    if has_reasoning:
                        reasoning_preview = result['reasoning'][:100]
                        print(f"    🤔 Reasoning: {reasoning_preview}...")
                else:
                    print(f"  ✗ [{completed_count}/{total_models}] {model_id}: Failed")

            except Exception as e:
                print(f"  ✗ [{completed_count}/{total_models}] {model_id}: Error - {e}")
                results[model_id] = None

    return results


def save_dataset(dataset: List[Dict], output_path: str):
    """Save the enriched dataset to a JSON file."""
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        logging.info(f"Successfully saved dataset to {output_path}")
    except Exception as e:
        logging.error(f"Error saving dataset: {e}")
        raise


def main():
    """Main test function."""
    print("="*80)
    print(f"TEST: Generate Explanations for First {NUM_QUESTIONS_TO_TEST} Question(s)")
    print("="*80)

    # Validate API key
    if not OPENROUTER_API_KEY:
        print("Error: OPENROUTER_API_KEY not found!")
        print("Please set it in your .env file")
        return

    # Initialize client
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            timeout=REQUEST_TIMEOUT - 5,  # Slightly less than REQUEST_TIMEOUT
        )
        print(f"✓ OpenAI client initialized (timeout: {REQUEST_TIMEOUT}s per request)\n")
    except Exception as e:
        print(f"Error initializing client: {e}")
        return

    # Load dataset (limited)
    try:
        dataset = load_benchmark_dataset(BENCHMARK_DATA_PATH, limit=NUM_QUESTIONS_TO_TEST)
        print(f"✓ Loaded {len(dataset)} question(s) for testing\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Process questions
    print(f"Using {len(MODELS_TO_USE)} models with {MAX_CONCURRENT_REQUESTS} concurrent requests")
    print(f"Models: {', '.join(MODELS_TO_USE)}\n")

    for q_idx, question in enumerate(dataset):
        question_id = question.get('id', f'index_{q_idx}')
        print(f"Processing Question ID: {question_id}")
        print("-" * 80)

        if 'options' not in question or len(question['options']) != 4:
            print(f"⚠️  Skipping question {question_id}: invalid options\n")
            continue

        question['explanations'] = []

        # Process each option
        for opt_idx, option_text in enumerate(question['options']):
            print(f"\n[Option {opt_idx + 1}/4]")
            print(f"Text: {option_text}")
            print()

            # Get explanations from all models concurrently
            llm_explanations = get_explanations_concurrent(
                client=client,
                model_ids=MODELS_TO_USE,
                poetry_text=option_text,
                max_workers=MAX_CONCURRENT_REQUESTS
            )

            # Store explanations
            option_explanation = {
                "option_index": opt_idx,
                "option_text": option_text,
                "llm_explanations": llm_explanations
            }
            question['explanations'].append(option_explanation)

        print("\n" + "="*80)

    # Save results
    save_dataset(dataset, OUTPUT_PATH)
    print(f"\n✓ Test completed! Results saved to: {OUTPUT_PATH}")
    print(f"✓ Log saved to: {LOG_FILE_PATH}")
    print("\nReview the output to ensure everything looks correct before running the full experiment.")


if __name__ == "__main__":
    main()
