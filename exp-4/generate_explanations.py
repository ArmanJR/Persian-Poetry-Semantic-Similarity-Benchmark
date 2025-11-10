"""
Generate LLM explanations for Persian poetry couplets.

This script takes the benchmark dataset and asks multiple LLMs to provide
simple interpretations for each poetry couplet (option) in the dataset.

Features:
- Concurrent requests (6 simultaneous by default) for faster processing
- Automatic timeout protection (150s per request)
- Incremental saving after each option (never lose progress)
- Resume capability: automatically skips already-completed options
- Crash recovery: restart anytime, picks up where it left off
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
OUTPUT_PATH = 'benchmark_dataset_with_explanations.json'
LOG_FILE_PATH = 'generation.log'
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Timeout configuration
REQUEST_TIMEOUT = 150  # Maximum seconds to wait for a single API request
MAX_CONCURRENT_REQUESTS = 6  # Number of simultaneous API requests

# Cache configuration
ENABLE_RESUME = True  # Skip already-processed options
SAVE_AFTER_EACH_OPTION = True  # Save progress after each option (not just each question)

# List of models to use for generating explanations
MODELS_TO_USE = [
    "moonshotai/kimi-linear-48b-a3b-instruct",
    "moonshotai/kimi-k2-thinking",
    "minimax/minimax-m2",
    "anthropic/claude-haiku-4.5",
    "google/gemini-2.5-flash",
    "google/gemini-2.5-pro",
    "z-ai/glm-4.6",
    "deepseek/deepseek-chat-v3.1",
    "openai/gpt-oss-120b",
    "openai/gpt-5",
    "x-ai/grok-4",
    "deepseek/deepseek-r1-0528",
    "qwen/qwen3-235b-a22b-thinking-2507",
]

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


def load_benchmark_dataset(path: str) -> List[Dict]:
    """Load the benchmark dataset from JSON file."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logging.info(f"Successfully loaded {len(dataset)} questions from {path}")
        return dataset
    except FileNotFoundError:
        logging.error(f"Error: Benchmark data file not found at '{path}'")
        raise
    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from {path}. Details: {e}")
        raise
    except Exception as e:
        logging.error(f"An unexpected error occurred loading dataset: {e}")
        raise


def load_existing_progress(path: str) -> Optional[List[Dict]]:
    """
    Load existing progress file if it exists.

    Returns the dataset with partial progress, or None if file doesn't exist.
    """
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        logging.info(f"Found existing progress file with {len(dataset)} questions")
        return dataset
    except Exception as e:
        logging.warning(f"Could not load existing progress file: {e}")
        return None


def merge_datasets(base_dataset: List[Dict], progress_dataset: List[Dict]) -> List[Dict]:
    """
    Merge progress dataset into base dataset.

    Copies over any existing 'explanations' from progress to base.
    """
    # Create a mapping of question ID to progress data
    progress_map = {q.get('id', f'index_{i}'): q for i, q in enumerate(progress_dataset)}

    for i, question in enumerate(base_dataset):
        question_id = question.get('id', f'index_{i}')

        if question_id in progress_map:
            progress_question = progress_map[question_id]

            # Copy over explanations if they exist
            if 'explanations' in progress_question:
                question['explanations'] = progress_question['explanations']
                logging.debug(f"Restored {len(progress_question['explanations'])} option explanations for question {question_id}")

    return base_dataset


def is_option_complete(question: Dict, option_index: int, required_models: List[str]) -> bool:
    """
    Check if a specific option already has explanations from all required models.

    Returns True if we can skip this option.
    """
    if 'explanations' not in question:
        return False

    # Find the explanation entry for this option
    for exp in question['explanations']:
        if exp.get('option_index') == option_index:
            llm_explanations = exp.get('llm_explanations', {})

            # Check if all required models have non-null responses
            for model in required_models:
                if model not in llm_explanations or llm_explanations[model] is None:
                    return False

            # All models present and non-null
            return True

    return False


def _get_explanation_internal(client: OpenAI, model_id: str, poetry_text: str) -> Optional[Dict]:
    """
    Get an explanation for a poetry couplet from an LLM.

    Args:
        client: OpenAI client instance
        model_id: The model to use
        poetry_text: The Persian poetry couplet

    Returns:
        Dict with 'explanation' and optionally 'reasoning' (for thinking models), or None if failed
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
                    logging.info(f"      (with reasoning: {len(message.reasoning)} chars)")

                return result

        logging.warning(f"Received unexpected response from {model_id}")
        return None

    except RateLimitError as e:
        wait_time = 60
        logging.warning(f"Rate limit hit for model {model_id}. Waiting {wait_time} seconds... Error: {e}")
        time.sleep(wait_time)
        return None

    except APIError as e:
        err_msg = e.message or str(e.body)
        logging.error(f"API Error for model {model_id} (Code: {e.status_code}): {err_msg}")
        return None

    except Exception as e:
        logging.error(f"Unexpected error calling model {model_id}: {type(e).__name__} - {e}")
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
                    logging.info(
                        f"    ✓ [{completed_count}/{total_models}] {model_id}: "
                        f"{len(explanation_text)} chars{', +reasoning' if has_reasoning else ''}"
                    )
                else:
                    logging.warning(f"    ✗ [{completed_count}/{total_models}] {model_id}: Failed")

            except Exception as e:
                logging.error(f"    ✗ [{completed_count}/{total_models}] {model_id}: Unexpected error - {e}")
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
    """Main execution function."""
    logging.info("="*80)
    logging.info("Generate Explanations for Persian Poetry Couplets")
    logging.info("="*80)

    # Validate API key
    if not OPENROUTER_API_KEY:
        logging.error("FATAL: OPENROUTER_API_KEY environment variable not set.")
        print("Error: OPENROUTER_API_KEY not found!")
        print("Please create a .env file with your OpenRouter API key:")
        print("  OPENROUTER_API_KEY=your_api_key_here")
        return

    # Initialize OpenAI client
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            timeout=REQUEST_TIMEOUT - 5,  # Slightly less than REQUEST_TIMEOUT
        )
        logging.info(f"OpenAI client initialized (timeout: {REQUEST_TIMEOUT}s per request)")
    except Exception as e:
        logging.error(f"FATAL: Failed to initialize OpenAI client: {e}")
        return

    # Load dataset
    try:
        dataset = load_benchmark_dataset(BENCHMARK_DATA_PATH)
    except Exception as e:
        logging.error(f"Failed to load dataset: {e}")
        return

    # Load existing progress and merge if resume is enabled
    if ENABLE_RESUME:
        progress_dataset = load_existing_progress(OUTPUT_PATH)
        if progress_dataset:
            logging.info("Resume mode enabled - merging existing progress...")
            dataset = merge_datasets(dataset, progress_dataset)
            logging.info("✓ Existing progress merged. Will skip already-completed options.")
        else:
            logging.info("Resume mode enabled but no existing progress found. Starting fresh.")
    else:
        logging.info("Resume mode disabled. Starting fresh (will overwrite existing file).")

    # Process each question
    total_questions = len(dataset)
    total_options = total_questions * 4
    processed_options = 0
    skipped_options = 0

    logging.info(f"\nProcessing {total_questions} questions ({total_options} total options)")
    logging.info(f"Using {len(MODELS_TO_USE)} models with {MAX_CONCURRENT_REQUESTS} concurrent requests")
    logging.info(f"Models: {', '.join(MODELS_TO_USE)}")
    if ENABLE_RESUME:
        logging.info(f"Resume enabled: Will skip already-completed options")
    if SAVE_AFTER_EACH_OPTION:
        logging.info(f"Auto-save enabled: Progress saved after each option")
    logging.info("")

    try:
        for q_idx, question in enumerate(dataset):
            question_id = question.get('id', f'index_{q_idx}')
            logging.info(f"[Question {q_idx + 1}/{total_questions}] Processing Question ID: {question_id}")

            # Validate question structure
            if 'options' not in question or len(question['options']) != 4:
                logging.warning(f"Skipping question {question_id}: invalid options")
                continue

            # Initialize explanations structure if not exists
            if 'explanations' not in question:
                question['explanations'] = []

            # Process each option
            for opt_idx, option_text in enumerate(question['options']):
                # Check if this option is already complete
                if ENABLE_RESUME and is_option_complete(question, opt_idx, MODELS_TO_USE):
                    skipped_options += 1
                    logging.info(f"  [Option {opt_idx + 1}/4] ⏭️  SKIPPED (already complete): {option_text[:50]}...")
                    continue

                logging.info(f"  [Option {opt_idx + 1}/4] Processing: {option_text[:50]}...")
                logging.info(f"    Requesting explanations from {len(MODELS_TO_USE)} models concurrently (max {MAX_CONCURRENT_REQUESTS} simultaneous)...")

                # Get explanations from all models concurrently
                llm_explanations = get_explanations_concurrent(
                    client=client,
                    model_ids=MODELS_TO_USE,
                    poetry_text=option_text,
                    max_workers=MAX_CONCURRENT_REQUESTS
                )

                # Find and update existing explanation entry for this option, or create new one
                existing_entry = None
                for exp in question['explanations']:
                    if exp.get('option_index') == opt_idx:
                        existing_entry = exp
                        break

                if existing_entry:
                    # Update existing entry (merge new results with any existing ones)
                    existing_entry['llm_explanations'].update(llm_explanations)
                    logging.debug(f"  Updated existing explanation entry for option {opt_idx}")
                else:
                    # Create new entry
                    option_explanation = {
                        "option_index": opt_idx,
                        "option_text": option_text,
                        "llm_explanations": llm_explanations
                    }
                    question['explanations'].append(option_explanation)

                processed_options += 1
                logging.info(f"  ✓ Completed option {opt_idx + 1}/4 ({processed_options}/{total_options} processed, {skipped_options} skipped)\n")

                # Save progress after each option if enabled
                if SAVE_AFTER_EACH_OPTION:
                    save_dataset(dataset, OUTPUT_PATH)
                    logging.debug(f"  💾 Auto-saved after option {opt_idx + 1}")

            # Save progress after each question (if not already saving after each option)
            if not SAVE_AFTER_EACH_OPTION:
                save_dataset(dataset, OUTPUT_PATH)
                logging.info(f"✓ Progress saved after question {q_idx + 1}/{total_questions}\n")
            else:
                logging.info(f"✓ Question {q_idx + 1}/{total_questions} complete\n")

    except KeyboardInterrupt:
        logging.warning("\n\nInterrupted by user! Saving progress...")
        save_dataset(dataset, OUTPUT_PATH)
        logging.info("Partial results saved.")
        return
    except Exception as e:
        logging.error(f"Unexpected error during processing: {e}")
        import traceback
        logging.error(traceback.format_exc())
        logging.info("Saving progress before exit...")
        save_dataset(dataset, OUTPUT_PATH)
        return

    logging.info("="*80)
    logging.info(f"✓ All questions processed successfully!")
    logging.info(f"  Total options processed: {processed_options}/{total_options}")
    if skipped_options > 0:
        logging.info(f"  Options skipped (already complete): {skipped_options}")
    logging.info(f"  Results saved to: {OUTPUT_PATH}")
    logging.info("="*80)


if __name__ == "__main__":
    main()
