import json
import os
import logging
import time
from openai import OpenAI, RateLimitError, APIError

# --- Configuration ---
JSON_FILE_PATH = "data/questions.json"  # Path to your exam JSON file
LOG_FILE_PATH = "results.log"  # Path to save the logs
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Get API key from environment variable

# List of models to test (use the exact identifiers from OpenRouter)
MODELS_TO_TEST = [
    "openai/gpt-4o",
    "anthropic/claude-3-opus",
    "google/gemini-pro",
    "mistralai/mistral-large-latest",
    "meta-llama/llama-3-70b-instruct",
    "cognitivecomputations/dolphin-mixtral-8x7b",
    # Add more model identifiers as needed from OpenRouter
    # "arliai/qwq-32b-arliai-rpr-v1:free", # Example free model
]

# Optional headers for OpenRouter ranking (replace with your actual info if desired)
YOUR_SITE_URL = "https://example.com" # Optional
YOUR_SITE_NAME = "AI Exam Tester"    # Optional
EXTRA_HEADERS = {
    "HTTP-Referer": YOUR_SITE_URL,
    "X-Title": YOUR_SITE_NAME,
}

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler() # Also print logs to console
    ]
)

# --- Helper Functions ---

def load_questions(file_path):
    """Loads questions from the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded {len(data.get('questions', []))} questions from {file_path}")
        return data.get('questions', [])
    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        logging.error(f"Error: Could not decode JSON from {file_path}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading JSON: {e}")
        return None

def format_prompt(question_data):
    """Formats the question and answers into a prompt for the AI."""
    prompt = f"Question: {question_data['question_text']}\n\n"
    prompt += "Options:\n"
    for answer in question_data['answers']:
        prompt += f"{answer['answer_number']}. {answer['answer_text_1']} | {answer['answer_text_2']}\n"
    prompt += "\nInstruction: Analyze the options. Identify the *single* option (by its number) that has a different concept and message from the others. Respond with *only* the number (1, 2, 3, or 4). Do not provide any explanation or other text."
    return prompt

def get_model_response(client, model_id, prompt):
    """Sends the prompt to the specified model and retrieves the response."""
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": "You are an AI assistant tasked with identifying the outlier in a multiple-choice question based on conceptual difference. Respond only with the number of the answer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, # Lower temperature for more deterministic output
            max_tokens=10,   # Limit response length
            extra_headers=EXTRA_HEADERS,
            # extra_body={}, # Usually not needed unless specific OpenRouter features require it
        )
        response_text = completion.choices[0].message.content.strip()
        return response_text
    except RateLimitError as e:
        logging.warning(f"Rate limit hit for model {model_id}. Waiting and retrying... Error: {e}")
        time.sleep(30) # Wait 30 seconds before potential retry (adjust as needed)
        # Basic retry logic (could be made more robust)
        try:
             completion = client.chat.completions.create(
                model=model_id, messages=[{"role": "user", "content": prompt}],
                temperature=0.1, max_tokens=10, extra_headers=EXTRA_HEADERS)
             response_text = completion.choices[0].message.content.strip()
             return response_text
        except Exception as retry_e:
             logging.error(f"Retry failed for model {model_id}. Error: {retry_e}")
             return f"ERROR: Retry Failed - {retry_e}"

    except APIError as e:
        logging.error(f"API Error for model {model_id}: {e}")
        return f"ERROR: API Error - {e}"
    except Exception as e:
        logging.error(f"Unexpected error calling model {model_id}: {e}")
        return f"ERROR: Unexpected - {e}"

def validate_response(response_text):
    """Checks if the response is one of the expected numbers."""
    if response_text in ['1', '2', '3', '4']:
        return response_text
    else:
        return None # Indicates invalid format

# --- Main Execution ---
if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        logging.error("FATAL: OPENROUTER_API_KEY environment variable not set.")
        exit(1)

    # Initialize OpenAI client for OpenRouter
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )

    questions = load_questions(JSON_FILE_PATH)

    if not questions:
        logging.error("No questions loaded. Exiting.")
        exit(1)

    # Iterate through each model
    for model_id in MODELS_TO_TEST:
        logging.info(f"\n--- Starting Exam for Model: {model_id} ---")

        # Iterate through each question
        for question in questions:
            q_num = question.get("question_number", "N/A")
            logging.info(f"Asking Model '{model_id}' Question {q_num}...")

            # Format the prompt
            prompt = format_prompt(question)
            # logging.debug(f"Formatted Prompt for Q{q_num}:\n{prompt}") # Uncomment for debugging prompts

            # Get response from the model
            raw_response = get_model_response(client, model_id, prompt)

            # Validate and log the response
            validated_answer = validate_response(raw_response)

            if "ERROR:" in raw_response:
                 logging.error(f"Model: {model_id} | Question: {q_num} | Status: API Call Failed | Raw Response: {raw_response}")
            elif validated_answer:
                logging.info(f"Model: {model_id} | Question: {q_num} | Status: Success | Answer: {validated_answer}")
            else:
                logging.warning(f"Model: {model_id} | Question: {q_num} | Status: Invalid Response Format | Raw Response: '{raw_response}'")

            # Optional: Add a small delay between API calls to avoid rate limiting
            time.sleep(1) # Wait 1 second

        logging.info(f"--- Finished Exam for Model: {model_id} ---")

    logging.info("\n=== All Models Tested ===")