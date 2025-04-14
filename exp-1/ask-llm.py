import json
import os
import logging
import time
from openai import OpenAI, RateLimitError, APIError
from dotenv import load_dotenv
import csv
import re

load_dotenv(dotenv_path="../.env")

# --- Configuration ---
JSON_FILE_PATH = "data/few-shot-questions.json"  # Path to questions (zero-shot and few-shot questions are different)
LOG_FILE_PATH = "results.log"  # Path to save the detailed logs
CSV_RESULT_FILE_PATH = "few-shot/few-shot-answers.csv" # Path for the results CSV file
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") # Get API key from environment variable

# List of models to test (use the exact identifiers from OpenRouter)
MODELS_TO_TEST = [
    "google/gemini-2.0-flash-001",
    "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "openai/o1-mini",
    "meta-llama/llama-3.3-70b-instruct",
    "google/gemini-2.5-pro-preview-03-25",
    "deepseek/deepseek-chat-v3-0324",
    "anthropic/claude-3.5-sonnet",
    "deepseek/deepseek-r1",
    "google/gemma-3-27b-it",
    "google/gemma-3-12b-it",
    "qwen/qwen-2.5-72b-instruct",
]

# Zero-shot
#SYSTEM_PROMPT="You are an AI assistant analyzing Persian poetry couplets. Identify the outlier based on concept/message. Respond *only* with the single digit number (1, 2, 3, or 4) of the outlier option. Output nothing else."

# Few-shot
SYSTEM_PROMPT="""You are an expert literary critic with a deep understanding of Persian poetry, its cultural nuances, and its stylistic features. Your task is to analyze a set of poetic options—each option presenting two parts of a couplet—and identify the one option that deviates in conceptual meaning or thematic message from the others. Focus exclusively on the underlying concepts, disregarding stylistic or linguistic differences.

For example:

---
Options:
1. طریق عشق پرآشوب و فتنه است ای دل - بیفتد آن که در این راه با شتاب رود
2. گر نور عشق حق به دل و جانت اوفتد - بالله از آفتاب فلک خوبتر شوی
3. شکوه عشق نگه کن که موی مجنون را - فلک به شعشعه آفتاب، شانه کند
4. فرزانه درآید به پری خانه مقصود - هر کس که در این بادیه دیوانه عشق است

Correct answer: 1

(Option 1 warns against hastily pursuing the turbulent path of love, whereas the other options present love as an uplifting force)

---
Options:
1. شمشیر نیک از آهن بد چون کند کسی؟ - ناکس تربیت نشود ای حکیم کس
2. سگ به دریای هفت گانه بشوی - که چو تر شد پلیدتر باشد
3. ز وحشی نیاید که مردم شود - به سعی اندر او تربیت گم شود
4. سگ اصحاب کهف روزی چند - پی نیکان گرفت و مردم شد

Correct answer: 4

(Option 4 emphasizes the significant impact of upbringing, unlike the other options which imply that upbringing makes little difference)

---
Options:
1. هر چند خوشگوار بود باده غرور - زین می فزون از سنگ نگه دار شیشه را
2. از ساده دلی هر که دهد پند به مغرور - بیدار به افسانه کند خواب گران را
3. کبر مفروش به مردم که به میزان نظر - زود گردد سبک آن کس که بود سنگین تر
4. خاک بر فرقش اگر از کبر سر بالا کند - هر که داند بازگشت او به غیر از خاک نیست

Correct answer: 2

(The meaning of option 2 is the ineffectiveness of giving advice to the arrogant, while the common meaning of the other options is the recommendation to avoid arrogance)"""

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE_PATH),
        logging.StreamHandler()
    ]
)

# --- Helper Functions ---

def load_questions(file_path):
    """Loads questions from the JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        questions_list = data.get('questions')
        if not isinstance(questions_list, list):
             logging.error(f"Error: 'questions' key not found or is not a list in {file_path}")
             return None
        logging.info(f"Successfully loaded {len(questions_list)} questions from {file_path}")
        return questions_list
    except FileNotFoundError:
        logging.error(f"Error: JSON file not found at {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error: Could not decode JSON from {file_path}. Details: {e}")
        return None
    except Exception as e:
        logging.error(f"An unexpected error occurred loading JSON: {e}")
        return None

def format_prompt(question_data):
    """Formats the answers into a prompt for the AI, omitting the redundant question text."""
    # Validate question_data structure - only 'answers' is strictly needed now for the prompt
    if 'answers' not in question_data or not isinstance(question_data['answers'], list):
        logging.error(f"Invalid question structure or missing 'answers' for question_number: {question_data.get('question_number', 'N/A')}")
        return None # Indicate error

    # Start directly with the options
    prompt = "Analyze the conceptual meaning of the following options:\n\n"
    prompt += "Options:\n"
    try:
        # Ensure options are presented consistently, sorted by answer_number
        sorted_answers = sorted(question_data['answers'], key=lambda x: int(x.get('answer_number', 0)))
        if not sorted_answers:
             logging.error(f"No valid answers found to format for question {question_data.get('question_number', 'N/A')}")
             return None
        for answer in sorted_answers:
            # Validate answer structure
            if not all(k in answer for k in ['answer_number', 'answer_text_1', 'answer_text_2']):
                 logging.warning(f"Skipping malformed answer in question {question_data.get('question_number', 'N/A')}")
                 continue
            prompt += f"{answer['answer_number']}. {answer['answer_text_1']} - {answer['answer_text_2']}\n"
    except (TypeError, ValueError, KeyError) as e:
        logging.error(f"Error formatting options for question {question_data.get('question_number', 'N/A')}: {e}")
        return None # Indicate error

    # Add the core instruction
    prompt += "\nInstruction: Identify the *single* option (by its number) that has a different concept and message from the others. Respond with *only* the number (1, 2, 3, or 4). Do not provide any explanation or other text."
    print(prompt)
    return prompt


def get_model_response(client, model_id, prompt):
    """
    Sends the prompt to the specified model and retrieves the response.
    Returns a tuple: (response_text: str | None, completion_tokens: int | None)
    response_text contains 'ERROR:' prefix on failure.
    """
    if not prompt:
        return "ERROR: Invalid Prompt", None
    try:
        completion = client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            stream=False,
            temperature=0,
        )

        tokens_used = None
        response_text = "ERROR: Malformed Response Object"

        print(completion)

        if hasattr(completion, 'usage') and hasattr(completion.usage, 'completion_tokens'):
             tokens_used = completion.usage.completion_tokens

        if completion.choices and isinstance(completion.choices, list) and len(completion.choices) > 0 and \
           hasattr(completion.choices[0], 'message') and hasattr(completion.choices[0].message, 'content'):
             content = completion.choices[0].message.content
             finish_reason = completion.choices[0].finish_reason
             if content is None:
                  logging.warning(f"Received None content from {model_id}. Finish Reason: {finish_reason}")
                  response_text = "ERROR: None Content Received"
             else:
                 response_text = content.strip()
                 if finish_reason == 'length':
                     logging.warning(f"Model {model_id} response may be truncated (finish_reason='length'). Raw content: '{response_text}'")
        else:
             logging.warning(f"Received unexpected response object structure from {model_id}. Completion: {completion}")

        return response_text, tokens_used

    except RateLimitError as e:
        wait_time = 60
        logging.warning(f"Rate limit hit for model {model_id}. Waiting {wait_time} seconds... Error: {e}")
        time.sleep(wait_time)
        return f"ERROR: Rate Limit Hit - {e}", None
    except APIError as e:
        err_msg = e.message or str(e.body) # Get error details
        logging.error(f"API Error for model {model_id} (Code: {e.status_code}): {err_msg}")
        if e.status_code == 429:
             return f"ERROR: API Rate Limit (429) - {err_msg}", None
        elif "context_length_exceeded" in err_msg:
             return f"ERROR: Context Length Exceeded - {err_msg}", None
        # Add check for potentially unavailable free models
        elif e.status_code == 500 and "Model is overloaded" in err_msg:
             logging.warning(f"Model {model_id} might be overloaded (500 error).")
             return f"ERROR: Model Overloaded (500) - {err_msg}", None
        elif e.status_code == 400 and "does not exist" in err_msg:
             logging.error(f"Model {model_id} may not exist or is unavailable.")
             return f"ERROR: Model Not Found (400) - {err_msg}", None
        else:
             return f"ERROR: API Error ({e.status_code}) - {err_msg}", None
    except Exception as e:
        logging.error(f"Unexpected error calling model {model_id}: {type(e).__name__} - {e}")
        return f"ERROR: Unexpected - {type(e).__name__} - {e}", None


def validate_response(response_text):
    """
    Checks if the response contains one of the expected single digits (1, 2, 3, or 4).
    Uses regex to find the first occurrence.
    """
    if not isinstance(response_text, str) or not response_text:
        return None
    match = re.search(r'[1234]', response_text)
    if match:
        return match.group(0)
    else:
        return None

# --- Main Execution ---
if __name__ == "__main__":
    if not OPENROUTER_API_KEY:
        logging.error("FATAL: OPENROUTER_API_KEY environment variable not set.")
        exit(1)

    # Initialize OpenAI client
    try:
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=OPENROUTER_API_KEY,
            timeout=60.0, # 1 minute timeout per request
        )
        logging.info("OpenAI client initialized with 60 second timeout.")
    except Exception as e:
         logging.error(f"FATAL: Failed to initialize OpenAI client: {e}")
         exit(1)

    questions = load_questions(JSON_FILE_PATH)

    if not questions:
        logging.error("No questions loaded or error loading questions. Exiting.")
        exit(1)

    # --- Initialize CSV File ---
    try:
        with open(CSV_RESULT_FILE_PATH, 'w', newline='', encoding='utf-8') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['model_name', 'question_number', 'model_answer', 'completion_tokens'])
            logging.info(f"Opened {CSV_RESULT_FILE_PATH} for writing results.")

            total_questions = len(questions)
            # Iterate through each model
            for model_index, model_id in enumerate(MODELS_TO_TEST):
                logging.info(f"\n--- Starting Exam for Model {model_index+1}/{len(MODELS_TO_TEST)}: {model_id} ---")
                model_success_count = 0
                model_unavailable = False # Flag to skip model if it's not found

                # Iterate through each question
                for question_index, question in enumerate(questions):
                    # Skip remaining questions for this model if it was flagged as unavailable
                    if model_unavailable:
                        logging.warning(f"[{progress}] Skipping Question {q_num} for unavailable model {model_id}")
                        # Still write a blank row to CSV for completeness
                        csv_writer.writerow([model_id, q_num, "", ""])
                        continue

                    q_num = question.get("question_number", f"Unknown_{question_index+1}")
                    progress = f"Model {model_index+1}/{len(MODELS_TO_TEST)} - Q {question_index+1}/{total_questions}"
                    logging.info(f"[{progress}] Asking Model '{model_id}' Question {q_num}...")

                    validated_answer = None
                    tokens_used = None
                    raw_response = "ERROR: Prompt Formatting Failed"

                    # Format the prompt (now omits question_text)
                    prompt = format_prompt(question)
                    if prompt:
                        # Get response and tokens from the model
                        raw_response, tokens_used = get_model_response(client, model_id, prompt)

                        # Check for specific "Model Not Found" error to skip further attempts
                        if raw_response and "ERROR: Model Not Found" in raw_response:
                             model_unavailable = True
                             logging.error(f"[{progress}] Model {model_id} flagged as unavailable. Skipping remaining questions for this model.")


                        # Validate the response only if no error occurred
                        if raw_response and "ERROR:" not in raw_response:
                             validated_answer = validate_response(raw_response)

                    # Determine the values to write to CSV
                    csv_answer = validated_answer if validated_answer is not None else ""
                    csv_tokens = tokens_used if tokens_used is not None else ""

                    # Write result to CSV
                    csv_writer.writerow([model_id, q_num, csv_answer, csv_tokens])

                    # Log the outcome
                    if raw_response and "ERROR:" in raw_response:
                        logging.error(f"[{progress}] Model: {model_id} | Question: {q_num} | Status: Failed | Detail: {raw_response}")
                    elif validated_answer:
                        logging.info(f"[{progress}] Model: {model_id} | Question: {q_num} | Status: Success | Answer: {validated_answer} | Tokens: {tokens_used}")
                        model_success_count += 1
                    else:
                        # Invalid format after successful API call
                        logging.warning(f"[{progress}] Model: {model_id} | Question: {q_num} | Status: Invalid Response Format | Raw: '{raw_response}' | Tokens: {tokens_used} | Saved Answer: ''")


                    # Optional: Adjust sleep time
                    # Consider slightly longer sleep for free models or after errors
                    sleep_time = 3 if ":free" in model_id or (raw_response and "ERROR:" in raw_response) else 2
                    logging.debug(f"Sleeping for {sleep_time} seconds...")
                    time.sleep(sleep_time)

                logging.info(f"--- Finished Exam for Model: {model_id} ({model_success_count}/{total_questions} valid answers) ---")

    except IOError as e:
         logging.error(f"FATAL: Could not write to CSV file {CSV_RESULT_FILE_PATH}. Error: {e}")
         exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during main execution: {e}")
        import traceback
        logging.error(traceback.format_exc()) # Log detailed traceback for unexpected errors


    logging.info(f"\n=== All Models Tested. Results saved to {CSV_RESULT_FILE_PATH} ===")
