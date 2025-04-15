import json
import os

def create_benchmark_dataset(questions_filepath, answers_filepath, output_filepath):
    """
    Reads questions and answers from separate JSON files, combines them
    into the desired benchmark format, and saves to a new JSON file.

    Args:
        questions_filepath (str): Path to the JSON file containing questions and options.
        answers_filepath (str): Path to the JSON file containing correct answer numbers.
        output_filepath (str): Path to save the formatted benchmark dataset JSON file.
    """
    print(f"Starting dataset creation...")
    print(f"Reading questions from: {questions_filepath}")
    print(f"Reading answers from: {answers_filepath}")

    try:
        with open(questions_filepath, 'r', encoding='utf-8') as f:
            questions_data = json.load(f)
        print("Successfully loaded questions file.")
    except FileNotFoundError:
        print(f"Error: Questions file not found at '{questions_filepath}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{questions_filepath}'. Check file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred reading questions file: {e}")
        return

    try:
        with open(answers_filepath, 'r', encoding='utf-8') as f:
            answers_data = json.load(f)
        print("Successfully loaded answers file.")
    except FileNotFoundError:
        print(f"Error: Answers file not found at '{answers_filepath}'")
        return
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from '{answers_filepath}'. Check file format.")
        return
    except Exception as e:
        print(f"An unexpected error occurred reading answers file: {e}")
        return

    formatted_dataset = []
    processed_count = 0
    skipped_count = 0

    # Validate questions_data structure
    if 'questions' not in questions_data or not isinstance(questions_data.get('questions'), list):
        print(f"Error: Input questions file '{questions_filepath}' must contain a key 'questions' with a list value.")
        return

    print(f"Processing {len(questions_data['questions'])} questions...")

    for question_item in questions_data['questions']:
        try:
            question_number = question_item['question_number']
            question_number_str = str(question_number) # Keys in answers_data are strings

            # --- Retrieve Correct Answer ---
            if question_number_str not in answers_data:
                print(f"Warning: No answer found for question number {question_number}. Skipping.")
                skipped_count += 1
                continue

            correct_answer_number = answers_data[question_number_str] # 1-based

            # Convert to 0-based index and validate
            correct_answer_index = correct_answer_number - 1
            if not (0 <= correct_answer_index <= 3): # Assuming 4 options
                 print(f"Warning: Invalid correct answer number ({correct_answer_number}) for question {question_number}. Skipping.")
                 skipped_count += 1
                 continue

            # --- Combine Couplet Parts ---
            if 'answers' not in question_item or len(question_item['answers']) != 4:
                 print(f"Warning: Question {question_number} does not have exactly 4 'answers'. Skipping.")
                 skipped_count += 1
                 continue

            combined_options = []
            # Assuming answers list is ordered 1, 2, 3, 4
            for answer_option in question_item['answers']:
                 text1 = answer_option.get('answer_text_1', '') # Use .get for safety
                 text2 = answer_option.get('answer_text_2', '')
                 # Combine with a separator consistent with your examples
                 combined_text = f"{text1} - {text2}".strip()
                 combined_options.append(combined_text)

            # --- Create Formatted Entry ---
            formatted_question = {
                "id": question_number,
                "options": combined_options,
                "correct_answer_index": correct_answer_index
            }
            formatted_dataset.append(formatted_question)
            processed_count += 1

        except KeyError as e:
            q_num_str = question_item.get('question_number', 'UNKNOWN')
            print(f"Warning: Missing key {e} in data for question number {q_num_str}. Skipping.")
            skipped_count += 1
            continue
        except Exception as e:
            q_num_str = question_item.get('question_number', 'UNKNOWN')
            print(f"Warning: An unexpected error occurred processing question {q_num_str}: {e}. Skipping.")
            skipped_count += 1
            continue

    # --- Save the Output File ---
    print(f"\nProcessed {processed_count} questions successfully.")
    if skipped_count > 0:
        print(f"Skipped {skipped_count} questions due to issues.")

    if not formatted_dataset:
        print("Error: No data was successfully processed. Output file will not be created.")
        return

    try:
        print(f"Saving formatted dataset to: {output_filepath}")
        with open(output_filepath, 'w', encoding='utf-8') as f:
            # Use ensure_ascii=False for correct Persian text, indent for readability
            json.dump(formatted_dataset, f, ensure_ascii=False, indent=2)
        print("Benchmark dataset successfully created.")
    except IOError as e:
        print(f"Error: Could not write output file to '{output_filepath}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred while saving the output file: {e}")


# --- Main Execution ---
if __name__ == "__main__":
    QUESTIONS_JSON_PATH = 'questions.json'
    ANSWERS_JSON_PATH = 'answers.json'
    OUTPUT_JSON_PATH = 'benchmark_dataset.json' # This file will be created

    # Check if input files exist before running
    if not os.path.exists(QUESTIONS_JSON_PATH):
        print(f"Error: Input file not found: {QUESTIONS_JSON_PATH}")
    elif not os.path.exists(ANSWERS_JSON_PATH):
         print(f"Error: Input file not found: {ANSWERS_JSON_PATH}")
    else:
        create_benchmark_dataset(QUESTIONS_JSON_PATH, ANSWERS_JSON_PATH, OUTPUT_JSON_PATH)