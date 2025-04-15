import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd # For saving to CSV
import matplotlib.pyplot as plt # For plotting the chart

def find_outlier_index(embeddings):
    """
    Identifies the index of the outlier embedding based on lowest similarity
    to the centroid of the other embeddings.

    Args:
        embeddings: A list or numpy array of 4 embedding vectors.

    Returns:
        The index (0-3) of the predicted outlier embedding, or -1 if prediction fails.
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.array(embeddings)

    if len(embeddings) != 4 or embeddings.ndim != 2:
        print(f"Warning: find_outlier_index received invalid input shape {embeddings.shape if isinstance(embeddings, np.ndarray) else 'N/A'}. Cannot predict.")
        return -1 # Indicate prediction failure

    num_options = len(embeddings)
    similarity_to_others_centroid = []

    for i in range(num_options):
        others_indices = [j for j in range(num_options) if j != i]
        if not others_indices: # Should not happen with 4 options, but good practice
            similarity_to_others_centroid.append(-np.inf) # Assign very low similarity
            continue

        others_embeddings = embeddings[others_indices]
        # Ensure centroid calculation is valid (avoid mean of empty array)
        if others_embeddings.shape[0] == 0:
             similarity_to_others_centroid.append(-np.inf)
             continue

        centroid_others = np.mean(others_embeddings, axis=0, keepdims=True) # keepdims=True maintains 2D shape

        # Ensure embedding[i] is also 2D for cosine_similarity
        current_embedding = embeddings[i].reshape(1, -1)

        similarity = cosine_similarity(current_embedding, centroid_others)[0][0]
        similarity_to_others_centroid.append(similarity)

    # Find index of minimum similarity (most dissimilar to others)
    # Check if we have valid similarities before finding the minimum
    valid_similarities = [s for s in similarity_to_others_centroid if s > -np.inf]
    if not valid_similarities:
        print("Warning: Could not calculate valid similarities for outlier detection.")
        return -1 # Indicate prediction failure

    predicted_outlier_index = np.argmin(similarity_to_others_centroid)
    return predicted_outlier_index

# --- Configuration ---
# Define the models to test
model_names = [
    'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
    'sentence-transformers/LaBSE',
    'sentence-transformers/all-MiniLM-L6-v2',
    'HooshvareLab/bert-fa-zwnj-base',
    'HooshvareLab/bert-base-parsbert-uncased',
    'HooshvareLab/bert-fa-base-uncased',
    'BAAI/bge-m3',
    'Snowflake/snowflake-arctic-embed-l-v2.0',
    'myrkur/sentence-transformer-parsbert-fa',
    'antoinelouis/colbert-xm',
    'PartAI/Tooka-SBERT',
    'PartAI/TookaBERT-Large',
    'intfloat/multilingual-e5-small',
    'intfloat/multilingual-e5-large',
    'Msobhi/Persian_Sentence_Embedding_v3',
]

# List models potentially requiring trust_remote_code
models_requiring_trust = []

# Define the path to your preprocessed dataset
BENCHMARK_DATA_PATH = '../preprocess-data/benchmark_dataset.json'
# Define output file names
RESULTS_CSV_PATH = 'benchmark_results.csv'
CHART_OUTPUT_PATH = 'benchmark_chart.png'

# --- Load the dataset ---
if not os.path.exists(BENCHMARK_DATA_PATH):
    print(f"Error: Benchmark data file not found at '{BENCHMARK_DATA_PATH}'.")
    print("Please run the preprocessing script first.")
    exit()

try:
    with open(BENCHMARK_DATA_PATH, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    print(f"Successfully loaded benchmark dataset from {BENCHMARK_DATA_PATH}")
    if not dataset:
         print("Warning: Loaded dataset is empty. Cannot run benchmark.")
         exit()
except Exception as e:
    print(f"An unexpected error occurred loading the dataset: {e}")
    exit()


# --- Load Models ---
print("\n--- Loading Embedding Models ---")
models = {}
for name in model_names:
    print(f"Loading model: {name}...")
    trust_remote = False
    if name in models_requiring_trust:
        print(f"  Note: Model {name} requires executing remote code.")
        trust_remote = True # Set to True ONLY if you trust the source code
        print("  Setting trust_remote_code=True")

    try:
        models[name] = SentenceTransformer(name, trust_remote_code=trust_remote)
        print(f"Successfully loaded {name}{' (with trust_remote_code=True)' if trust_remote else ''}")
    except Exception as e:
        print(f"Failed to load {name}: {e}. Skipping this model.")
        if name in models: del models[name] # Ensure failed model is removed
    print("-" * 20)

if not models:
    print("Error: No models were successfully loaded. Exiting.")
    exit()

# --- Run Benchmark ---
results = {} # Dictionary to store {model_name: accuracy}
total_questions = len(dataset)

print(f"\n--- Starting Benchmark ({total_questions} questions) ---")

for model_name, model in models.items():
    print(f"\n--- Testing Model: {model_name} ---")
    correct_predictions = 0
    prediction_failures = 0 # Count cases where outlier detection failed or data was bad

    for i, item in enumerate(dataset):
        question_id = item.get('id', f'index_{i}') # Use ID if available, else index
        # print(f"  Processing question {question_id} ({i+1}/{total_questions})...") # Verbose logging

        if 'options' not in item or len(item['options']) != 4:
            # print(f"  Warning: Skipping question ID {question_id} due to missing/invalid options.")
            prediction_failures += 1
            continue
        if 'correct_answer_index' not in item:
            # print(f"  Warning: Skipping question ID {question_id} due to missing correct answer index.")
            prediction_failures += 1
            continue

        options_text = item['options']
        true_outlier_index = item['correct_answer_index']

        # 1. Generate embeddings
        try:
            option_embeddings = model.encode(options_text, show_progress_bar=False)
            # Basic check for embedding quality (e.g., not all zeros, correct shape)
            if not isinstance(option_embeddings, np.ndarray) or option_embeddings.shape != (4, model.get_sentence_embedding_dimension()):
                 raise ValueError(f"Unexpected embedding shape: {option_embeddings.shape if isinstance(option_embeddings, np.ndarray) else 'N/A'}")
        except Exception as e:
            # print(f"  Error encoding options for question ID {question_id} with model {model_name}: {e}. Skipping.")
            prediction_failures += 1
            continue

        # 2. Predict the outlier index
        predicted_outlier_index = find_outlier_index(option_embeddings)

        # 3. Compare prediction with the ground truth
        if predicted_outlier_index == -1: # Check if outlier detection failed
            prediction_failures += 1
            # print(f"  Prediction failed for question ID {question_id}")
        elif predicted_outlier_index == true_outlier_index:
            correct_predictions += 1

    # Calculate accuracy, considering only valid predictions
    effective_total = total_questions - prediction_failures
    if effective_total > 0:
        accuracy = (correct_predictions / effective_total) * 100
        print(f"Model: {model_name} - Accuracy: {accuracy:.2f}% ({correct_predictions}/{effective_total} valid predictions)")
        results[model_name] = accuracy
    else:
        print(f"Model: {model_name} - No valid predictions could be made.")
        results[model_name] = 0.0 # Assign 0 accuracy if no questions could be processed

    if prediction_failures > 0:
        print(f"  ({prediction_failures} questions skipped or failed prediction)")


# --- Process and Save Results ---

print("\n--- Benchmark Results Summary ---")

if not results:
    print("No results generated. Cannot save CSV or create chart.")
else:
    # Convert results to DataFrame for easier handling
    results_df = pd.DataFrame(results.items(), columns=['Model Name', 'Accuracy (%)'])
    # Sort by accuracy (descending)
    results_df = results_df.sort_values(by='Accuracy (%)', ascending=False).reset_index(drop=True)

    print(results_df.to_string(index=False)) # Print formatted table to console

    # --- Save results to CSV ---
    try:
        results_df.to_csv(RESULTS_CSV_PATH, index=False, encoding='utf-8')
        print(f"\nResults successfully saved to: {RESULTS_CSV_PATH}")
    except IOError as e:
        print(f"\nError saving results to CSV file '{RESULTS_CSV_PATH}': {e}")
    except Exception as e:
        print(f"\nAn unexpected error occurred while saving CSV: {e}")


    # --- Generate and Save Bar Chart ---
    print(f"\nGenerating results chart...")
    try:
        plt.figure(figsize=(10, 6)) # Adjust size as needed
        bars = plt.bar(results_df['Model Name'], results_df['Accuracy (%)'], color='skyblue')

        plt.xlabel("Model Name")
        plt.ylabel("Accuracy (%)")
        plt.title("Persian Poetry Outlier Detection Benchmark Results")
        plt.ylim(0, 105) # Set Y-axis limit slightly above 100%
        plt.xticks(rotation=30, ha='right') # Rotate labels to prevent overlap

        # Add accuracy values on top of bars
        for bar in bars:
            yval = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f'{yval:.2f}%', va='bottom', ha='center') # Adjust position slightly

        # Add horizontal line at 25% to indicate random selection
        plt.axhline(y=25, color='red', linestyle='--', linewidth=2, label='Random Selection (25%)')
        plt.legend()

        plt.tight_layout() # Adjust layout to prevent labels being cut off

        # Save the chart to a file
        plt.savefig(CHART_OUTPUT_PATH)
        print(f"Chart successfully saved to: {CHART_OUTPUT_PATH}")

        # Display the chart
        plt.show()

    except Exception as e:
        print(f"\nError generating or saving the chart: {e}")

print("\n--- Benchmark Complete ---")