# Experiment 6: Persian Poetry Semantic Similarity Benchmark

## Overview
This experiment evaluates Large Language Models (LLMs) on their ability to identify semantic outliers in Persian poetry couplets using the Gherabat dataset.

### Key Features
- **Structured Outputs**: Uses OpenRouter's structured outputs feature to ensure reliable, type-safe responses
- **Zero-shot and Few-shot modes**: Configurable prompting strategies
- **Automatic accuracy calculation**: Compares model responses with ground truth answer keys
- **Comprehensive error handling**: Gracefully handles API errors, rate limits, and unsupported models

## What Changed

### Data Source Migration
- **Old Dataset**: `exp-1/data/zero-shot-questions.json` and `exp-1/data/few-shot-questions.json`
- **New Dataset**: `data/gherabat-book/questions-outliers.json` and `data/gherabat-book/answer_keys.json`

### Key Changes Made

1. **Converted Python Script to Jupyter Notebook**
   - Old: `ask-llm.py` (Python script)
   - New: `persian-poetry-llm-benchmark.ipynb` (Jupyter notebook)

2. **Adapted to New Dataset Structure**

   **Old Format:**
   ```json
   {
     "questions": [
       {
         "question_number": 6,
         "question_text": "کدام گزینه مفهومی متفاوت...",
         "answers": [
           {
             "answer_number": 1,
             "answer_text_1": "...",
             "answer_text_2": "..."
           }
         ]
       }
     ]
   }
   ```

   **New Format:**
   ```json
   {
     "pages": [
       {
         "page_number": 1,
         "data": {
           "questions": [
             {
               "id": 6,
               "stem": "کدام گزینه مفهومی متفاوت...",
               "options": [
                 {
                   "label": 1,
                   "mesra1": "...",
                   "mesra2": "..."
                 }
               ]
             }
           ]
         }
       }
     ]
   }
   ```

3. **Field Mapping**
   - `question_number` → `id`
   - `question_text` → `stem`
   - `answers` → `options`
   - `answer_number` → `label`
   - `answer_text_1` → `mesra2` (note: order swapped)
   - `answer_text_2` → `mesra1` (note: order swapped)

4. **Added Answer Keys Integration**
   - Separated answer keys into `answer_keys.json`
   - Format: `{"1": 1, "2": 4, ...}` (question_id → correct_answer)
   - Notebook now calculates accuracy automatically

5. **Implemented Structured Outputs**
   - Uses OpenRouter's structured outputs feature (JSON Schema validation)
   - Enforces type-safe responses: models must return `{"answer": <1-4>}`
   - Eliminates need for regex-based response parsing
   - Automatically validates responses against schema
   - Gracefully handles models that don't support structured outputs

## Structured Outputs

The notebook uses OpenRouter's structured outputs feature to ensure reliable, type-safe responses. Instead of parsing free-form text, we define a JSON schema that models must follow:

```python
ANSWER_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "poetry_outlier_answer",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "integer",
                    "description": "The number (1, 2, 3, or 4) of the option...",
                    "enum": [1, 2, 3, 4]
                }
            },
            "required": ["answer"],
            "additionalProperties": False
        }
    }
}
```

### Benefits

- **No parsing errors**: Responses are guaranteed to be valid JSON with the correct structure
- **Type safety**: The `answer` field must be an integer between 1 and 4
- **Simplified validation**: No need for regex patterns or string matching
- **Better reliability**: Models can't return ambiguous or malformed responses

### Model Compatibility

Most modern LLMs support structured outputs, including:
- OpenAI models (GPT-4o and later)
- Google Gemini models
- Anthropic Claude models
- Most open-source models via Fireworks

If a model doesn't support structured outputs, the notebook will skip it with an appropriate error message.

## Dataset Statistics

- **Total Questions**: 591 (from questions-outliers.json)
- **Total Answer Keys**: 1,500 (answer_keys.json contains more keys than questions)
- **Used Questions**: 591 (all available questions have answer keys)

## How to Use the Notebook

### Prerequisites
```bash
pip install -r ../requirements.txt
```

### Environment Setup
Create a `.env` file in the project root with:
```
OPENROUTER_API_KEY=your_api_key_here
```

### Running the Experiment

1. **Open the Notebook**
   ```bash
   jupyter notebook persian-poetry-llm-benchmark.ipynb
   ```

2. **Configure Experiment Mode**
   In the "Configuration" cell, set:
   ```python
   EXPERIMENT_MODE = 'zero-shot'  # or 'few-shot'
   ```

3. **Select Models to Test**
   Edit the `MODELS_TO_TEST` list to include desired models

4. **Run All Cells**
   - Execute cells sequentially
   - The experiment will test each model on all 591 questions
   - Results are saved to `experiment-answers.csv`

5. **View Results**
   - Results are automatically analyzed in the final cells
   - Accuracy metrics are calculated per model
   - Visualizations show comparative performance

### Output Files

- `experiment-answers.csv`: Detailed results for each model and question
  - Columns: model_name, question_id, model_answer, correct_answer, is_correct, completion_tokens
- `experiment-results.log`: Detailed execution log

## Notebook Structure

1. **Introduction**: Overview and experiment description
2. **Setup and Imports**: Load dependencies
3. **Configuration**: Set experiment parameters
4. **System Prompts**: Zero-shot and few-shot prompts
5. **Logging Setup**: Configure logging
6. **Helper Functions**: Data loading, prompting, validation
7. **Load Data**: Load questions and answer keys
8. **Initialize Client**: Set up OpenRouter API client
9. **Run Experiment**: Execute the benchmark
10. **Results Analysis**: Calculate and display metrics
11. **Visualization**: Plot performance comparison

## Testing

A test script is provided to verify data loading:

```bash
python test_data_loading.py
```

This will:
- Load and validate the dataset structure
- Show sample question format
- Verify all questions have answer keys

## Notes

- The order of mesra1 and mesra2 is swapped in the new dataset compared to the old format
- The notebook automatically handles the new nested JSON structure
- Answer validation and accuracy calculation are built into the notebook
- Rate limiting and error handling are included for robust API calls

## Legacy Files

The original Python script and data files are preserved:
- `ask-llm.py`: Original experiment script
- `data/`: Original dataset files
- `zero-shot/`: Zero-shot experiment results
- `few-shot/`: Few-shot experiment results
