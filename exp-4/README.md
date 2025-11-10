# Experiment 4: LLM Explanations of Persian Poetry Couplets

This experiment generates interpretations/explanations of Persian poetry couplets using multiple large language models via OpenRouter API.

## Overview

For each poetry couplet (option) in the benchmark dataset, we ask multiple LLMs to provide a simple interpretation of its meaning. This creates an enriched dataset that includes both the original poetry and machine-generated explanations.

## Methodology

1. **Input**: The original benchmark dataset (`preprocess-data/benchmark_dataset.json`)
2. **Process**: For each question in the dataset:
   - Iterate through all 4 poetry options
   - Ask each LLM model to provide a simple interpretation
   - Collect explanations from all models
3. **Output**: Enhanced dataset with explanations (`benchmark_dataset_with_explanations.json`)

## Models Used

The experiment uses the following LLMs via OpenRouter (configurable in the script):
- `moonshotai/kimi-linear-48b-a3b-instruct`
- `moonshotai/kimi-k2-thinking` (thinking model with reasoning)
- `minimax/minimax-m2`
- `anthropic/claude-haiku-4.5`
- `google/gemini-2.5-flash`
- `z-ai/glm-4.6`
- `deepseek/deepseek-chat-v3.1`
- `openai/gpt-oss-120b`
- `x-ai/grok-4`

**Note**: Some of these are "thinking models" that provide chain-of-thought reasoning in addition to the final answer.

## Output Format

The output JSON maintains the original structure but adds an `explanations` field to each question:

```json
{
  "id": 6,
  "options": [...],
  "correct_answer_index": 0,
  "explanations": [
    {
      "option_index": 0,
      "option_text": "طریق عشق پرآشوب و فتنه است ای دل - بیفتد آن که در این راه با شتاب رود",
      "llm_explanations": {
        "openai/gpt-4o-mini": {
          "explanation": "This couplet warns that..."
        },
        "openai/gpt-4o": {
          "explanation": "The verse cautions..."
        },
        "moonshotai/kimi-k2-thinking": {
          "explanation": "راه عشق پر از آشوب و فتنه است...",
          "reasoning": "The user asks for a simple interpretation..."
        }
      }
    },
    ...
  ]
}
```

**Note**: Thinking models (like `kimi-k2-thinking`, `deepseek-r1`, etc.) will include both `explanation` and `reasoning` fields. The `reasoning` field contains the model's chain-of-thought process before generating the final explanation.

## Usage

### Setup

1. Ensure you have your OpenRouter API key in the `.env` file:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

2. Install required dependencies (if not already installed):
   ```bash
   pip install openai python-dotenv
   ```

### Running the Experiment

```bash
cd exp-4
python generate_explanations.py
```

The script will:
- Process each question sequentially
- Save progress after each question (so you can stop and resume)
- Log all activity to `generation.log`
- Output the final dataset to `benchmark_dataset_with_explanations.json`

### Monitoring Progress

- Check the console output for real-time progress
- View detailed logs in `generation.log`
- The script saves after each question, so partial results are always available

## Notes

- **Cost**: This experiment makes API calls to multiple models. Monitor your OpenRouter usage at https://openrouter.ai/activity
- **Performance**: The script uses **concurrent requests** (6 simultaneous by default) to significantly speed up processing
- **Timeout**: Each request has a 150-second timeout. If a model takes longer, the request is automatically skipped and the script moves to the next model
- **Auto-save**: Progress is saved after **every single option** (not just every question), so you never lose more than one option's work
- **Resume capability**: The script automatically detects existing progress and skips already-completed options. You can safely stop and restart anytime!
- **Interruption**: Press Ctrl+C to safely interrupt. All progress is already saved.

### Configuration

You can adjust various settings in the script:

```python
# Performance
MAX_CONCURRENT_REQUESTS = 6  # Number of simultaneous API requests
REQUEST_TIMEOUT = 150  # Maximum seconds per request

# Caching and Resume
ENABLE_RESUME = True  # Skip already-processed options
SAVE_AFTER_EACH_OPTION = True  # Save after each option (recommended)
```

**Speed improvement**: With 13 models and 6 concurrent requests, processing is approximately **2-3x faster** than sequential processing.

### Resume and Recovery

The script has robust crash recovery:

1. **Automatic resume**: Run the script again after a crash - it automatically detects and loads existing progress
2. **Incremental saves**: Progress is saved after each option completes (every ~3-5 minutes)
3. **Smart skipping**: Already-completed options are detected and skipped (shown as ⏭️ SKIPPED in logs)
4. **Merge on restart**: New results are merged with existing ones, so you never lose work

**Example workflow**:
```bash
# First run - processes 10 options then crashes
python generate_explanations.py

# Second run - automatically resumes from option 11
python generate_explanations.py  # Will skip options 1-10

# Output shows: "Resume mode enabled - merging existing progress..."
```

## Applications

The enriched dataset can be used for:
- Understanding how different LLMs interpret Persian poetry
- Comparing interpretation quality across models
- Creating training data for poetry explanation models
- Analyzing semantic understanding capabilities of LLMs for Persian literature
