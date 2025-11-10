# OpenRouter Embedding Models Benchmark

This directory contains scripts for benchmarking embedding models via the OpenRouter API on Persian poetry semantic similarity tasks.

## Overview

The benchmark evaluates various embedding models on their ability to identify semantic outliers in Persian poetry couplets. Each test question contains 4 poetry options, and the task is to identify which one has a different conceptual meaning from the others.

## Features

- **API-based Benchmarking**: Test multiple embedding models through OpenRouter's unified API
- **Smart Caching**: Embeddings are cached locally to save API costs and time
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Error Handling**: Robust retry logic for handling API failures
- **Progress Tracking**: Real-time progress updates during benchmarking
- **Visualization**: Automated chart generation for results

## Setup

### 1. Install Dependencies

```bash
# Create virtual environment (if not already done)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install required packages
pip install -r ../requirements.txt
```

### 2. Configure API Key

Get your OpenRouter API key from [https://openrouter.ai/keys](https://openrouter.ai/keys)

Create a `.env` file in the project root:

```bash
# Copy the example file
cp ../.env.example ../.env

# Edit and add your API key
OPENROUTER_API_KEY=your_actual_api_key_here
```

### 3. Verify Dataset

Ensure the benchmark dataset exists:
```bash
ls -la ../preprocess-data/benchmark_dataset.json
```

## Usage

### Run the Benchmark

```bash
python openrouter_benchmark.py
```

This will:
1. Load the benchmark dataset (42 questions)
2. Test each embedding model in the list
3. Cache embeddings for reuse
4. Calculate accuracy for each model
5. Save results to `openrouter_results.csv`

### Visualize Results

```bash
python plot_openrouter_results.py
```

This will:
1. Load results from `openrouter_results.csv`
2. Generate a bar chart comparing model performance
3. Save the chart to `openrouter_results.png`
4. Display the chart

## Configuration

### Models to Test

Edit `EMBEDDING_MODELS` list in `openrouter_benchmark.py`:

```python
EMBEDDING_MODELS = [
    'openai/text-embedding-3-small',
    'openai/text-embedding-3-large',
    'openai/text-embedding-ada-002',
    'google/text-embedding-004',
    'cohere/embed-english-v3.0',
    'cohere/embed-multilingual-v3.0',
    'voyage/voyage-3',
    'voyage/voyage-3-lite',
]
```

Browse available models at [https://openrouter.ai/models](https://openrouter.ai/models) (filter by "Embeddings")

### Rate Limiting

Adjust rate limiting in the configuration section:

```python
RATE_LIMIT_DELAY = 0.5  # seconds between requests
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
```

### Cache Directory

Embeddings are cached in `openrouter_cache/` by default. To change:

```python
CACHE_DIR = 'your_custom_cache_directory'
```

To clear cache and re-run:
```bash
rm -rf openrouter_cache/
```

## Output Files

- **`openrouter_results.csv`**: Detailed results with accuracy, correct predictions, and totals
- **`openrouter_results.png`**: Bar chart visualization of model performance
- **`openrouter_cache/`**: Cached embeddings (JSON files)

## How It Works

### Outlier Detection Algorithm

For each question with 4 options:

1. **Get Embeddings**: Request embeddings for all 4 poetry options
2. **Calculate Similarity**: For each option:
   - Calculate the centroid of the other 3 options
   - Compute cosine similarity between the option and the centroid
3. **Identify Outlier**: The option with the **lowest** similarity score is predicted as the outlier
4. **Compare**: Check if the prediction matches the ground truth

### Evaluation Metric

**Accuracy** = (Correct Predictions / Total Valid Questions) × 100

**Baseline**: Random guessing yields 25% accuracy (1 in 4 chance)

## Cost Considerations

OpenRouter charges per API request. To minimize costs:

1. **Use Caching**: Embeddings are cached by default (don't delete `openrouter_cache/` unnecessarily)
2. **Test Incrementally**: Start with fewer models to test the setup
3. **Monitor Usage**: Check your usage at [https://openrouter.ai/activity](https://openrouter.ai/activity)
4. **Choose Wisely**: Some models are more expensive than others

Estimated cost per model: 42 questions × 4 options = 168 embedding requests

## Troubleshooting

### API Key Not Found

```
Error: OPENROUTER_API_KEY not found!
```

**Solution**: Ensure `.env` file exists in the project root with your API key.

### Rate Limiting

```
Rate limited. Waiting Xs before retry...
```

**Solution**: The script handles this automatically. Increase `RATE_LIMIT_DELAY` if persistent.

### Insufficient Credits

```
Error: Insufficient credits for model X
```

**Solution**: Add credits to your OpenRouter account at [https://openrouter.ai/credits](https://openrouter.ai/credits)

### Connection Timeout

```
Timeout on attempt X/3
```

**Solution**: Check your internet connection. The script will retry automatically.

### Model Not Available

```
Error: API returned status 404
```

**Solution**: The model may not be available or the ID is incorrect. Check available models at [https://openrouter.ai/models](https://openrouter.ai/models)

## Example Results

```
===========================================
Model Name                               Accuracy (%)
-------------------------------------------
openai/text-embedding-3-large                 78.57
cohere/embed-multilingual-v3.0                 71.43
openai/text-embedding-3-small                  69.05
google/text-embedding-004                      66.67
openai/text-embedding-ada-002                  64.29
===========================================

Best performing model: openai/text-embedding-3-large
Accuracy: 78.57%

Baseline (random selection): 25.00%
```

## Comparison with Local Models

This OpenRouter benchmark complements the local SentenceTransformer benchmark (`benchmark.py`):

| Approach | Pros | Cons |
|----------|------|------|
| **OpenRouter API** | Access to latest models, No GPU required, Easy to use | Costs money, Requires internet, Rate limits |
| **Local Models** | Free, No rate limits, Offline | Requires GPU/RAM, Limited model selection, Setup complexity |

## Contributing

To add more embedding models:

1. Find the model ID on [OpenRouter](https://openrouter.ai/models)
2. Add to `EMBEDDING_MODELS` list
3. Run the benchmark
4. Submit results via pull request

## References

- OpenRouter API Documentation: [https://openrouter.ai/docs](https://openrouter.ai/docs)
- OpenRouter Embeddings API: [https://openrouter.ai/docs/api-reference/embeddings](https://openrouter.ai/docs/api-reference/embeddings)
- Available Models: [https://openrouter.ai/models](https://openrouter.ai/models)

## License

See the main repository LICENSE file.
