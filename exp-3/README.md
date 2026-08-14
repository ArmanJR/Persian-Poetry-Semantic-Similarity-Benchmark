# OpenRouter Embedding Models Benchmark

This directory contains scripts for benchmarking embedding models via the OpenRouter API on Persian poetry semantic similarity tasks.

## Overview

The benchmark evaluates various embedding models on their ability to identify semantic outliers in Persian poetry couplets. Each test question contains 4 poetry options, and the task is to identify which one has a different conceptual meaning from the others.

## Features

- **API-based Benchmarking**: Test multiple embedding models through OpenRouter's unified API
- **Smart Caching**: Embeddings are cached locally to save API costs and time
- **Batched Requests**: Up to 32 poetry options are prefetched per API request
- **Rate Limiting**: Built-in rate limiting to avoid API throttling
- **Error Handling**: Robust retry logic for handling API failures
- **Structured Logging**: Per-model and per-question progress and errors
- **Result Merging**: Rerun models are updated without discarding other CSV rows
- **Visualization**: Automated chart generation for results

## Setup

### 1. Verify `uv`

```bash
uv --version
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
uv run --with numpy --with pandas --with requests --with python-dotenv --with scikit-learn openrouter_benchmark.py
```

This will:

1. Load and validate the benchmark dataset (41 questions)
2. Test each embedding model in the list
3. Prefetch and cache embeddings in bounded batches
4. Calculate accuracy for each model
5. Merge rerun model rows into `openrouter_results.csv`

To run only selected models:

```bash
uv run --with numpy --with pandas --with requests --with python-dotenv --with scikit-learn openrouter_benchmark.py --models voyageai/voyage-4-large google/gemini-embedding-2
```

### Visualize Results

```bash
uv run --with pandas --with matplotlib plot_openrouter_results.py
```

This will:

1. Load results from `openrouter_results.csv`
2. Generate a bar chart comparing model performance
3. Save the chart to `openrouter_results.png`
4. Display the chart

## Configuration

### Models to Test

Edit `EXISTING_EMBEDDING_MODELS` or `NEW_EMBEDDING_MODELS` in `openrouter_benchmark.py`. The latest evaluated additions are:

```python
NEW_EMBEDDING_MODELS = (
    "voyageai/voyage-4-lite",
    "voyageai/voyage-4",
    "voyageai/voyage-4-large",
    "nvidia/nemotron-3-embed-1b:free",
    "google/gemini-embedding-2",
    "perplexity/pplx-embed-v1-4b",
    "perplexity/pplx-embed-v1-0.6b",
    "sentence-transformers/paraphrase-minilm-l6-v2",
    "sentence-transformers/all-minilm-l12-v2",
)
```

Browse available models at [https://openrouter.ai/models](https://openrouter.ai/models) (filter by "Embeddings")

### Rate Limiting

Adjust rate limiting in the configuration section:

```python
RATE_LIMIT_DELAY = 0.1
MAX_RETRIES = 5
RETRY_DELAY = 2.0
PREFETCH_BATCH_SIZE = 32
```

### Cache Directory

Embeddings are cached in `openrouter_cache/` by default.

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

1. **Get Embeddings**: Prefetch and cache embeddings for all poetry options
2. **Calculate Similarity**: Compute each option's cosine similarity to the other 3 options
3. **Aggregate Similarity**: Average those three pairwise similarities
4. **Identify Outlier**: Predict the option with the **lowest average** similarity
5. **Compare**: Check if the prediction matches the ground truth

### Evaluation Metric

**Accuracy** = (Correct Predictions / Total Valid Questions) × 100

**Baseline**: Random guessing yields 25% accuracy (1 in 4 chance)

## Cost Considerations

OpenRouter embedding prices vary by model input usage. To minimize costs and latency:

1. **Use Caching**: Embeddings are cached by default (don't delete `openrouter_cache/` unnecessarily)
2. **Test Incrementally**: Start with fewer models to test the setup
3. **Monitor Usage**: Check your usage at [https://openrouter.ai/activity](https://openrouter.ai/activity)
4. **Choose Wisely**: Some models are more expensive than others

Each model embeds 41 questions × 4 options = 164 texts. A normal empty-cache pass groups them into 6 prefetch requests; provider errors may add retries or question-level fallback requests.

## Troubleshooting

### API Key Not Found

```
OPENROUTER_API_KEY is missing
```

**Solution**: Ensure `.env` file exists in the project root with your API key.

### Rate Limiting

```
OpenRouter returned retryable status 429
```

**Solution**: The script handles this automatically. Increase `RATE_LIMIT_DELAY` if persistent.

### Insufficient Credits

```
OpenRouter rejected model MODEL with status 402
```

**Solution**: Add credits to your OpenRouter account at [https://openrouter.ai/credits](https://openrouter.ai/credits)

### Connection Timeout

```
Embedding request for MODEL failed on attempt X/5
```

**Solution**: Check your internet connection. The script will retry automatically.

### Model Not Available

```
OpenRouter rejected model MODEL with status 404
```

**Solution**: The model may not be available or the ID is incorrect. Check available models at [https://openrouter.ai/models](https://openrouter.ai/models)

## Example Results

| New model | Accuracy (%) | Correct | Total |
|-----------|--------------|---------|-------|
| voyageai/voyage-4-large | 36.59 | 15 | 41 |
| google/gemini-embedding-2 | 34.15 | 14 | 41 |
| perplexity/pplx-embed-v1-0.6b | 34.15 | 14 | 41 |
| perplexity/pplx-embed-v1-4b | 31.71 | 13 | 41 |
| sentence-transformers/all-minilm-l12-v2 | 31.71 | 13 | 41 |
| sentence-transformers/paraphrase-minilm-l6-v2 | 31.71 | 13 | 41 |
| voyageai/voyage-4 | 24.39 | 10 | 41 |
| nvidia/nemotron-3-embed-1b:free | 21.95 | 9 | 41 |
| voyageai/voyage-4-lite | 21.95 | 9 | 41 |

The random-selection baseline is 25%.

## Comparison with Local Models

This OpenRouter benchmark complements the local SentenceTransformer benchmark (`benchmark.py`):

| Approach | Pros | Cons |
|----------|------|------|
| **OpenRouter API** | Access to latest models, No GPU required, Easy to use | Costs money, Requires internet, Rate limits |
| **Local Models** | Free, No rate limits, Offline | Requires GPU/RAM, Limited model selection, Setup complexity |

## Contributing

To add more embedding models:

1. Find the model ID on [OpenRouter](https://openrouter.ai/models)
2. Add it to one of the configured embedding-model tuples
3. Run the benchmark
4. Submit results via pull request

## References

- OpenRouter API Documentation: [https://openrouter.ai/docs](https://openrouter.ai/docs)
- OpenRouter Embeddings API: [https://openrouter.ai/docs/api-reference/embeddings](https://openrouter.ai/docs/api-reference/embeddings)
- Available Models: [https://openrouter.ai/models](https://openrouter.ai/models)

## License

See the main repository LICENSE file.
