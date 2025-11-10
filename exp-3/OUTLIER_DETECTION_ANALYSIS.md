# Outlier Detection Algorithm Analysis

## Current vs. Improved Methods

### Problem Statement
Given 4 poetry embeddings where 3 are semantically similar and 1 is different, identify the outlier.

---

## Method Comparison

### 1. **Centroid-Based** (Original Method)

**Algorithm:**
```
For each option i:
  - Calculate centroid of the OTHER 3 options
  - Compute cosine similarity between option i and this centroid
  - Return option with LOWEST similarity
```

**Pros:**
- Intuitive and simple
- Fast computation
- Works well when the 3 similar items are very cohesive

**Cons:**
- Centroid can be skewed if one of the "others" is somewhat different
- Not perfectly symmetric
- Depends on centroid representation quality

**Example Issue:**
If options are: [A, B, C, D] where A, B, C are similar but C is slightly different:
- When testing D: centroid of {A, B, C} may be slightly biased by C
- When testing C: centroid of {A, B, D} is heavily skewed by outlier D

---

### 2. **Pairwise Average** (Recommended Improvement)

**Algorithm:**
```
For each option i:
  - Calculate cosine similarity to EACH of the other 3 options
  - Take the average of these 3 similarities
  - Return option with LOWEST average similarity
```

**Pros:**
- ✅ More robust and symmetric
- ✅ Each option compared to all others equally
- ✅ Less sensitive to individual variations
- ✅ Better theoretical foundation
- ✅ Aligns with intuition: "which is least similar to the group?"

**Cons:**
- Slightly more computation (negligible: 6 comparisons vs 4)

**Why It's Better:**
```python
# Centroid method:
sim(i, mean([j, k, l]))  # Single comparison to averaged representation

# Pairwise method:
mean([sim(i,j), sim(i,k), sim(i,l)])  # Average of actual comparisons
```

The pairwise method is more direct and doesn't rely on centroid representation.

---

### 3. **Maximum Distance Sum**

**Algorithm:**
```
For each option i:
  - Sum cosine distances (1 - similarity) to all others
  - Return option with MAXIMUM total distance
```

**Notes:**
- Mathematically equivalent to pairwise average for ranking
- Just inverts the similarity to distance
- Same results as pairwise_avg but different interpretation

---

### 4. **Euclidean Distance**

**Algorithm:**
```
For each option i:
  - Calculate Euclidean distance to all others
  - Return option with MAXIMUM average distance
```

**When to Use:**
- When embeddings have meaningful magnitudes (not just directions)
- Less common for semantic embeddings which typically use cosine similarity

**Note:**
- For L2-normalized embeddings, Euclidean distance is related to cosine similarity
- Generally, cosine similarity is preferred for text embeddings

---

### 5. **Isolation Score** (Advanced)

**Algorithm:**
```
For each option i:
  - Calculate average similarity among the OTHER 3 options (cohesion)
  - Calculate average similarity from i to the OTHER 3 options
  - Isolation score = cohesion - external_similarity
  - Return option with HIGHEST isolation score
```

**Pros:**
- ✅ Explicitly measures group cohesion
- ✅ Most theoretically sophisticated

**Cons:**
- More complex
- May be overkill for 4 items

**Best for:**
- When you want to maximize the cohesion of the non-outlier group

---

## Normalization

All methods support **L2 normalization** of embeddings before comparison.

**What it does:**
```python
normalized = embedding / ||embedding||₂
```

**When to use:**
- When embedding magnitudes vary significantly
- To focus purely on directional similarity
- Some models benefit from normalization, others don't

**Recommendation:** Test both with and without normalization.

---

## Recommendations

### For Persian Poetry Benchmark:

1. **Primary Recommendation: `pairwise_avg`**
   - Most robust and theoretically sound
   - Best balance of simplicity and accuracy
   - Default in the updated benchmark

2. **Test with and without normalization**
   - Use `compare_outlier_methods.py` to compare
   - Some models may benefit from normalization

3. **If pairwise_avg doesn't improve results:**
   - Try `isolation` method for explicit cohesion measurement
   - Consider that the issue may be with embedding quality, not the algorithm

---

## How to Use

### 1. Run the benchmark with the improved method (default):

```bash
cd exp-3
python openrouter_benchmark.py
```

The benchmark now uses `pairwise_avg` by default.

### 2. Change the method:

Edit `openrouter_benchmark.py`:
```python
OUTLIER_METHOD = 'pairwise_avg'  # or 'centroid', 'isolation', etc.
NORMALIZE_EMBEDDINGS = False     # or True
```

### 3. Compare all methods on cached data:

After running the benchmark once (to generate cache):

```bash
# Test the simple demo
python outlier_detection_methods.py

# Compare methods on actual data
python compare_outlier_methods.py openai/text-embedding-3-small
```

This will show which method performs best for each model.

---

## Expected Improvements

Based on theoretical analysis, you should see:

- **0-5% improvement** in accuracy with `pairwise_avg` over `centroid`
- Improvement more noticeable when:
  - Similar items have moderate variation
  - Outliers are only moderately different (not extremely obvious)
  - Embedding quality is medium (not perfect)

- Less improvement when:
  - Outliers are very obvious (any method works)
  - Similar items are nearly identical (any method works)

---

## Testing Framework

### Quick Test
```bash
python outlier_detection_methods.py
```

### Full Comparison
```bash
python compare_outlier_methods.py [model_name]
```

### Example Output
```
Method                Normalized  Correct  Total  Accuracy (%)
------------------------------------------------------------------
pairwise_avg          No          32       41     78.05
isolation             Yes         31       41     75.61
centroid              No          30       41     73.17
max_distance_sum      Yes         30       41     73.17
euclidean             No          28       41     68.29
```

---

## Conclusion

The **pairwise average similarity** method is recommended because:
1. ✅ More theoretically sound
2. ✅ More robust to variations in the similar group
3. ✅ Symmetric and fair comparison
4. ✅ Minimal computational overhead
5. ✅ Better interpretability

The improvement may be modest (2-5%), but it's the **correct** approach from a mathematical standpoint.
