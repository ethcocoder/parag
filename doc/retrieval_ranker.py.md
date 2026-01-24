# `retrieval/ranker.py` - Result Ranker

## Overview
`ResultRanker` provides advanced logic for refining retrieval results. While standard vector search relies purely on semantic similarity, the `ResultRanker` can adjust scores based on diversity, recency, or custom business logic.

## Ranking Strategies

### 1. Score-Based (Default)
Orders results by their raw cosine similarity score or the metric provided by the vector store.

### 2. Source Diversity
Prevents the top-k from being dominated by a single document.
- **`diversity_weight`**: Higher values penalize repeated sources.
- **`promote_source_diversity(max_per_source)`**: Explicitly limits how many items can come from the same origin (e.g., only 2 chunks per PDF).

### 3. Recency Weighting
Gives a boost to newer information.
- **Mechanism**: Uses an exponential decay function `e^(-age_days / 30)`.
- **`recency_weight`**: Configurable coefficient to determine how much the date affects the final rank.

### 4. Custom Ranking
- **`custom_rank(scoring_fn)`**: Allows providing a lambda or function that takes a `KnowledgeUnit` and its score and returns a new score.

## Core Methods

### `rank(result)`
The main entry point. Re-calculates scores based on the initialized weights and returns a new, sorted `RetrievalResult`.

### `filter_by_confidence(min_confidence)`
Removes results that fall below a specific similarity threshold.

## Usage Example
```python
from parag.retrieval import ResultRanker

# Boost new documents and ensure diversity
ranker = ResultRanker(
    diversity_weight=0.3,
    recency_weight=0.4
)

# Apply to a retrieval result
better_results = ranker.rank(original_result)

# Ensure we don't get 10 chunks from the same manual
diverse_results = ranker.promote_source_diversity(better_results, max_per_source=1)
```

## Implementation Details
- **Decay Logic**: The recency decay is normalized so that today's documents score near 1.0, and 30-day-old documents score ~0.37.
- **Safeguards**: Handles missing timestamps gracefully by assigning a neutral score (0.5).
