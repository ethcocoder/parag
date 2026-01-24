# `retrieval_result.py` - Structured Retrieval Container

## Overview
Defines `RetrievalResult`, a structured container for similarity search outputs. Provides a clean abstraction for retrieved knowledge units with scores, rankings, and statistical analysis.

## Purpose
- **Result Encapsulation**: Clean API for retrieval outputs
- **Score Management**: Tracks similarity/relevance scores for each result
- **Filtering & Ranking**: Methods for top-k selection and threshold filtering
- **Statistical Analysis**: Compute average, max, min scores
- **Result Merging**: Combine results from multiple retrievals

## Key Components

### `RetrievalResult` Class
Dataclass container for retrieval results.

#### Attributes
- **`units`**: List of retrieved `KnowledgeUnit` objects
- **`scores`**: Similarity/relevance scores (one per unit)
- **`query`**: The original query string
- **`query_embedding`**: Embedding vector of the query
- **`metadata`**: Additional retrieval metadata (dict)

### Core Methods

#### Validation
- **`__post_init__()`**: Ensures units and scores lists have same length

#### Size & State
- **`__len__()`**: Returns number of results
- **`is_empty()`**: Returns True if no results

#### Filtering & Selection
- **`get_top_k(k)`**: Returns new RetrievalResult with top k results
- **`filter_by_threshold(threshold)`**: Returns results with score >= threshold

#### Source Tracking
- **`get_sources()`**: Returns list of unique source names from all units

#### Statistical Analysis
- **`get_average_score()`**: Compute mean score (uses Paradma if available!)
- **`get_max_score()`**: Get highest score
- **`get_min_score()`**: Get lowest score
- **`has_high_confidence(threshold)`**: Check if any result >= threshold (default 0.7)

#### Content Access
- **`get_content_list()`**: Returns list of content strings from all units

#### Merging
- **`merge(other)`**: Merge with another RetrievalResult, sort by score descending

#### Serialization
- **`to_dict()`**: Convert to dictionary with stats and metadata
- **`__repr__()`**: Human-readable string representation

## Paradma Integration

### Self-Learning Statistics
The `get_average_score()` method uses Paradma's learning manifold when available:
```python
if PARADMA_AVAILABLE and learning:
    scores_axiom = Axiom(self.scores, manifold=learning)
    mean_result = scores_axiom.mean()  # Paradma learns from this!
    return float(mean_result.value)
```

This means **every statistical operation helps Paradma learn**!

## Usage Examples

### Basic Retrieval Result
```python
from parag import RetrievalResult, KnowledgeUnit

units = [
    KnowledgeUnit(content="Paradma is a self-learning math framework"),
    KnowledgeUnit(content="ParadoxLF uses quantum emotions"),
    KnowledgeUnit(content="Parag is a RAG system"),
]

scores = [0.95, 0.87, 0.76]

result = RetrievalResult(
    units=units,
    scores=scores,
    query="What is Paradma?",
    metadata={"retrieval_time_ms": 42}
)

print(len(result))  # 3
print(result.get_average_score())  # 0.86
```

### Top-K Selection
```python
# Get top 2 results
top_2 = result.get_top_k(2)
print(len(top_2))  # 2
print(top_2.scores)  # [0.95, 0.87]
```

### Threshold Filtering
```python
# Only keep results with score >= 0.8
high_conf = result.filter_by_threshold(0.8)
print(len(high_conf))  # 2
print(high_conf.scores)  # [0.95, 0.87]
```

### Statistical Analysis
```python
print(f"Average: {result.get_average_score():.3f}")  # 0.860
print(f"Max: {result.get_max_score():.3f}")          # 0.950
print(f"Min: {result.get_min_score():.3f}")          # 0.760

if result.has_high_confidence(threshold=0.9):
    print("At least one high-confidence result!")
```

### Source Tracking
```python
# If units have source metadata
sources = result.get_sources()
print(f"Retrieved from: {sources}")
# ['documentation.md', 'wiki_page.txt']
```

### Content Extraction
```python
# Get all content as strings
content_list = result.get_content_list()
for i, content in enumerate(content_list):
    print(f"{i+1}. {content}")
```

### Merging Results
```python
# Merge two retrieval results (same query)
result1 = RetrievalResult(units=[unit1], scores=[0.9], query="test")
result2 = RetrievalResult(units=[unit2], scores=[0.8], query="test")

merged = result1.merge(result2)
print(len(merged))  # 2
print(merged.scores)  # [0.9, 0.8] (sorted descending)
```

### Serialization
```python
# Convert to dict
data = result.to_dict()
print(data.keys())
# ['query', 'num_results', 'units', 'scores', 'average_score', 'sources', 'metadata']

# String representation
print(result)
# RetrievalResult(query='What is Paradma?', num_results=3, avg_score=0.860)
```

## Design Features

### Validation
- **Length Matching**: Raises `ValueError` if units and scores don't match
- **Merge Safety**: Raises `ValueError` if merging results from different queries

### Immutability Pattern
- **New Instance**: `get_top_k()` and `filter_by_threshold()` return **new** `RetrievalResult` objects
- **Original Preserved**: Original result remains unchanged

### Metadata Tracking
Methods that transform results add metadata:
- `get_top_k(k)` adds `{"top_k": k}`
- `filter_by_threshold(t)` adds `{"threshold": t}`
- `merge()` adds `{"merged": True}`

## Dependencies
- **NumPy**: For numerical operations
- **Paradma** (optional): For self-learning statistics via `Axiom`
- **`KnowledgeUnit`**: From `parag.core`

## Error Handling
- Falls back to NumPy if Paradma operations fail
- Handles empty results gracefully (returns 0.0 for statistics)

## Future Enhancements
- Diversity-aware ranking (avoid similar results)
- Multi-query result fusion
- Temporal decay for older results
- Explanation generation (why each result was selected)
