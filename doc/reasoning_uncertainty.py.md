# `reasoning/uncertainty.py` - Uncertainty Quantification

## Overview
The `UncertaintyCalculator` provides a mathematical lens through which Parag views its own knowledge. Instead of returning a simple confidence score, it provides a multi-faceted analysis of how reliable, complete, and consistent the retrieved information is.

## Metrics

### 1. Retrieval Uncertainty
Calculated from the raw similarity scores of the `RetrievalResult`.
- Factors in the average score and the decay in score across results.

### 2. Information Completeness
Measures whether the system has "enough" information to form a reliable conclusion.
- **`min_units_threshold`**: Configurable minimum number of units (default: 3).

### 3. Conflict-based Uncertainty
Heavily penalizes the overall confidence if the `ConflictDetector` has identified contradictions in the state.

## Core Methods

### `calculate_state_uncertainty(state)`
The primary method. Returns a dictionary with:
- **`total_uncertainty`**: A normalized value [0.0 - 1.0].
- **`breakdown`**: Individual scores for completeness, consistency, and confidence.

### `is_information_sufficient(state, max_uncertainty)`
A boolean check used by generators to decide if they should answer the user or ask for more info. Default threshold is 0.5.

### `detect_missing_information(state)`
Returns a list of human-readable issues, such as "Not enough sources found" or "High conflict detected between sources".

### `get_confidence_breakdown(state)`
Detailed statistical report on fact-level confidence.

## Paradma Integration
When Paradma is available, `UncertaintyCalculator` uses `Axiom` objects to calculate standard deviations and variances across confidence scores. This allows Paradma to "learn" what distribution of scores indicates a reliable retrieval.

## Usage Example
```python
from parag.reasoning import UncertaintyCalculator

calc = UncertaintyCalculator(min_units_threshold=5)

# Analyze the state
metrics = calc.calculate_state_uncertainty(state)

print(f"Overall Uncertainty: {metrics['total_uncertainty']:.2f}")

if not calc.is_information_sufficient(state):
    issues = calc.detect_missing_information(state)
    print("Gaps detected:", issues)
```

## Impact on Generation
The `UncertaintyCalculator` is the gatekeeper for the `DeterministicGenerator` and `PromptBuilder`. A high uncertainty score will:
1. Trigger a "Reflexive" notice in Paradox cognitive loops.
2. Force the assistant to append warnings to its responses.
3. Potentially switch from an LLM response to a "Data Insufficient" report.
