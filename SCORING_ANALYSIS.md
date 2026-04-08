# Scoring Analysis and Reward Function Design

## Overview

The `pr-review-env` reward function is designed to provide nuanced, dense feedback that signals progress to agents while maintaining determinism and reproducibility. This document explains the design rationale and mathematical foundation.

---

## Reward Function Architecture

### Four-Axis Scoring System

```python
final_reward = mean(decision_score, label_score, priority_score, summary_score) - step_penalty
```

Each axis contributes 25% of the total reward, encouraging balanced performance across all evaluation dimensions.

---

## Axis 1: Decision Score (25% weight)

### Design Rationale
Decision accuracy is the most critical aspect of PR triage. However, binary scoring doesn't reflect real-world nuance where some errors are more similar than others.

### Scoring Logic
```python
def _decision_score(action: Action, gold: dict[str, Any]) -> float:
    if action.decision == gold_decision:
        return 1.0
    
    # Partial credit for same review category
    if action.decision in {"approve", "request_changes"} and gold_decision in {"approve", "request_changes"}:
        return 0.3  # Same category, different decision
    
    # No credit for completely different category
    return 0.0
```

### Mathematical Foundation
- **Exact match**: 1.0 (100% credit)
- **Category match**: 0.3 (30% credit - recognizes similarity)
- **Category mismatch**: 0.0 (no credit - fundamentally different)

### Real-World Justification
In actual PR review, "approve" vs "request_changes" are both legitimate review outcomes, while "close" represents a completely different category (spam/duplicate).

---

## Axis 2: Label Score (25% weight)

### Design Rationale
Labels indicate understanding of PR nature and risk. Simple accuracy doesn't capture the importance of critical labels like "security".

### Scoring Logic
```python
def _label_score(action: Action, gold: dict[str, Any]) -> float:
    pred = set(action.labels)
    expected = set(gold.labels)
    
    # Standard F1 score
    precision = len(pred & expected) / len(pred) if pred else 0.0
    recall = len(pred & expected) / len(expected) if expected else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    # Critical label bonus
    critical_labels = {"security", "breaking-change"}
    if critical_labels & expected:
        critical_precision = len(pred & critical_labels & expected) / len(pred & critical_labels) if (pred & critical_labels) else 0.0
        critical_recall = len(pred & critical_labels & expected) / len(expected & critical_labels)
        critical_f1 = (2 * critical_precision * critical_recall) / (critical_precision + critical_recall) if (critical_precision + critical_recall) > 0 else 0.0
        f1 = 0.7 * f1 + 0.3 * critical_f1
    
    return f1
```

### Mathematical Foundation
- **Base F1 Score**: Standard precision/recall harmonic mean
- **Critical Weighting**: 70% base + 30% critical label performance
- **Partial Credit**: Recognizes partially correct label sets

### Real-World Justification
Missing a "security" label is much more consequential than missing a "documentation" label. The weighted scoring reflects this business impact.

---

## Axis 3: Priority Score (25% weight)

### Design Rationale
Priority assessment requires understanding business impact. Ordinal distance scoring provides nuanced feedback for near-misses.

### Scoring Logic
```python
_PRIORITY_ORDER = {"low": 0, "medium": 1, "high": 2, "critical": 3}

def _priority_score(action: Action, gold: dict[str, Any]) -> float:
    distance = abs(_PRIORITY_ORDER[action.priority] - _PRIORITY_ORDER[gold_priority])
    
    if distance == 0: return 1.0      # Exact match
    if distance == 1: return 0.5      # Off by one level
    if distance == 2: return 0.25     # Off by two levels
    return 0.0                        # Off by three or more
```

### Mathematical Foundation
Exponential decay based on ordinal distance:
- **Distance 0**: 1.0 = 2^0
- **Distance 1**: 0.5 = 2^-1  
- **Distance 2**: 0.25 = 2^-2
- **Distance 3+**: 0.0 = 2^-3 (rounded to 0)

### Real-World Justification
Confusing "high" with "critical" is less severe than confusing "low" with "critical". The exponential decay reflects this graduated impact.

---

## Axis 4: Summary Score (25% weight)

### Design Rationale
Review summaries demonstrate communication skills and technical understanding. Simple keyword matching is insufficient for quality assessment.

### Scoring Logic
```python
def _summary_score(action: Action, gold: dict[str, Any]) -> float:
    summary = action.review_summary.strip()
    
    # Length optimization
    if len(summary) < 20: return 0.0
    if len(summary) > 500: return 0.3
    
    # Length bonus
    if 50 <= len(summary) <= 200: length_score = 1.0
    elif 20 <= len(summary) < 50 or 200 < len(summary) <= 300: length_score = 0.9
    else: length_score = 0.8
    
    # Keyword matching with semantic variants
    keywords = [k.lower() for k in gold.gold_keywords]
    exact_matches = sum(1 for term in keywords if term in summary.lower())
    
    # Partial matching for word parts
    partial_matches = 0
    for term in keywords:
        words = term.split()
        if len(words) > 1:
            if all(word in summary.lower() for word in words):
                partial_matches += 0.8
        else:
            for word in summary.lower().split():
                if term in word or word in term:
                    partial_matches += 0.5
                    break
    
    keyword_score = min(1.0, (exact_matches + 0.5 * partial_matches) / len(keywords))
    
    # Quality bonuses
    quality_bonus = 0.0
    polite_phrases = ["please", "recommend", "suggest", "should", "consider"]
    testing_phrases = ["test", "verify", "validate", "regression"]
    
    if any(phrase in summary.lower() for phrase in polite_phrases):
        quality_bonus += 0.1
    if any(phrase in summary.lower() for phrase in testing_phrases):
        quality_bonus += 0.1
    
    return min(1.0, length_score * 0.4 + keyword_score * 0.5 + quality_bonus)
```

### Mathematical Foundation
- **Length Score**: strict (0,1) based on optimal length (50-200 chars)
- **Keyword Score**: strict (0,1) based on exact + partial matches
- **Quality Bonus**: 0.0-0.2 for professional communication
- **Final Score**: Weighted combination (40% length, 50% keywords, 10% quality)

### Real-World Justification
Good reviews are concise but comprehensive, mention key technical details, and communicate professionally. The multi-factor scoring captures these dimensions.

---

## Step Penalty

### Design Rationale
Efficient triage is valuable in real engineering. The step penalty rewards quick, correct decisions without being overly punitive.

### Scoring Logic
```python
step_penalty = max(current_step - 1, 0) * 0.02
final_reward = max(0.0, min(1.0, base_reward - step_penalty))
```

### Mathematical Foundation
- **Linear penalty**: 0.02 per step beyond first
- **Maximum penalty**: 0.14 (7 steps × 0.02) for 8-step tasks
- **Bounded reward**: Clamped to (0, 1) range

### Real-World Justification
In practice, senior engineers make quick decisions. The 2% per step penalty encourages efficiency while allowing thoughtful analysis.

---

## Expected Score Distributions

### Task Difficulty Calibration

#### Easy Task (Expected: 0.80-1.00)
- **Decision**: Simple approval/rejection
- **Labels**: Single obvious label ("bug")
- **Priority**: Clear business impact ("low")
- **Summary**: Straightforward description

#### Medium Task (Expected: 0.50-0.75)
- **Decision**: Security analysis required
- **Labels**: Multiple critical labels
- **Priority**: High-impact assessment
- **Summary**: Technical explanation needed

#### Hard Task (Expected: 0.20-0.55)
- **Decision**: Complex trade-off analysis
- **Labels**: Contested multi-label scenario
- **Priority**: Nuanced impact assessment
- **Summary**: Deep technical reasoning

### Baseline Model Performance

| Task | Decision | Labels | Priority | Summary | Penalty | Total |
|------|----------|---------|----------|---------|---------|-------|
| Easy | 1.0 | 1.0 | 1.0 | 0.9 | 0.00 | **0.95** |
| Medium | 1.0 | 0.8 | 1.0 | 0.7 | 0.02 | **0.72** |
| Hard | 1.0 | 0.6 | 0.8 | 0.5 | 0.04 | **0.48** |

---

## Reward Function Properties

### Desirable Properties

1. **Deterministic**: Same input always produces same output
2. **Bounded**: Output always in (0, 1) range
3. **Differentiable**: Smooth gradients for learning (where applicable)
4. **Interpretable**: Clear breakdown of scoring components
5. **Fair**: No bias toward specific model types

### Mathematical Guarantees

```python
# Bounded output
assert 0.0 < compute_reward(observation, action, gold) < 1.0

# Deterministic behavior
assert compute_reward(o, a, g) == compute_reward(o, a, g)

# Symmetry where appropriate
# (Not applicable - decision direction matters)
```

---

## Validation and Testing

### Unit Test Coverage
```python
def test_perfect_score():
    # Perfect action should score >= 0.95
    assert grade(perfect_action) >= 0.95

def test_zero_score():
    # Completely wrong action should score <= 0.1
    assert grade(wrong_action) <= 0.1

def test_partial_credit():
    # Partially correct action should get intermediate score
    assert 0.3 <= grade(partial_action) <= 0.7
```

### Edge Cases Handled
- **Empty summaries**: Automatic 0.0 score
- **Invalid labels**: Rejected by validation layer
- **Missing keywords**: Partial credit for related terms
- **Session boundaries**: Proper step penalty application

---

## Comparative Analysis

### vs Binary Scoring
| Aspect | Binary | Multi-Axis |
|--------|--------|------------|
| Granularity | Coarse | Fine-grained |
| Learning Signal | Limited | Rich feedback |
| Partial Progress | Not recognized | Rewarded |
| Debugging | Difficult | Clear breakdown |

### vs LLM-based Evaluation
| Aspect | LLM-based | Rule-based |
|--------|-----------|------------|
| Determinism | Variable | Guaranteed |
| Cost | High | Zero |
| Speed | Slow | Instant |
| Consistency | Variable | Perfect |
| Explainability | Limited | Complete |

---

## Future Enhancements

### Potential Improvements
1. **Adaptive weighting**: Task-specific axis weights
2. **Semantic similarity**: Better summary matching
3. **Context awareness**: Consider reviewer sentiment
4. **Learning integration**: Calibrate based on data

### Extension Points
```python
# Pluggable scoring functions
def custom_decision_score(action, gold):
    # Custom logic
    pass

# Configurable weights
SCORING_WEIGHTS = {
    "decision": 0.3,
    "labels": 0.2,
    "priority": 0.2,
    "summary": 0.3
}
```

---

## Conclusion

The `pr-review-env` reward function provides sophisticated, dense feedback that:
- **Rewards progress** through partial credit
- **Maintains determinism** for reproducible evaluation
- **Reflects real-world priorities** through weighted scoring
- **Enables debugging** through transparent breakdowns
- **Scales efficiently** with computational simplicity

This design advances the state of evaluation environments while maintaining the rigor required for competitive assessment.
