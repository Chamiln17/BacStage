# Strategy 1 Implementation: Feature Combination

## Implementation Date
January 19, 2026

## Overview
Successfully implemented **Strategy 1** from the AraBERT Implementation Guide: Combining TF-IDF + AraBERT + Numerical features.

## Changes Made

### 1. Added TF-IDF Feature Extraction (Cell 11)

```python
# New cell added after AraBERT extraction
- TfidfVectorizer with 1500 max features
- Unigrams and bigrams (ngram_range=(1,2))
- Trained on combined title + description text
- Outputs: 1500 TF-IDF features
```

**Parameters**:
- `max_features=1500`: Top 1500 most discriminative terms
- `min_df=2`: Term must appear in ≥2 documents  
- `max_df=0.8`: Ignore terms in >80% of documents
- `ngram_range=(1,2)`: Capture single words and phrases
- `sublinear_tf=True`: Log scaling for term frequency

### 2. Added Feature Combination (Cell 13)

```python
# Combines all three feature types:
X_train_combined = np.hstack([
    X_train_scaled,      # 14 numerical features
    X_train_tfidf,       # 1500 TF-IDF features  
    X_train_bert         # 768 AraBERT features
])
# Total: 2282 features
```

### 3. Added Documentation (Cell 12)

- Markdown cell explaining the strategy
- Expected results table
- Rationale based on research

### 4. Updated Test Results Tracking

Modified test_results dictionary to track:
```python
'features': {
    'numerical': 14,
    'tfidf': 1500,
    'arabert': 768,
    'total': 2282,
    'strategy': 'combined_all'
}
```

## Feature Breakdown

| Feature Type | Dimensions | Purpose |
|-------------|------------|---------|
| Numerical | 14 | Duration, views, likes, engagement metrics |
| TF-IDF | 1500 | Discriminative keywords (task-specific) |
| AraBERT | 768 | Semantic understanding (context-aware) |
| **Total** | **2282** | **Combined strengths** |

## Expected vs Baseline Performance

### Baseline (Before Strategy 1)

| Model | Features | R² | MAE | RMSE |
|-------|----------|-----|------|------|
| XGBoost | 1514 (TF-IDF only) | 0.4617 | 0.6255 | 1.0152 |
| LightGBM | 782 (AraBERT only) | 0.4296 | 0.6725 | 1.0451 |

### Expected (After Strategy 1)

| Model | Features | Expected R² | Expected MAE | Expected RMSE |
|-------|----------|-------------|--------------|---------------|
| XGBoost/LightGBM | 2282 (Combined) | **0.48-0.52** | **0.58-0.62** | **0.98-1.02** |

**Expected improvement**: +4-13% in R² score

## Why This Works

### 1. Complementary Strengths
- **TF-IDF**: Captures specific keywords ("بكالوريا", "رياضيات", subject names)
- **AraBERT**: Understands semantic relationships and context
- **Numerical**: Engagement patterns and video characteristics

### 2. Research Support
From `arabert_implementation_guide.md`:
- Study on Arabic short answer grading: Combination beats individual features
- LLM embeddings research: Task-specific + general features = best performance
- AraBERT works best when combined, not in isolation

### 3. Reduces Individual Weaknesses
- TF-IDF alone: Misses semantic relationships
- AraBERT alone: Too general, lacks discriminative power
- Numerical alone: Can't capture content meaning

## Next Steps

### Phase 1: Test Current Implementation ✅
1. Run notebook cells 11-13 to extract and combine features
2. Train models with combined features (cells 14+)
3. Compare results with baseline

### Phase 2: If Results Meet Expectations (R² > 0.48)
- Document success
- Analyze feature importances
- Identify which feature type contributes most

### Phase 3: If Results Don't Meet Expectations (R² < 0.48)
Move to **Strategy 2** or **Strategy 3** from guide:
- Strategy 2: Change AraBERT pooling (mean instead of CLS)
- Strategy 3: Apply dimensionality reduction (PCA on AraBERT)
- Strategy 4: Feature selection (SelectKBest)

## Implementation Notes

### Cell Execution Order
1. Cell 9: Extract AraBERT embeddings (X_train_bert)
2. **Cell 11 (NEW)**: Extract TF-IDF features (X_train_tfidf)
3. **Cell 12 (NEW)**: Documentation
4. **Cell 13 (NEW)**: Combine all features (X_train_combined)
5. Cell 14+: Train models with combined features

### Variables Created
- `X_train_tfidf`, `X_val_tfidf`, `X_test_tfidf`: TF-IDF features (1500 dims)
- `X_train_combined`, `X_val_combined`, `X_test_combined`: All features (2282 dims)
- `X_train_with_text` updated to use combined features (for model training)

### Files Modified
- `notebooks/02_model_baseline_testing.ipynb`:
  - 3 new cells added (11, 12, 13)
  - 1 cell modified (21 - test results tracking)

## Testing Instructions

### Run the Implementation
```python
# In Jupyter notebook:
1. Run Cell 11: TF-IDF extraction (~5-10 seconds)
2. Run Cell 12: Documentation (instant)
3. Run Cell 13: Feature combination (~1 second)
4. Run Cell 14+: Model training (~2-3 minutes on GPU)
```

### Verify Success
Check that:
- TF-IDF extraction completes: "✅ TF-IDF features extracted"
- Feature combination succeeds: "✅ Feature combination complete: 2282 dims"
- Model training uses combined features
- Results improve over baseline (R² > 0.46)

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Implementation complete | ✅ | ✅ Done |
| Code executes without errors | ✅ | ⏳ To test |
| R² > 0.48 | ✅ | ⏳ To measure |
| MAE < 0.62 | ✅ | ⏳ To measure |
| Improvement > 5% | ✅ | ⏳ To verify |

## References

- **Guide**: `docs/02_model_baseline_testing/arabert_implementation_guide.md`
- **Research**: Section "Why AraBERT Alone Underperforms"
- **Baseline Results**: `data/modeling/baseline_results.csv`
- **Previous Test**: `data/modeling/test_results.json`

---

## Appendix: Code Snippets

### TF-IDF Extraction (Cell 11)
```python
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(
    max_features=1500,
    min_df=2,
    max_df=0.8,
    ngram_range=(1, 2),
    sublinear_tf=True
)

X_train_tfidf = tfidf.fit_transform(train_text_combined).toarray()
X_val_tfidf = tfidf.transform(val_text_combined).toarray()
X_test_tfidf = tfidf.transform(test_text_combined).toarray()
```

### Feature Combination (Cell 13)
```python
X_train_combined = np.hstack([
    X_train_scaled,      # 14 dims
    X_train_tfidf,       # 1500 dims
    X_train_bert         # 768 dims
])
# Total: 2282 dims
```

### Result Tracking (Cell 21 - Modified)
```python
'features': {
    'numerical': 14,
    'tfidf': 1500,
    'arabert': 768,
    'total': 2282,
    'strategy': 'combined_all'
}
```
