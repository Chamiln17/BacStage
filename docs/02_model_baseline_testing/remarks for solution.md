# recomendations from mentors
- each feature must be mesearable 
- it needs a script to extract the features from the video directly 
- the content must be features not the dates
- If the video is published : they give the url of video 
  - else they gave access to draft video  could be perspective
- Video characteristics should be extracted directly (by upload)
- some models characteriscts automiatically
- engagment rate what does mean (in this case type of videos, learning because of this catergory is valuable because you need effort and type of people is different than other type of videos.)
- sometimes doesn't mean a channel 


# idea
we should focus on extracting content related features from the video (either by uploading the video inside our system in production or by scraping for training because post posting , the content crearor)

i got some remarks from the mentors about the idea. i want you to undersand it and improve it since the solution should focus more on pre engagemnt rate prediction. do comprehensive search

### MVP Metric: Enhanced Weighted Engagement Score

Based on the provided document, the recommended Minimum Viable Product (MVP) metric for labeling educational video engagement is the **Enhanced Weighted Engagement Score**.

#### The Formula

$$\text{Engagement Score} = \left( 0.40 \times \frac{\text{Likes}}{\text{Views}} + 0.50 \times \frac{\text{Comments}}{\text{Views}} + 0.10 \times \text{Age Factor} \right) \times 100$$

#### Component Definitions

* 
**Comments/Views (50% Weight):** This is weighted heavily because comments are the strongest signal for educational engagement, indicating active learning and cognitive processing.


* 
**Likes/Views (40% Weight):** Serves as a signal of general approval.


* 
**Age Factor (10% Weight):** Normalizes the score to prevent recency bias or the domination of old viral videos. It is calculated as:


$$\text{Age Factor} = \min\left(1.0, \frac{\text{Days Since Publish}}{365}\right)$$





#### Output

This formula generates a continuous score on a **0–100 scale**, making it interpretable and suitable for regression models.

---

## [2026-01-19] Learning Interaction Rate (LIR) Update

### Problem with Previous Formula

The old engagement score formula had several issues:
1. **Age factor circularity**: Used `days_since_publish` in the target, which is also used as a feature
2. **Arbitrary weights**: 0.40/0.50/0.10 had no theoretical basis
3. **Low variance**: Most scores clustered around 6-7, making prediction difficult
4. **R² ceiling**: Model stagnated at R² ~0.52 despite TF-IDF + AraBERT features

### New Formula: Learning Interaction Rate (Percentage)

For educational videos, **comments** are the strongest proxy for active learning (questions, discussions, clarifications). The formula outputs an **interpretable percentage** for non-technical stakeholders.

```python
# Learning Interaction Rate (LIR) - Interpretable percentage
engagement_score = (comment_count * 3 + like_count) / view_count * 100
```

### How to Explain to Stakeholders

> "This video has a **6.4% learning interaction rate** - meaning for every 100 views, it receives about 6 weighted interactions (with comments counting 3× more than likes because they indicate active learning)."

| Component         | Rationale                                                  |
| ----------------- | ---------------------------------------------------------- |
| **Comment × 3**   | Comments indicate active learning (questions, discussions) |
| **Like × 1**      | Likes indicate passive approval ("this helped")            |
| **/ views × 100** | Converts to percentage for interpretability                |

### Distribution Comparison

| Metric         | Old Formula     | New Formula (LIR)     |
| -------------- | --------------- | --------------------- |
| Mean           | 6.97            | 6.42%                 |
| Std            | 1.34            | 4.06%                 |
| Min            | 0.67            | 0.00%                 |
| Max            | 22.49           | 74.41%                |
| Interpretation | Arbitrary score | "X% interaction rate" |

### Changes Made

- Modified `_create_engagement_features()` in [engineer.py](file:///e:/programming/SIC/src/features/engineer.py#L335-345)
- Removed `age_factor` computation entirely
- Changed from log-transformed to simple percentage for stakeholder interpretability
- Re-ran pipeline via `run_pipeline.py engineer`

---

## [2026-01-19] TransformedTargetRegressor Implementation

### Problem with Simple Percentage

The simple percentage formula (R² ~0.48) performed worse than log-transformed (R² ~0.52+) because:
- High variance and outliers (max 74%)
- Skewed distribution harder to predict

### Solution: Hybrid Approach

Train on log-transformed data (better R²), but display predictions as interpretable values for stakeholders.

### engineer.py Formula (Log-Transformed for Training)

```python
# In _create_engagement_features()
weighted_interaction = (comment_count * 3 + like_count) / np.sqrt(views)
df["engagement_score"] = np.log1p(weighted_interaction)  # Log-transformed for training
df["weighted_interaction_raw"] = weighted_interaction    # Raw value for display
```

### Notebook Code: TransformedTargetRegressor

Add this in your model training cell:

```python
from sklearn.compose import TransformedTargetRegressor
from lightgbm import LGBMRegressor
import numpy as np

# Wrap LightGBM with TransformedTargetRegressor
# Model trains on log-scale internally, predictions are auto-converted back
model = TransformedTargetRegressor(
    regressor=LGBMRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbose=-1
    ),
    func=np.log1p,      # Transform target before training
    inverse_func=np.expm1  # Inverse transform predictions
)

# Train as usual
model.fit(X_train_combined, y_train)

# Predictions are automatically in original scale!
y_pred = model.predict(X_test_combined)
```

### Converting Predictions for Stakeholders

```python
# The predictions are in weighted_interaction scale
# To explain to stakeholders, you can convert to approximate % 
# or use categories:

def interpret_prediction(pred, view_count=1000):
    """Convert prediction to interpretable format for stakeholders."""
    # Approximate interaction count for a video with N views
    approx_interactions = pred * np.sqrt(view_count)
    likes = approx_interactions / 4  # rough split (1 like + 3*0 comments = 1)
    comments = approx_interactions / 4 / 3
    
    if pred < 1.5:
        category = "Low Engagement"
    elif pred < 3.0:
        category = "Medium Engagement"
    else:
        category = "High Engagement"
    
    return {
        "predicted_score": round(pred, 2),
        "category": category,
        "explanation": f"Expected ~{int(likes)} likes and ~{int(comments)} comments per 1000 views"
    }
```