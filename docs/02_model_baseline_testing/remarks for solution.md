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

## [2026-01-19] Learning Interaction Score (LIS) Update

### Problem with Previous Formula

The old engagement score formula had several issues:
1. **Age factor circularity**: Used `days_since_publish` in the target, which is also used as a feature
2. **Arbitrary weights**: 0.40/0.50/0.10 had no theoretical basis
3. **Low variance**: Most scores clustered around 6-7, making prediction difficult
4. **R² ceiling**: Model stagnated at R² ~0.52 despite TF-IDF + AraBERT features

### New Formula: Learning Interaction Score

For educational videos, **comments** are the strongest proxy for active learning (questions, discussions, clarifications).

```python
# Learning Interaction Score (LIS)
engagement_score = np.log1p(
    (comment_count * 3 + like_count) / np.sqrt(view_count + 1)
)
```

| Component       | Rationale                                                  |
| --------------- | ---------------------------------------------------------- |
| **Comment × 3** | Comments indicate active learning (questions, discussions) |
| **Like × 1**    | Likes indicate passive approval ("this helped")            |
| **√(views)**    | Normalizes by reach, but less aggressively than `/views`   |
| **log1p()**     | Smooths distribution, handles zeros, improves regression   |

### Distribution Comparison

| Metric       | Old Formula             | New Formula (LIS)   |
| ------------ | ----------------------- | ------------------- |
| Mean         | 6.97                    | 2.42                |
| Std          | 1.34                    | 0.74                |
| Min          | 0.67                    | 0.00                |
| Max          | 22.49                   | 5.14                |
| Distribution | Right-skewed, clustered | More normal, spread |

### Expected Improvement

Research shows log-transformed engagement metrics achieve R² 0.65-0.77 on similar tasks. The new formula should break the 0.52 ceiling.

### Changes Made

- Modified `_create_engagement_features()` in [engineer.py](file:///e:/programming/SIC/src/features/engineer.py#L338-342)
- Removed `age_factor` computation entirely
- Re-ran pipeline via `run_pipeline.py engineer`