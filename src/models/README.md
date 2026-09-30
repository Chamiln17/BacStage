# Engagement prediction

Training and prediction for the engagement model. The feature pipeline lives in
`src/features/engineer.py` (`VideoFeatureEngineer`); this package trains a model
on top of it and saves both together.

## How training works

1. Videos with zero views are dropped (they have no meaningful engagement score).
2. The target is the engagement score: `log1p((3·comments + likes) / √views)`.
3. The videos are split into train and test: either the test `video_id`s from `--split-file`, or a random 20%.
4. `VideoFeatureEngineer` is fit on the **training split only**. It learns the channel table, subject categories, VIF-selected columns, scaler, TF-IDF vocabulary and embedding PCA.
5. The model is trained on the training split and scored on the test split.
6. Features and model are refit on all videos, and that production bundle is saved.

Fitting on the training split only keeps test videos out of the channel statistics, feature selection and scalers, so the test metrics reflect unseen videos.

## CLI

```bash
# Train (tuned random forest by default); reads cleaned videos + transcripts, writes models/
uv run python run_pipeline.py train
uv run python run_pipeline.py train --model-type catboost --no-arabert
uv run python run_pipeline.py train --split-file splits_mapping.csv   # hold out split=test rows

# Predict for planned videos (JSON object, JSON list, or CSV)
uv run python run_pipeline.py predict --input planned.json
uv run python run_pipeline.py predict --input planned.csv --output predictions.csv
```

`--model-type` accepts `catboost`, `xgboost`, `lightgbm` or `rf`. `--no-arabert` skips the AraBERT embeddings, so no ~500 MB model download is needed.

## Input: a planned video

A planned video is described before it is published, so it has no statistics.

| Field | Required | Used for |
| --- | --- | --- |
| `title` | yes | Lengths, question mark, exam keywords, Bac markers, TF-IDF, embeddings |
| `duration_sec` | yes | Duration; speech rate when a transcript is given |
| `channel_id` or `subject` | one of them | Channel statistics learned in training (medians for an unseen channel); subject one-hot |
| `description` | no | Length, pedagogical markers, TF-IDF, embeddings |
| `tags` | no | Tag count (comma-separated) |
| `publish_date` | no | Weekday flag; defaults to now |
| `transcript_text` | no | Readability, pacing and pedagogical transcript features; `has_transcript` |

```python
from src.models.predict_model import EngagementPredictor

predictor = EngagementPredictor("models")
predictor.predict({
    "title": "مراجعة بكالوريا 2026: الدالة الأسية",
    "duration_sec": 1500,
    "channel_id": "UC...",
})
# -> {"engagement_score": <float>, "engagement_category": "Low" | "Medium" | "High", "known_channel": <bool>}
```

`engagement_category` uses the Low/Medium/High thresholds (33rd and 67th percentiles) of the training scores. `known_channel: False` means the channel was not in the training data, so its channel features are training medians and the score is less reliable.

## Artifacts

| File | Contents |
| --- | --- |
| `model.joblib` | The fitted `VideoFeatureEngineer` and the model, saved together so they cannot drift apart. The AraBERT model itself is not stored; it is loaded again by name. |
| `metadata.json` | Model type, split, test metrics, feature names, engagement thresholds, embedding model name, seed. |

Load only bundles you trained yourself: joblib files can execute code when loaded.
