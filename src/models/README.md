# Engagement Prediction Models

This module handles the training and inference for the YouTube Engagement Prediction component of the pipeline.

## 🚀 CLI Usage

You can access the model functions through the main `run_pipeline.py` CLI.

### Training

Train a new model using the engineered features.

```bash
# Standard training (CatBoost default)
uv run python run_pipeline.py train

# Train specific model type
uv run python run_pipeline.py train --model-type individual --output-dir models/v1
# Supported types: catboost, xgboost, lightgbm, rf

# Advanced options
uv run python run_pipeline.py train --tune --n-trials 50  # Enable Optuna tuning
uv run python run_pipeline.py train --cv-folds 5          # 5-fold Cross-validation
uv run python run_pipeline.py train --no-arabert          # Skip AraBERT (faster)
```

### Prediction

Generate engagement scores for new videos.

**Input Format (JSON or CSV):**
Your input file must contain at least: `title`, `description`, `duration_sec`, `view_count`, `like_count`.

```bash
# Predict from JSON file
uv run python run_pipeline.py predict --input new_videos.json --output predictions.json

# Predict from CSV file
uv run python run_pipeline.py predict --input test_videos.csv --output results.csv --format csv
```

## 💻 Python API Usage

For integration into other applications (e.g., Streamlit, RAG), use the `EngagementPredictor` class.

```python
from src.models.predict_model import EngagementPredictor

# Initialize (loads model artifacts from 'models/' directory by default)
predictor = EngagementPredictor(model_dir="models")

# Predict for a single video
video_data = {
    "title": "Revision Bac 2024 Math - Functions",
    "description": "Comprehensive review of exponential functions...",
    "duration_sec": 1200,
    "view_count": 0,    # 0 for new/unseen videos
    "like_count": 0,
    "channel_id": "UC123..." # Optional
}

result = predictor.predict(video_data)

print(f"Engagement Score: {result['engagement_score']:.2f}")
print(f"Metric: {result['predicted_metric']:.2f}")
```

## 🔍 Input Data Specification

The model expects a dictionary or JSON object with the following fields. The `EngagementPredictor` uses `src/features/engineer.py` to transform these raw inputs into the complex features used by the model.

### Required Fields

| Field          | Type  | Description        | Feature Extraction Impact                                                                                                                                                                                                        |
| -------------- | ----- | ------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `title`        | `str` | Video title        | • **Text**: TF-IDF vectorization, AraBERT embeddings<br>• **Meta**: `title_length`, `title_word_count`, `is_title_question` ("?" detection)<br>• **Keywords**: `is_exam_focused` (bac, exam, revision), `exam_keyword_intensity` |
| `description`  | `str` | Video description  | • **Text**: Combined with title for TF-IDF/AraBERT<br>• **Meta**: `description_length`<br>• **Keywords**: `desc_pedagogical_markers`                                                                                             |
| `duration_sec` | `int` | Length in seconds  | • **Content**: Direct feature<br>• **Derived**: Used for `speech_rate_wpm` (if transcript exists)                                                                                                                                |
| `view_count`   | `int` | Current view count | • **Engagement**: Used for normalization. For **new predictions**, set to `0`.                                                                                                                                                   |
| `like_count`   | `int` | Current like count | • **Engagement**: Used for `like_ratio`. For **new predictions**, set to `0`.                                                                                                                                                    |

### Optional Fields

| Field             | Type  | Description        | Feature Extraction Impact                                                                                                                                                                                                                         |
| ----------------- | ----- | ------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `publish_date`    | `str` | ISO format date    | • **Temporal**: `publish_hour` (cyclic), `day_of_week`, `is_evening_upload` (17h-21h), `is_weekday`, `days_since_publish`                                                                                                                         |
| `channel_id`      | `str` | YouTube Channel ID | • **Context**: Retrieves `channel_avg_views`, `channel_age_days`, `channel_video_count` from historical data (if trained on it).                                                                                                                  |
| `transcript_text` | `str` | Full transcript    | • **Linguistic**: `transcript_word_count`, `lexical_diversity` (unique/total words)<br>• **Readability**: `flesch_reading_ease`, `gunning_fog_index` (complexity)<br>• **Content**: `question_count`, `technical_term_density`, `speech_rate_wpm` |

### Feature Engineering Pipeline

The raw input follows this transformation path:

1.  **Cleaning**: Null handling, date parsing.
2.  **Temporal Extraction**: Converts `publish_date` into cyclic time features (hour_sin/cos) and polynomial recency features.
3.  **Text Processing**:
    *   **Simple**: Lengths, counts, regex-based keyword detection (e.g., "Bac", "Revision").
    *   **Advanced**: TF-IDF vectorization (30 dims) + AraBERT Embeddings reduced via PCA (100 dims).
4.  **Transcript Analysis** (if provided): Computes readability scores and pedagogical markers (examples, explanations, questions).
5.  **Scaling**: All numerical features are standardized using the saved `scaler_production.pkl`.

> **Note**: For new video prediction (where engagement is unknown), the model relies primarily on **Title/Description embeddings**, **Duration**, **Temporal features**, and **Transcript quality** to predict the potential `engagement_score`.

## 📁 Artifacts

The training process produces the following artifacts in the output directory:

- `*_production.pkl`: The trained model object.
- `scaler_production.pkl`: StandardScaler for numerical features.
- `tfidf_production.pkl`: TfidfVectorizer for text.
- `pca_production.pkl`: PCA model for AraBERT dimensionality reduction.
- `feature_columns.json`: List of features expected by the model.
- `training_metadata.json`: Metrics and configuration details.
