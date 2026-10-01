# Engagement model card

## What it predicts

The **engagement score** of a Bac 3AS lesson on YouTube:

```text
engagement score = log(1 + (3 × comments + likes) / √views)
```

Comments count three times as much as likes because on lesson videos they are mostly students asking questions. Dividing by the square root of views keeps large channels from dominating, and the log tames the long tail.

The predictor also returns a **category**: Low, Medium or High, meaning the bottom, middle or top third of training videos. The cut-offs are learned when the model is trained (2.106 and 2.764 for the current model).

## Intended use

Scoring a **planned video**, before it is published, so a creator can compare versions of a title, description or length. It is an estimate for one niche (Algerian Bac 3AS lessons), not a guarantee of views.

## Inputs

| Field | Required | Notes |
| --- | --- | --- |
| `title` | yes | |
| `duration_sec` | yes | |
| `channel_id` or `subject` | one of them | An unseen channel gets the training medians for channel features, and the result says `known_channel: false`. |
| `description`, `tags`, `publish_date`, `transcript_text` | no | A missing date counts as "published now". |

No statistics are needed, and none are read: a test checks that the features are identical with view, like and comment counts present, set to NaN, or absent.

## Model

- **Algorithm:** random forest (357 trees, max depth 21, `max_features=0.5`). These are the best settings from the notebook's Optuna search.
- **171 features:**
    - 41 numeric features, kept by VIF selection from 55 candidates. The candidates are title and description lengths, exam and pedagogical keywords, duration, channel statistics, subject, and 32 transcript features such as readability, speaking pace and questions asked
    - 30 TF-IDF terms
    - 100 PCA components of AraBERT v2 embeddings (`aubmindlab/bert-base-arabertv2`) of title + description
- **Fitting:** everything data-dependent is fit on the training split only: channel statistics, VIF selection, scaler, TF-IDF and PCA.

## Training data

| | |
| --- | --- |
| Collected | 2026-09-30, YouTube Data API v3 |
| Channels | 36 curated, 35 still available |
| Videos | 9,801 Bac 3AS lessons (filtered from 19,042) |
| With a valid transcript | 4,381 |
| Published | 2014-03-15 to 2026-06-06 |
| Subjects | Maths 4,097 · History & Geography 1,910 · Natural Sciences 1,710 · Physics 894 · Islamic Sciences 369 · Arabic 352 · English 236 · French 124 · Philosophy 109 |

## Results

Held-out test set: the 1,963 videos of the notebook's test split that are still online (random 20%, seed 42). Training used the other 7,838.

| Metric | Value |
| --- | --- |
| R² | **0.703** |
| MAE | 0.314 |
| RMSE | 0.410 |

### How it compares to the notebook

The exploration notebook reported R² 0.693 for the same algorithm. That number is optimistic: the notebook fitted channel statistics, feature selection and the scaler on all videos, test videos included. The production pipeline fixes that leak. The data also differs: statistics are eight months more mature and 52 test videos are gone. The two numbers therefore measure different things, and the drop from leak-free fitting is not separately measured.

The notebook's comparison of algorithms (January data, same leaky protocol, useful for ranking only):

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Random Forest (Optuna) | 0.310 | 0.413 | 0.692 |
| Ensemble (all 4 models) | 0.312 | 0.414 | 0.690 |
| CatBoost (Optuna) | 0.313 | 0.418 | 0.684 |
| Random Forest | 0.317 | 0.422 | 0.678 |
| Neural network (MLP) | 0.316 | 0.426 | 0.672 |
| XGBoost (regularized) | 0.330 | 0.434 | 0.660 |
| LightGBM | 0.337 | 0.441 | 0.649 |
| XGBoost | 0.337 | 0.443 | 0.645 |
| Ridge regression | 0.416 | 0.530 | 0.492 |
| Lasso regression | 0.599 | 0.744 | ≈ 0 |

## Limitations

- **Same channels in train and test.** The split is random over videos, so every test video's channel is also in training. How well the model does on a channel it has never seen is not measured, and that is the coach's main use case.
- **The score is a proxy.** It measures visible interaction, not learning.
- **Snapshot in time.** Older videos have had longer to collect comments and likes; the model has no "age" feature, by design, because a planned video has no age.
- **Uneven subjects.** Maths is 42% of the data and Philosophy 1%; predictions for small subjects rest on fewer examples.
- **Three language subjects have no curriculum keyword list.** Their keyword features use all subjects' keywords.

## Reproducing

The model is trained from data that is not published (see [Data and ethics](data.md)). With your own collection:

```bash
uv run python run_pipeline.py train --split-file splits_mapping.csv   # or omit for a random 20%
```
