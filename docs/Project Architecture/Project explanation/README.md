# SIC Project: AI-Driven Analytics and Optimization System for Algerian Baccalaureate Educational Content

> **Comprehensive Technical Whitepaper**
> *A Three-Stage AI Pipeline: Data Analysis → Predictive Modeling → Generative AI Recommendation*

---

## Table of Contents

1.  [Introduction & Problem Statement](#1-introduction--problem-statement)
2.  [System Architecture Overview](#2-system-architecture-overview)
3.  [Phase 1: Data Collection & Ingestion](#3-phase-1-data-collection--ingestion)
4.  [Phase 2: Feature Engineering ("The Nibras Script")](#4-phase-2-feature-engineering-the-nibras-script)
5.  [Phase 3: Machine Learning Pipeline](#5-phase-3-machine-learning-pipeline)
6.  [Phase 4: Vector Database & RAG (Planned)](#6-phase-4-vector-database--rag-planned)
7.  [Phase 5: The "Analyser Agent" (Planned)](#7-phase-5-the-analyser-agent-planned)
8.  [Data Quality & Exploratory Analysis](#8-data-quality--exploratory-analysis)
9.  [Technology Stack](#9-technology-stack)
10. [Gap Analysis & Implementation Status](#10-gap-analysis--implementation-status)
11. [Reflection & Future Work](#11-reflection--future-work)
12. [Appendix: Data Schemas](#12-appendix-data-schemas)

---

## 1. Introduction & Problem Statement

### 1.1 Project Objective

The primary objective of this project is to **democratize access to high-quality educational content**—specifically targeting the Algerian Baccalaureate ecosystem—by empowering content creators with data-driven insights.

Content creators in the educational space often fly blind, relying on vanity metrics (like raw view counts) rather than indicators of actual educational impact. This project addresses the challenge of "content visibility" not by guessing, but by engineering a robust pipeline that:

1.  **Analyzes** historical success factors from top-performing videos.
2.  **Predicts** the potential engagement of a new video *before* it is published.
3.  **Recommends** actionable improvements using Generative AI (LLMs).

### 1.2 The Target: Algerian Bac Creators

The Algerian Baccalaureate ("Bac") is a high-stakes national exam. A growing ecosystem of YouTube educators creates video content to help students prepare. This project aims to provide these creators with a "virtual data scientist" that not only scores their work but actively guides them toward creating more effective, engaging educational content.

### 1.3 Core Innovation: The Engagement Score

Standard "View Count" is a poor proxy for educational quality—it rewards clickbait. We engineered a custom, robust target variable to represent **true "Engagement"**:

$$\text{Engagement Score} = \ln\left(1 + \frac{(3 \times \text{Comments}) + \text{Likes}}{\sqrt{\text{Views}}}\right)$$

*   **Why 3x Comments?** In education, a comment (question, thank-you, discussion) is a stronger "Active Learning Signal" than a passive Like. Comments represent students engaging with the material.
*   **Why `sqrt(Views)`?** Linear normalization unfairly penalizes viral videos. Square root normalization penalizes "empty clicks" while still respecting massive reach, reducing the impact of outliers.
*   **Why `ln(1 + x)`?** Engagement data is power-law distributed. The log-transform normalizes the distribution, improving regression model performance (R² increased from ~0.4 to **0.62** on test sets).

---

## 2. System Architecture Overview

The solution is architected as a **three-stage pipeline** combining statistical analysis, supervised machine learning (Regression), and Generative AI (RAG).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              SYSTEM ARCHITECTURE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ┌───────────────────┐    ┌───────────────────┐    ┌───────────────────┐   │
│   │  PHASE 1: DATA    │    │  PHASE 2: ML      │    │  PHASE 3: AI      │   │
│   │  ANALYSIS         │───▶│  PREDICTION       │───▶│  ADVISOR          │   │
│   │                   │    │                   │    │                   │   │
│   │ • YouTube API     │    │ • Nibras Script   │    │ • Vector DB       │   │
│   │ • Transcript      │    │ • Engagement      │    │ • LLM Agent       │   │
│   │   Collection      │    │   Scoring         │    │ • RAG Retrieval   │   │
│   │ • Bac Filtering   │    │ • Regression      │    │ • Advice Gen      │   │
│   └───────────────────┘    └───────────────────┘    └───────────────────┘   │
│          ✅                       ⚠️                       ❌               │
│      PRODUCTION              PROTOTYPE                 NOT STARTED          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Phase 1: Data Collection & Ingestion

### 3.1 YouTube Metadata Collection

**Implementation:** `src/data/youtube_collector.py`
**Class:** `YouTubeCollector`

#### 3.1.1 API Quota Optimization (The 100x Efficiency Gain)

The YouTube Data API v3 has a strict daily quota (10,000 units). Standard `search.list` calls cost **100 units per request**. We reverse-engineered a "sidebar" approach:

| Operation | Old Method (search.list) | Our Method (uploads playlist) | Savings |
|-----------|--------------------------|-------------------------------|---------|
| Discover 200 videos | 400 units | **4 units** | **100x** |
| Enrich 200 videos | 200 units | **4 units** | **50x** |
| **Total (4 channels)** | **2,400 units** | **24 units** | **100x** |

**How it works:**
1.  `get_uploads_playlist_id(channel_id)`: Retrieves the hidden "Uploads" playlist ID (1 API unit via `channels.list`).
2.  `get_playlist_videos(playlist_id)`: Enumerates videos via `playlistItems.list` (1 unit per 50 videos).
3.  `get_videos_metadata_batch(video_ids)`: Enriches video details in batches of 50 using `videos.list` (1 unit per request).

**Key Methods (with quota costs):**
```python
def get_uploads_playlist_id(self, channel_id: str) -> Optional[str]:
    """API quota cost: 1 unit (channels.list)"""
    # ...

def get_playlist_videos(self, playlist_id: str, max_videos: Optional[int] = None) -> List[str]:
    """API quota cost: 1 unit per request (50 items max per request)"""
    # ...

def get_videos_metadata_batch(self, video_ids: List[str], ...) -> List[Dict[str, Any]]:
    """API quota cost: 1 unit per 50 videos. Example: 1000 videos = 20 units."""
    # ...
```

#### 3.1.2 Video Registry for Incremental Updates

**Implementation:** `src/data/video_registry.py`
**Class:** `VideoRegistry`

The registry (`data/raw/video_registry.csv`) tracks all discovered video IDs, enabling:
*   Skip re-fetching known videos.
*   Track `discovered_at` and `last_seen_at` timestamps.
*   Bootstrap from existing CSVs.

### 3.2 Transcript Collection

**Implementation:** `src/data/transcript_collector.py`

#### 3.2.1 The Challenge: Rate Limits & Blocking

YouTube aggressively blocks transcript requests from single IPs. Our solution:

*   **Engine:** `yt-dlp` for fetching manual and auto-generated captions.
*   **Tor Proxy Integration:** Routes traffic through the Tor network via the `stem` library for automatic IP rotation.
*   **Smart Retry Queue:** The `SmartRetryQueue` class handles transient failures. Bot-detection errors are queued and retried *after* IP rotation.
*   **Language Priority:** Arabic (`ar`) > French (`fr`) > English (`en`).
*   **Content Validation:** Rejects corrupted transcripts (HTML/JS artifacts).

```python
def renew_tor_identity(control_port=9151, password=None):
    """Request a new identity from Tor to rotate IP."""
    # ...
    controller.signal(Signal.NEWNYM)
    logger.info("🔄 Tor Identity Rotated (New IP requested)")
    time.sleep(5)  # Wait for new circuit
    # ...
```

**Output:** `data/processed/transcripts_merged.csv` (~211 MB, 4,622 transcripts)

### 3.3 Data-Driven Bac Filtering

**Implementation:** `src/features/bac_filter_balanced.py`
**Class:** `BalancedBacFilter`

Manually curating keywords is brittle. We implemented a **hybrid, data-driven filtering strategy**:

**Strategy:**
1.  **Channel Priors (Auto-Computed):** Channels are analyzed to compute their "Bac ratio" (percentage of videos that are clearly Bac-related). High-prior channels are trusted more for ambiguous videos. Stored in `data/processed/channel_priors.csv`.
2.  **TF-IDF Discovery:** TF-IDF analysis on known Bac vs. Non-Bac titles discovers new Bac-associated keywords dynamically. Stored in `data/processed/tfidf_bac_terms.json`.
3.  **Decision Tree Logic:**
    *   **Hard Exclude:** Non-Bac grade markers present (e.g., "1AS", "2AS", "BEM").
    *   **Hard Include:** Explicit Bac 3AS grade markers (e.g., "bac", "3as", "باكالوريا", "terminale").
    *   **Soft Positives (Ambiguous):** For videos from Bac-heavy channels without explicit markers, we require: Channel is Bac-heavy + (TF-IDF hit OR Duration > 5 min OR Subject is assigned).

**Filter Output Columns:**
*   `is_bac_3as`: Boolean classification.
*   `filter_category`: `bac_3as | non_bac | unknown | bac_3as_ambiguous`.
*   `filter_confidence`: 0-1 score.
*   `filter_reason`: Human-readable explanation.
*   `subject`: From channel mapping.

**Output:** `data/processed/videos_bac_only.csv` (**10,030 videos** for ML)

### 3.4 Pipeline Orchestration

**Implementation:** `run_pipeline.py` (Unified CLI Entry Point)

**Workflow:**
```
Collect → Filter → Clean → Engineer → Analyze → Model (Notebook)
```

**CLI Commands:**
| Command | Description |
|---------|-------------|
| `full-pipeline` | Run everything: `collect` + `filter` + `clean` + `engineer` + `analyze` |
| `collect` | Gather video metadata and channel statistics |
| `filter_data` | Apply data-driven Bac 3AS filter |
| `clean` | Handle missing values, create `has_transcript` flag |
| `engineer` | Build 50+ ML features from raw data |

---

## 4. Phase 2: Feature Engineering ("The Nibras Script")

**Implementation:** `src/features/engineer.py`
**Class:** `VideoFeatureEngineer`

This is the heart of the project. The system computes **50+ distinct features** per video, going far beyond basic metadata.

### 4.1 Temporal Features (`_create_temporal_features`)

*   **Basic:** `days_since_publish`, `publish_hour`, `publish_day_of_week`, `publish_month`, `publish_year`.
*   **Binary:** `is_evening_upload` (17-21h optimal window), `is_weekday` (Sun-Thu, Algerian school week).
*   **Cyclic Encoding (Critical for ML):** Standard encoding treats Hour 23 as "far" from Hour 0. Cyclic encoding uses trigonometry to preserve continuity:
    ```python
    df["publish_hour_sin"] = np.sin(2 * np.pi * df["publish_hour"] / 24)
    df["publish_hour_cos"] = np.cos(2 * np.pi * df["publish_hour"] / 24)
    # Similarly for month (12 cycle), day of month (31 cycle)
    ```
*   **Polynomial Features:** `days_since_publish_squared`, `log_days_since_publish` (captures non-linear decay).
*   **Interaction Features:** `days_since_publish_x_hour`, `is_weekday_x_hour` (timing effects differ on weekends).

### 4.2 Text Features (`_create_text_features`)

*   **Basic Lengths:** `title_length`, `description_length`, `title_word_count`, `tag_count`.
*   **Subject Mapping:** `subject` column derived from `channels.csv` mapping (channel_id → subject).
*   **Exam Focus Detection (`is_exam_focused`):**
    ```python
    exam_keywords = ["bac", "exam", "examen", "2024", "2025", "revision", "exercise", "تمرين", "امتحان", "باك"]
    pattern = "|".join(exam_keywords)
    df["is_exam_focused"] = df["title"].str.lower().str.contains(pattern, na=False).astype(int)
    ```

### 4.3 Engagement Features (`_create_engagement_features`)

The **Target Variable (Engagement Score)** is computed here:
```python
# Line 358 in src/features/engineer.py
safe_views = df["view_count"].replace(0, 1)
df["engagement_score"] = np.log1p(
    (df["comment_count"] * 3 + df["like_count"]) / np.sqrt(safe_views)
)
```

*   **Ratios (for analysis, NOT model input):** `like_ratio`, `comment_ratio`.
*   **Engagement Category (for stratification):** `Low | Medium | High` based on 33rd/67th percentiles.

### 4.4 Channel Features (`_create_channel_features`)

*   `channel_video_count`: Number of videos in dataset from this channel.
*   `channel_avg_views`: Average views for channel.
*   `channel_video_avg_duration`: Average video duration for channel.
*   `channel_age_days`: Days since oldest video from channel.

### 4.5 Transcript Features (`_create_transcript_features`)

**Implementation:** `src/features/transcript_features.py`
**Function:** `extract_transcript_features(transcript_text, duration_sec, subject)`

This is where the project **goes beyond simple metadata** to analyze *how* teachers communicate.

#### 4.5.1 Linguistic Features
*   `transcript_word_count`, `transcript_char_count`, `transcript_sentence_count`
*   `avg_words_per_sentence`
*   `lexical_diversity` (Unique words / Total words)
*   `unique_word_count`

#### 4.5.2 Readability Scores (via `textstat` library)
*   `flesch_reading_ease`: Higher = easier to read (60-70 is ideal for high school).
*   `flesch_kincaid_grade`: US grade level equivalent.
*   `gunning_fog_index`: Higher = more complex.
*   `automated_readability_index`: Character-based readability.

#### 4.5.3 Pacing Features
*   `speech_rate_wpm`: `(word_count / duration_sec) * 60`
*   `speech_rate_optimal`: Binary flag, `1` if 120-180 WPM (optimal learning window).

#### 4.5.4 Pedagogical Markers (Custom Regex Detection)
These features detect teaching *style*, not just content.

| Feature | What it Detects | Example Markers (Multi-Lingual) |
|---------|-----------------|--------------------------------|
| `question_count`, `question_density` | Interactive teaching | `?`, `؟` |
| `example_count`, `example_density` | Concrete illustrations | "for example", "مثلا", "par exemple" |
| `explanation_count`, `explanation_density` | Logical reasoning | "because", "لأن", "therefore", "donc" |
| `contrast_count`, `contrast_density` | Nuanced teaching | "however", "لكن", "but", "cependant" |
| `technical_term_count`, `technical_term_density` | Domain-specific vocabulary | "theorem", "integral", "نظرية", "équation" |
| `subject_keyword_count`, `subject_keyword_density` | Subject relevance | Subject-specific keyword lists (Math, Physics, etc.) |

**Corruption Detection:**
Transcripts fetched via `yt-dlp` can sometimes return YouTube's HTML/JS error pages. We detect and reject these:
```python
corruption_markers = ['window.', 'ytcfg', 'u003d', 'javascript', '<html', ...]
if sum(1 for marker in corruption_markers if marker in text_lower) >= 2:
    logger.warning("Detected corrupted transcript (JS/HTML code), skipping.")
    return _empty_features()
```

### 4.6 Text Embeddings (Underutilized)

**Implementation:** `src/features/text_embeddings.py`
**Class:** `TextEmbeddingExtractor`

*   **Model:** `aubmindlab/bert-base-arabertv2` (AraBERT v2), specialized for Arabic text.
*   **Method:** `get_embeddings(texts, batch_size=32)` → Returns 768-dimensional vectors.
*   **Pooling:** Mean pooling over all token embeddings (better than CLS for regression).
*   **Output:** `data/modeling/arabert_embeddings.pkl` (30.8 MB)
*   **Current Status:** Embeddings are generated but **NOT connected to a Vector DB** yet. This is the bridge to Phase 3 (RAG).

---

## 5. Phase 3: Machine Learning Pipeline

### 5.1 Current State: Notebook Prototype

**Location:** `notebooks/02_model_baseline_testing.ipynb`

The ML logic exists but is **not yet refactored into production scripts**.

### 5.2 Models Tested

| Model | Validation R² | Test R² | Test MAE | Test RMSE | Notes |
|-------|---------------|---------|----------|-----------|-------|
| **Random Forest** | ~0.62 | **0.6248** | 0.3534 | 0.4630 | Best baseline |
| XGBoost | ~0.60 | - | - | - | Competitive |
| LightGBM | ~0.59 | - | - | - | Fastest training |

### 5.3 Data Splits

*   **Train:** 6,018 videos (60%)
*   **Validation:** 2,006 videos (20%)
*   **Test:** 2,006 videos (20%)
*   **Saved to:** `data/modeling/{train,val,test}.csv`

### 5.4 Feature Processing

*   **One-Hot Encoding:** `subject` (dropped first dummy).
*   **Standard Scaling:** All numerical features.
*   **TF-IDF (optional):** Title + Description (50 features, max_df=0.8) – tested in notebook.

### 5.5 Leakage Prevention (Critical)

The engagement score is *derived from* `view_count`, `like_count`, `comment_count`. Using these as model inputs would cause **severe data leakage**.

**Excluded from Feature Set (`exclude_cols`):**
```python
exclude_cols = [
    'video_id', 'channel_id', 'channel_title', 'engagement_score', 'engagement_category',
    'view_count', 'like_count', 'comment_count',  # CAUSES DATA LEAKAGE!
    'like_ratio', 'comment_ratio',  # Also derived from raw counts
    'title', 'description', 'tags', 'publish_date',  # Raw text (use extracted features)
    # Channel-level aggregates (removed to test if model overfits to channel stats):
    # 'channel_video_count', 'channel_avg_views', ...
]
```

### 5.6 Saved Model Artifacts

*   `models/random_forest_baseline.pkl` (23.6 MB)
*   `models/xgboost_baseline.pkl` (181 KB)
*   `models/lightgbm_baseline.pkl` (266 KB)
*   `models/scaler_baseline.pkl`
*   `models/tfidf_vectorizer.pkl`

---

## 6. Phase 4: Vector Database & RAG (Planned)

**Status:** ❌ **NOT IMPLEMENTED (0% Complete)**

**Evidence:**
*   No Vector DB libraries in `pyproject.toml` (`chromadb`, `faiss`, `pinecone`, `qdrant`, `milvus`).
*   No LLM libraries (`langchain`, `llama-index`, `openai`, `anthropic`).
*   Grep search for "Vector", "RAG", "Agent", "LLM" → **0 results** in `src/`.

### 6.1 Intended Architecture

**Goal (from Goal Document):**
1.  Filter top 10% videos by `engagement_score`.
2.  Chunk their transcripts/descriptions.
3.  Embed chunks using `TextEmbeddingExtractor` (AraBERT).
4.  Store in Vector DB with metadata: `{video_id, subject, topic, chunk_text, metrics}`.

**Recommended Implementation:**
*   **Vector DB:** ChromaDB (local dev) or FAISS (production).
*   **Script:** `scripts/build_vector_db.py`

---

## 7. Phase 5: The "Analyser Agent" (Planned)

**Status:** ❌ **NOT IMPLEMENTED (0% Complete)**

### 7.1 Intended Architecture (from Goal Document)

```
Input Video → Feature Extraction → Regression Model → Predicted Score
                                          ↓
                                    Vector DB Query (RAG)
                                          ↓
                                Retrieve Best Practices
                                          ↓
                    LLM Prompt: [Input Features + Score + Best Practices]
                                          ↓
                              Tailored Advice (Text Output)
```

### 7.2 Missing Components

1.  **Vector DB Retrieval Logic:** No `src/knowledge/retriever.py`.
2.  **LLM Client:** No OpenAI/Anthropic/Gemini API wrapper (`src/agent/llm_client.py`).
3.  **Agent Orchestration:** No `src/agent/analyzer_agent.py`.
4.  **Prompt Engineering:** No system prompts, few-shot examples, or output format constraints.

---

## 8. Data Quality & Exploratory Analysis

### 8.1 Dataset Overview (from `01_data_exploration.ipynb`)

| Dataset | File | Shape | Size |
|---------|------|-------|------|
| Raw Videos (Filtered) | `videos_bac_only.csv` | 10,030 rows × 21 cols | 16.6 MB |
| Channel Statistics | `channel_statistics.csv` | 36 rows | - |
| Transcripts (Merged) | `transcripts_merged.csv` | 4,622 transcripts | 211 MB |
| Engineered Features | `videos_engineered.csv` | 10,030 rows × 50 cols | 7.8 MB |

### 8.2 Missing Values

| Column | Missing Count | Missing % | Handling |
|--------|---------------|-----------|----------|
| `tags` | 5,904 | 58.9% | `fillna("")`, use `tag_count` feature |
| `description` | 3,745 | 37.3% | `fillna("")`, use `description_length` feature |
| Transcript features | ~5,408 | 53.9% | `fillna(0)`, use `has_transcript` flag |

### 8.3 Subject Distribution

| Subject | Video Count | Percentage |
|---------|-------------|------------|
| Maths | 4,057 | 40.4% |
| History & Geography | 1,962 | 19.6% |
| Natural Sciences | 1,743 | 17.4% |
| Physics | 1,027 | 10.2% |
| Islamic Sciences | 365 | 3.6% |
| Arabic | 357 | 3.6% |
| English | 241 | 2.4% |
| French | 148 | 1.5% |
| Philosophy | 130 | 1.3% |

### 8.4 Engagement Score Distribution

```
count    10,030
mean         2.419
std          0.742
min          0.000
25%          1.936
50%          2.452
75%          2.916
max          5.140
```

---

## 9. Technology Stack

### 9.1 Core Dependencies (`pyproject.toml`)

| Category | Libraries |
|----------|-----------|
| **Data Science** | `pandas >= 2.0.0`, `numpy >= 1.24.0` |
| **ML Models** | `scikit-learn >= 1.3.0`, `xgboost >= 3.1.3`, `lightgbm >= 4.6.0` |
| **Deep Learning** | `torch == 2.5.1` (CUDA 12.1), `transformers >= 4.40.0` |
| **NLP** | `textstat >= 0.7.12` (readability) |
| **YouTube API** | `google-api-python-client >= 2.100.0` |
| **Transcript Fetching** | `yt-dlp >= 2025.12.8` |
| **Network/Proxy** | `pysocks`, `stem` (Tor control) |
| **Visualization** | `matplotlib`, `seaborn` |
| **Dev Tools** | `pytest`, `black`, `ruff`, `mypy`, `jupyter` |

### 9.2 Missing for Phase 3/4

*   **Vector DB:** `chromadb` OR `faiss-cpu` OR `qdrant-client`
*   **LLM:** `openai` OR `anthropic` OR `langchain`

---

## 10. Gap Analysis & Implementation Status

### 10.1 Component Status Summary

| Component | Intended Architecture | Current State | Gap Severity |
|-----------|----------------------|---------------|--------------|
| **Data Collection** | YouTube API + Transcripts | ✅ Production-Ready | ✓ Low |
| **Feature Engineering** | "Nibras Script" with 50+ features | ✅ Production-Ready | ✓ Low |
| **Model Training** | Versioned, reproducible pipeline | ⚠️ Notebook only | 🔴 High |
| **Model Inference** | Script for single-instance prediction | ❌ Missing | 🔴 High |
| **Vector Database** | Best Practices storage | ❌ Missing | 🔴 Critical |
| **RAG Retrieval** | Query Vector DB for context | ❌ Missing | 🔴 Critical |
| **LLM Agent** | Advice generation from Score + Context | ❌ Missing | 🔴 Critical |

### 10.2 Specific "Left to Implement" List

| Component | Details | Priority |
|-----------|---------|----------|
| **`src/models/train_model.py`** | Refactor `02_model_baseline_testing.ipynb` into production script. | 🔴 High |
| **`src/models/predict_model.py`** | Accept JSON input (video draft) → Output `engagement_score`. | 🔴 High |
| **`scripts/build_vector_db.py`** | Select top 10% videos, chunk transcripts, embed, store in ChromaDB. | 🔴 Critical |
| **`src/knowledge/retriever.py`** | Query Vector DB based on input context. | 🔴 Critical |
| **`src/agent/llm_client.py`** | OpenAI/Gemini API wrapper. | 🔴 Critical |
| **`src/agent/analyzer_agent.py`** | Orchestrate: Features + Score + Retrieved Context → LLM Prompt → Advice. | 🔴 Critical |
| **`config/model_config.yaml`** | Externalize hardcoded values (3x comment weight, exam keywords). | 🟡 Medium |

---

## 11. Reflection & Future Work

### 11.1 Project Limitations

*   **Data Limitations:**
    *   No access to YouTube Analytics (Watch Time, Impressions). Engagement Score is a *proxy*.
    *   Transcript availability is ~46%, limiting linguistic analysis coverage.
*   **Technical Limitations:**
    *   ML pipeline is prototype-only (notebooks, not production scripts).
    *   No Vector DB or LLM Agent implementation yet.
*   **Time-Related Limitations:**
    *   Focused on robust data foundation; deployment phases remain.

### 11.2 Lessons Learned

*   **API Quota Mastery:** The 100x efficiency gain from the "uploads playlist" method was critical to project viability.
*   **Data-Driven Filtering:** Moving from manual keyword lists to auto-computed priors/TF-IDF significantly improved recall and reduced maintenance.
*   **Feature Engineering Depth:** Pedagogical markers (questions, examples) proved more informative than expected.

### 11.3 Concrete Future Improvements

1.  **Productionize ML:** `train_model.py`, `predict_model.py`, CLI integration.
2.  **Build Vector DB:** ChromaDB + AraBERT embeddings for "Best Practices" retrieval.
3.  **Implement RAG Agent:** Connect predicted score + retrieved practices → LLM → Actionable advice.
4.  **Model Experimentation:** CatBoost, Neural Networks, Ensemble stacking.
5.  **Hyperparameter Tuning:** GridSearch/Optuna for RF, XGB, LGBM.
6.  **SHAP Explainability:** Understand which features drive predictions.
7.  **Interactive Dashboard:** Streamlit/Gradio for creator-facing UI.

### 11.4 Potential Long-Term Impact

*   Democratize access to "data-driven content strategy" for small Algerian educators.
*   Provide a template for educational video analytics in other markets/languages.
*   Enable creators to iterate *before* publishing, reducing "content waste".

---

## 12. Appendix: Data Schemas

### A. `videos_metadata.csv` (Raw)
```
video_id, title, description, publish_date, channel_id, channel_title,
category_id, duration_iso, duration_sec, view_count, like_count, comment_count,
tags, thumbnail_url, snapshot_date, run_id
```

### B. `videos_bac_balanced.csv` (After Filtering)
Additional columns:
```
is_bac_3as, filter_category, filter_confidence, filter_reason, subject
```

### C. `videos_engineered.csv` (After Feature Engineering)
50 columns including:
```
# Temporal
days_since_publish, publish_hour, publish_month, is_weekday, is_evening_upload,
publish_hour_sin, publish_hour_cos, ...

# Content
duration_sec, title_length, description_length, tag_count, is_exam_focused, subject

# Engagement (Target)
engagement_score, engagement_category

# Channel
channel_video_count, channel_avg_views, channel_age_days

# Transcript (if available)
transcript_word_count, lexical_diversity, flesch_reading_ease, speech_rate_wpm,
question_count, example_count, explanation_count, has_transcript, ...
```

---

**END OF TECHNICAL WHITEPAPER**

---

> *"Success is not about having the best model, but about telling the best story with your data."* — Final Presentation Guide
