# Current Implementation State: AI-Driven Analytics System for Algerian Baccalaureate Content

**Date:** 2026-01-23  
**Reviewer:** Senior Technical Lead / Solutions Architect  
**Project:** YouTube Educational Video Engagement Analysis & Recommendation System

---

## I. EXECUTIVE SUMMARY

### A. Project Goal (From Goal Document)
Three-stage AI system to help educational content creators optimize their videos:
1. **Data Analysis:** Extract patterns from top videos → Store "Best Practices" in Vector DB
2. **Prediction:** Regression model predicts Engagement Score using video features
3. **Recommendation:** LLM Agent (RAG) provides actionable advice using predicted score + best practices

### B. Current Implementation Status
| Component                                 | Status             | Completeness |
| ----------------------------------------- | ------------------ | ------------ |
| **Data Collection Pipeline**              | ✅ Production-Ready | ~95%         |
| **Feature Engineering ("Nibras Script")** | ✅ Production-Ready | ~90%         |
| **ML Model (Regression)**                 | ⚠️ Prototype Only   | ~40%         |
| **Vector Database**                       | ❌ Not Started      | 0%           |
| **LLM Agent (RAG)**                       | ❌ Not Started      | 0%           |

---

## II. DATA COLLECTION & INGESTION ARCHITECTURE

### A. YouTube Metadata Collection
#### Implementation: `src/data/youtube_collector.py`
**Class:** `YouTubeCollector`

**Key Methods:**
- `get_uploads_playlist_id(channel_id)` → Retrieves playlist ID (1 API unit)
- `get_playlist_videos(playlist_id)` → Lists video IDs (1 unit per 50 videos)
- `get_videos_metadata_batch(video_ids)` → Batch enrichment (1 unit per 50 videos)
- `collect_from_channels(channels_df, max_videos, max_quota)` → Main orchestration

**Optimization Strategy:**
- **Uploads Playlist Method:** 100x more efficient than `search.list` API
- **Batching:** 50 videos per request vs. individual calls
- **Registry System:** `src/data/video_registry.py` tracks known videos to avoid re-fetching

**Data Output:** `data/raw/videos_metadata.csv`

**Schema:**
```python
{
    "video_id": str,
    "title": str,
    "description": str,
    "publish_date": datetime,
    "duration_sec": int,
    "view_count": int,
    "like_count": int,
    "comment_count": int,
    "channel_id": str,
    "channel_title": str,
    "snapshot_date": datetime,
    "run_id": str
}
```

### B. Transcript Collection
#### Implementation: `src/data/transcript_collector.py`
**Key Components:**
- **Library:** `yt-dlp` (for auto-generated and manual captions)
- **Proxy Support:** Tor integration via `stem` library
- **Retry Logic:** `SmartRetryQueue` class handles transient failures
- **Languages:** Arabic (priority), French, English fallback

**Function:** `get_best_transcript(video_id, proxies)` → Returns cleaned transcript text

**Data Output:** `data/processed/transcripts_merged.csv` (211 MB, 4,622 transcripts)

**Schema:**
```python
{
    "video_id": str,
    "transcript_text": str,
    "success": bool,
    "failure_reason": str (if failed)
}
```

### C. Pipeline Orchestration
#### Implementation: `run_pipeline.py`
**CLI Commands:**
- `full-pipeline --filter` → End-to-end execution
- `collect --channels <path>` → YouTube API collection
- `filter_data` → Apply Bac 3AS filtering
- `engineer --input <path>` → Feature engineering

**Current Workflow:**
```
Raw Data Collection → Bac Filtering → Data Cleaning → Feature Engineering
```

---

## III. FEATURE ENGINEERING ("NIBRAS SCRIPT")

### A. Core Implementation: `src/features/engineer.py`
**Class:** `VideoFeatureEngineer`

#### 1. Engagement Score Formula (Target Variable)
**Implementation:** `_create_engagement_features()`

```python
engagement_score = np.log1p(
    (comment_count * 3 + like_count) / np.sqrt(view_count)
)
```

**Rationale:**
- Comments weighted 3x (active learning signal)
- Normalized by sqrt(views) to penalize "empty clicks"
- Log-transformed for better regression performance (R² ~0.64)

**Location:** Line 358, `src/features/engineer.py`

#### 2. Feature Categories

**A. Temporal Features** (`_create_temporal_features()`)
- Basic: `days_since_publish`, `publish_hour`, `publish_day_of_week`, `publish_month`
- Binary: `is_evening_upload` (17-21h), `is_weekday` (Sun-Thu)
- Cyclic Encoding: `publish_hour_sin/cos`, `publish_month_sin/cos`, `publish_day_of_month_sin/cos`
- Polynomial: `days_since_publish_squared`, `log_days_since_publish`
- Interactions: `days_since_publish_x_hour`, `is_weekday_x_hour`

**B. Text Features** (`_create_text_features()`)
- `title_length`, `description_length`, `title_word_count`, `tag_count`
- `subject` (from `channels.csv` mapping)
- `is_exam_focused` (keyword detection: "bac", "امتحان", "باك", etc.)

**C. Channel Features** (`_create_channel_features()`)
- `channel_video_count`, `channel_avg_views`, `channel_video_avg_duration`, `channel_age_days`

**D. Transcript Features** (`_create_transcript_features()`)
**Implementation:** `src/features/transcript_features.py`
**Function:** `extract_transcript_features(transcript_text, duration_sec, subject)`

**Linguistic Features:**
- `transcript_word_count`, `unique_word_count`, `lexical_diversity`
- `transcript_sentence_count`, `avg_words_per_sentence`
- Readability: `flesch_reading_ease`, `flesch_kincaid_grade`, `gunning_fog_index`, `automated_readability_index`

**Pedagogical Features:**
- `speech_rate_wpm`, `speech_rate_optimal` (120-180 WPM)
- `question_count`, `example_count`, `explanation_count`, `contrast_count`
- `technical_term_count`, `subject_keyword_count` (subject-specific lexicons)

**Libraries Used:**
- `textstat` (readability metrics)
- Custom regex patterns for pedagogical markers

### B. Data-Driven Filtering: `src/features/bac_filter_balanced.py`
**Class:** `BalancedBacFilter`

**Strategy:**
- **Channel Priors:** Auto-computed Bac-heavy channels (`data/processed/channel_priors.csv`)
- **TF-IDF Discovery:** Bac-associated terms (`data/processed/tfidf_bac_terms.json`)
- **Decision Tree Logic:**
  1. Hard exclude: Non-Bac markers (unless strong Bac intent)
  2. Hard include: Explicit grade markers (3AS, terminale, etc.)
  3. Soft positives: TF-IDF hits + channel prior + duration \u003e 5 min

**Output:** `data/processed/videos_bac_only.csv` (10,030 videos for ML)

---

## IV. MACHINE LEARNING PIPELINE

### A. Current State: Notebook Prototype
**Location:** `notebooks/02_model_baseline_testing.ipynb`

**Models Tested:**
- Random Forest (best: R² = 0.6248 on test set)
- XGBoost
- LightGBM

**Data Splits:**
- Train: 60% (6,018 videos)
- Validation: 20% (2,006 videos)
- Test: 20% (2,006 videos)
- **Saved to:** `data/modeling/{train,val,test}.csv`

**Feature Processing:**
- One-hot encoding: `subject` (dropped first)
- Standard scaling: Numerical features
- TF-IDF: Title + Description (50 features, max_df=0.8)

**Models Saved:**
- `models/random_forest_baseline.pkl` (23.6 MB)
- `models/xgboost_baseline.pkl` (181 KB)
- `models/lightgbm_baseline.pkl` (266 KB)
- `models/scaler_baseline.pkl`
- `models/tfidf_vectorizer.pkl`

**Evaluation Metrics:**
```
Random Forest Test Set:
  MAE:  0.3534
  RMSE: 0.4630
  R²:   0.6248
```

### B. Production Gap Analysis

**❌ Missing Components:**
1. **No `src/models/train_model.py`**
   - Notebook logic not refactored to production script
   - No CLI for model training

2. **No `src/models/predict_model.py`**
   - Cannot accept new video → predict score workflow
   - No inference API or script

3. **No Automated Training Pipeline**
   - Manual notebook execution required
   - No versioning or experiment tracking (MLflow, Weights & Biases)

4. **No Model Serving**
   - No REST API (Flask/FastAPI)
   - No batch inference script

---

## V. VECTOR DATABASE & RAG (PHASE 1 \u0026 3)

### A. Current State: **NOT IMPLEMENTED**

**Evidence:**
- No Vector DB libraries in `pyproject.toml`:
  - No `chromadb`, `faiss`, `pinecone`, `qdrant`, `milvus`
- No LLM libraries:
  - No `langchain`, `llama-index`, `openai`, `anthropic`
- Grep search for "Vector", "RAG", "Agent", "LLM" → **0 results** in `src/`

### B. Existing Embedding Capability (Underutilized)
**Implementation:** `src/features/text_embeddings.py`
**Class:** `TextEmbeddingExtractor`

**Model:** AraBERT v2 (`aubmindlab/bert-base-arabertv2`)
**Method:** `get_embeddings(texts, batch_size=32)` → Returns 768-dim vectors

**Current Usage:**
- Saved embeddings: `data/modeling/arabert_embeddings.pkl` (30.8 MB)
- **NOT connected to Vector DB**

### C. Missing "Best Practices" Pipeline

**Required (per Goal Document):**
1. Filter top 10% videos by `engagement_score`
2. Chunk transcripts/descriptions
3. Embed chunks using `TextEmbeddingExtractor`
4. Store in Vector DB with metadata (subject, topic, metrics)

**Current Status:** ❌ None of this exists

---

## VI. LLM AGENT ("ANALYSER AGENT")

### A. Current State: **NOT IMPLEMENTED**

**Required Architecture (per Goal Document):**
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

**Missing Components:**
1. **No Vector DB Retrieval Logic**
2. **No LLM Client** (OpenAI, Anthropic, Google Gemini)
3. **No Prompt Engineering** (system prompts, few-shot examples)
4. **No Agent Orchestration** (LangChain Agents, custom pipeline)

---

## VII. TECHNOLOGY STACK SUMMARY

### A. Current Dependencies (`pyproject.toml`)
**Data Science:**
- `pandas >= 2.0.0`
- `numpy >= 1.24.0`
- `scikit-learn >= 1.3.0`

**ML Models:**
- `xgboost >= 3.1.3`
- `lightgbm >= 4.6.0`

**NLP:**
- `transformers >= 4.40.0` (HuggingFace)
- `torch == 2.5.1` (PyTorch with CUDA 12.1 support)
- `textstat >= 0.7.12` (readability)

**Data Collection:**
- `google-api-python-client >= 2.100.0` (YouTube API)
- `yt-dlp >= 2025.12.8` (transcript downloader)

**Other:**
- `matplotlib`, `seaborn` (visualization)
- `pysocks` (Tor proxy), `stem` (Tor control)

### B. Missing for Phase 3 (Agent)
**Vector DB:**
- Need: `chromadb` OR `faiss-cpu` OR `qdrant-client`

**LLM:**
- Need: `openai` OR `anthropic` OR `langchain`

---

## VIII. DATA SCHEMAS

### A. Engineered Features Dataset
**File:** `data/processed/videos_engineered.csv` (7.8 MB, 10,030 videos)
**Columns:** 50 total

**Sample Feature List:**
- Identifiers: `video_id`, `channel_id`, `title`, `description`
- Temporal: `publish_date`, `is_weekday`, `days_since_publish_*`
- Content: `duration_sec`, `title_length`, `subject`, `is_exam_focused`
- Target: `engagement_score`, `engagement_category` (Low/Medium/High)
- Transcript: `transcript_word_count`, `flesch_reading_ease`, `speech_rate_wpm`, `question_count`, etc.
- Channel: `channel_video_count`, `channel_avg_views`

### B. Modeling Splits
**Train:** 6,018 videos (60%)
**Val:** 2,006 videos (20%)
**Test:** 2,006 videos (20%)

**Subject Distribution:**
- Maths: 4,057
- History & Geography: 1,962
- Natural Sciences: 1,743
- Physics: 1,027
- Other subjects: ~1,241

---

## IX. GOAL vs. REALITY CHECK

### A. Alignment

| Goal Component                           | Implementation Status  | Notes                                           |
| ---------------------------------------- | ---------------------- | ----------------------------------------------- |
| **YouTube Data Ingestion**               | ✅ Fully Implemented    | Quota-optimized API calls                       |
| **Transcript Collection**                | ✅ Fully Implemented    | Multi-language support, Tor proxy               |
| **Engagement Score Formula**             | ✅ Matches Goal Exactly | `ln(1 + (3*comments + likes)/sqrt(views))`      |
| **Feature Extraction ("Nibras Script")** | ✅ Exceeds Goal         | Title, Description, Duration, **+ Transcripts** |
| **Regression Model**                     | ⚠️ Prototype Only       | Exists in notebook, not production code         |
| **Vector DB**                            | ❌ Missing              | No implementation                               |
| **Best Practices Storage**               | ❌ Missing              | No pipeline                                     |
| **LLM Agent**                            | ❌ Missing              | No code                                         |
| **RAG System**                           | ❌ Missing              | No retrieval logic                              |

### B. Discrepancies

**1. Features in Code NOT in Goal:**
- **Transcript Linguistic Analysis:** Readability scores, lexical diversity, pedagogical markers
- **Advanced Temporal Features:** Cyclic encoding, polynomial features, interaction terms
- **Channel-Level Aggregations:** Avg views, video count, channel age

**2. Features in Goal NOT in Code:**
- **Vector DB Infrastructure:** ChromaDB/FAISS implementation
- **Best Practices Extraction:** Top 10% video analysis + chunking
- **LLM Integration:** OpenAI/Anthropic API client
- **Agent Prompt Engineering:** System prompts for advice generation

**3. Logic Differences:**
- **Goal:** "Specialized script (Nibras) extracts features"
  - **Reality:** Python class `VideoFeatureEngineer` with modular methods
- **Goal:** "Vector DB stores Best Practices"
  - **Reality:** Embeddings exist (`arabert_embeddings.pkl`) but not in Vector DB

---

## X. RECOMMENDED NEXT STEPS

### Phase 1: Productionize ML Pipeline (High Priority)
1. **Refactor Notebook → Production Scripts**
   - `src/models/train_model.py` (hyperparameter tuning, cross-validation)
   - `src/models/predict_model.py` (single-instance inference)
   - CLI integration in `run_pipeline.py`

2. **Model Versioning**
   - Implement experiment tracking (MLflow or Weights & Biases)
   - Model registry for A/B testing

### Phase 2: Build Vector Database (Critical Path)
1. **Select Vector DB:** ChromaDB (recommended for local dev) or FAISS (production)
2. **Create `scripts/build_vector_db.py`:**
   - Filter top 10% videos by `engagement_score`
   - Chunk transcripts (512 tokens, 128 overlap)
   - Embed using `TextEmbeddingExtractor`
   - Store with metadata: `{video_id, subject, topic, chunk_text, metrics}`

3. **Implement Retrieval:**
   - `src/knowledge/retriever.py` → Query by video subject/topic

### Phase 3: LLM Agent Implementation (Final Mile)
1. **LLM Client:** `src/agent/llm_client.py`
   - OpenAI API wrapper (GPT-4 recommended)
   - Fallback to Anthropic Claude

2. **Agent Orchestration:** `src/agent/analyzer_agent.py`
   - Input: Video features + Predicted score
   - Retrieve: Top 5 relevant best practices from Vector DB
   - Prompt: "You are an AI advisor for educational content..."
   - Output: Structured advice (JSON or Markdown)

3. **Prompt Engineering:**
   - System prompt with role definition
   - Few-shot examples of good advice
   - Output format constraints

### Phase 4: Configuration Management
**Create:** `config/model_config.yaml`
```yaml
engagement_score:
  comment_weight: 3.0  # Currently hardcoded
  normalization: sqrt  # sqrt(views)
  
filtering:
  exam_keywords: ["bac", "امتحان", ...]  # Currently hardcoded in engineer.py
```

---

## XI. FILE STRUCTURE OVERVIEW

```
SIC/
├── src/
│   ├── data/
│   │   ├── youtube_collector.py       [✅ Production]
│   │   ├── transcript_collector.py    [✅ Production]
│   │   ├── storage.py                 [✅ Production]
│   │   └── video_registry.py          [✅ Production]
│   ├── features/
│   │   ├── engineer.py                [✅ Production - Nibras Script]
│   │   ├── transcript_features.py     [✅ Production]
│   │   ├── text_embeddings.py         [⚠️ Underutilized]
│   │   └── bac_filter_balanced.py     [✅ Production]
│   └── models/                         [❌ EMPTY - Critical Gap]
│
├── notebooks/
│   └── 02_model_baseline_testing.ipynb [⚠️ Prototype Only]
│
├── models/                             [⚠️ Manual Exports]
│   ├── random_forest_baseline.pkl
│   ├── xgboost_baseline.pkl
│   └── scaler_baseline.pkl
│
├── data/
│   ├── raw/
│   │   ├── videos_metadata.csv        [41 MB, snapshots]
│   │   └── channels.csv               [Subject mapping]
│   ├── processed/
│   │   ├── videos_bac_only.csv        [7.8 MB, 10k videos]
│   │   ├── videos_engineered.csv      [50 features]
│   │   └── transcripts_merged.csv     [211 MB, 4.6k transcripts]
│   └── modeling/
│       ├── train.csv, val.csv, test.csv
│       └── arabert_embeddings.pkl     [⚠️ Not in Vector DB]
│
└── run_pipeline.py                    [✅ CLI Orchestrator]
```

---

## XII. CONCLUSION

**Strengths:**
- Robust data collection with quota optimization
- Production-ready feature engineering exceeding initial scope
- Functional ML prototype with competitive R² (0.62)

**Critical Gaps:**
- **No production ML inference pipeline**
- **Vector DB completely missing (0% complete)**
- **LLM Agent not started (0% complete)**

**Current System Can:**
- Collect and clean YouTube data
- Engineer 50+ features including advanced transcript analysis
- Train regression models (manually via notebook)

**Current System CANNOT:**
- Predict engagement for new videos (no inference API)
- Store or retrieve best practices (no Vector DB)
- Generate actionable advice (no LLM Agent)

**Completion Estimate:** **~50% of intended system** (Phase 1 + partial Phase 2 complete)

---

**END OF REPORT**
