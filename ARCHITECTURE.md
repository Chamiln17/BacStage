# Project Architecture

This document describes the architecture and design decisions for the SIC project.

## Overview

SIC is an end-to-end machine learning pipeline for analyzing YouTube video engagement in Algerian Bac educational content. The project follows data science best practices with clear separation between exploration (notebooks) and production code (src/).

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     Data Sources                             │
│                  (YouTube Data API v3)                       │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                  Data Collection                             │
│         (src/data/youtube_collector.py)                      │
│  • Channel video enumeration                                 │
│  • Video metadata extraction                                 │
│  • Quota management                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Raw Data Storage                           │
│              (data/raw/*.csv)                                │
│  • videos_metadata.csv                                       │
│  • channels.csv                                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│               Feature Engineering                            │
│          (src/features/engineer.py)                          │
│  • Temporal features (upload timing, age)                    │
│  • Text features (title/desc analysis)                       │
│  • Engagement metrics (ratios, scores)                       │
│  • Channel aggregations                                      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Processed Data Storage                        │
│           (data/processed/*.csv)                             │
│  • videos_engineered.csv                                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            Exploratory Data Analysis                         │
│              (notebooks/*.ipynb)                             │
│  • Data quality checks                                       │
│  • Pattern discovery                                         │
│  • Visualization                                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                Model Development                             │
│              (src/models/*.py)                               │
│  • Feature selection                                         │
│  • Model training                                            │
│  • Model evaluation                                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   Deployment                                 │
│  • Model serving                                             │
│  • API endpoints                                             │
│  • Monitoring                                                │
└─────────────────────────────────────────────────────────────┘
```

## Module Design

### src/data/
**Purpose**: Data collection and acquisition

**Key Components:**
- `youtube_collector.py`: Core YouTube API interaction
  - `YouTubeCollector` class: Manages API calls, quota, error handling
  - Methods: `get_channel_videos()`, `get_video_metadata()`, `collect_from_channels()`
  
- `collect.py`: CLI script for running data collection
  - Argument parsing
  - Logging setup
  - Error handling and reporting

**Design Principles:**
- Class-based design for state management (quota tracking)
- Separation of concerns (API calls vs. CLI logic)
- Robust error handling with retries
- Rate limiting to respect API quotas

### src/features/
**Purpose**: Transform raw data into ML-ready features

**Key Components:**
- `engineer.py`: Feature engineering logic
  - `VideoFeatureEngineer` class: Stateful feature transformer
  - Methods for each feature group (temporal, text, engagement, channel)
  - Modular design allows easy feature addition/removal
  
- `build_features.py`: CLI script for feature engineering pipeline

**Design Principles:**
- Reproducibility: Collection date parameter for temporal features
- Modularity: Each feature group in separate method
- Extensibility: Easy to add new feature groups
- Type safety: Full type hints throughout

### src/models/
**Purpose**: Model training and prediction (future work)

**Planned Components:**
- `train.py`: Model training pipeline
- `predict.py`: Inference script
- `evaluate.py`: Model evaluation metrics

### src/visualization/
**Purpose**: Create plots and dashboards (future work)

**Planned Components:**
- `plots.py`: Reusable plotting functions
- `dashboard.py`: Interactive visualization dashboard

## Data Flow

1. **Collection**: YouTube API → Raw CSV
   ```
   channels.csv → YouTubeCollector → videos_metadata.csv
   ```

2. **Feature Engineering**: Raw CSV → Processed CSV
   ```
   videos_metadata.csv → VideoFeatureEngineer → videos_engineered.csv
   ```

3. **EDA**: Processed CSV → Insights
   ```
   videos_engineered.csv → Jupyter Notebooks → Visualizations/Insights
   ```

4. **Modeling**: Processed CSV → Trained Model
   ```
   videos_engineered.csv → Model Training → Saved Model
   ```

## Configuration Management

### Environment Variables (.env)
- API keys and secrets
- Service endpoints
- Configuration parameters

**Pattern:**
```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("YOUTUBE_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY missing")
```

### Project Configuration (pyproject.toml)
- Package dependencies
- Tool configurations (black, ruff, pytest)
- Project metadata

## Testing Strategy

### Unit Tests
- Location: `tests/`
- Framework: pytest
- Coverage target: >80% for src/ modules
- Mocking: External API calls mocked

**Test Categories:**
1. **Data Collection Tests**: API interaction, error handling, quota tracking
2. **Feature Engineering Tests**: Feature calculations, data transformations
3. **Integration Tests**: End-to-end pipeline tests

### Test Fixtures (tests/conftest.py)
- Sample data for testing
- Mock API responses
- Common test utilities

## Logging and Monitoring

### Logging
- Python's `logging` module
- Log levels: DEBUG, INFO, WARNING, ERROR
- Outputs: Console + file (data_collection.log, feature_engineering.log)

**Pattern:**
```python
import logging

logger = logging.getLogger(__name__)
logger.info("Processing started")
logger.error(f"Error occurred: {error}")
```

### Monitoring (Future)
- MLflow for experiment tracking
- Weights & Biases for model monitoring
- API usage dashboards

## Dependencies

### Core Dependencies
- `pandas`: Data manipulation
- `google-api-python-client`: YouTube API
- `python-dotenv`: Environment management
- `numpy`: Numerical operations

### Development Dependencies
- `pytest`: Testing framework
- `black`: Code formatting
- `ruff`: Linting
- `mypy`: Type checking
- `jupyter`: Notebooks

## Security Considerations

1. **Secrets Management**
   - Never commit .env file
   - Use environment variables only
   - API keys rotated regularly

2. **Data Privacy**
   - No personal data collected
   - Public video metadata only
   - Data anonymization in examples

3. **API Security**
   - Quota limits enforced
   - Rate limiting implemented
   - Error handling for API failures

## Performance Considerations

1. **Data Collection**
   - Batch API requests where possible
   - Respect rate limits (100ms between calls)
   - Quota tracking to avoid overuse

2. **Feature Engineering**
   - Vectorized operations (pandas/numpy)
   - Avoid Python loops for large datasets
   - Memory-efficient transformations

3. **Scalability**
   - Current design: 200-400 videos (MVP)
   - Future: Incremental collection for larger datasets
   - Consider database for >10k videos

## Future Enhancements

1. **Data Collection**
   - Incremental updates (only new videos)
   - Comment sentiment analysis
   - Transcript extraction

2. **Features**
   - NLP features (topic modeling, embeddings)
   - Thumbnail image features (OCR, color analysis)
   - Temporal trend features

3. **Modeling**
   - Engagement prediction (regression)
   - Success classification (classification)
   - Content recommendation (clustering)

4. **Deployment**
   - REST API for predictions
   - Scheduled data collection
   - Dashboard for insights

## Design Patterns Used

1. **Class-based Design**: YouTubeCollector, VideoFeatureEngineer
2. **Dependency Injection**: Pass API key, collection date as parameters
3. **Factory Pattern**: Feature creation methods
4. **Builder Pattern**: Feature engineering pipeline
5. **Repository Pattern**: Data access through consistent interface

## Coding Standards

See `CONTRIBUTING.md` for detailed coding standards and contribution guidelines.
