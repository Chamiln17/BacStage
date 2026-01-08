# Algerian Bac Educational Video Engagement Analysis

End-to-end machine learning project for analyzing YouTube engagement patterns in Algerian Bac educational content.

## Quick Start

### 1. Setup

```bash
# Create virtual environment and install dependencies using uv
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"

# Configure API key
cp .env.example .env
# Edit .env and add your YouTube Data API v3 key
```

### 2. Prepare Channel Data

Create a CSV file with channel IDs in `data/raw/channels.csv`:

```csv
channel_id,channel_name,subjects
UCxxxxxx,Channel Name,Math,Physics
```

See `algerian_bac_channels.csv.example` for template.

### 3. Run Data Collection Pipeline

```bash
# Step 1: Collect raw video metadata
uv run python -m src.data.collect \
    --channels data/raw/channels.csv \
    --output data/raw/videos_metadata.csv \
    --max-videos 100

# Step 2: Engineer features
uv run python -m src.features.build_features \
    --input data/raw/videos_metadata.csv \
    --output data/processed/videos_engineered.csv
```

### 4. Explore Data

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/
```

## Project Structure

```
SIC/
├── data/
│   ├── raw/              # Raw data from APIs
│   ├── processed/        # Cleaned, transformed data
│   └── external/         # Data from third-party sources
├── notebooks/            # Jupyter notebooks for exploration (01_*.ipynb)
├── src/                  # Source code for production use
│   ├── __init__.py
│   ├── data/            # Scripts to download/generate data
│   │   ├── youtube_collector.py
│   │   └── collect.py
│   ├── features/        # Scripts to create features
│   │   ├── engineer.py
│   │   └── build_features.py
│   ├── models/          # Scripts to train models
│   └── visualization/   # Scripts for visualizations
├── tests/               # Unit tests (pytest)
│   ├── test_youtube_collector.py
│   └── test_feature_engineer.py
├── .env.example         # Environment variables template
├── pyproject.toml       # Project configuration and dependencies
├── MVP-data-collection-EDA-guide.md  # Implementation guide
└── README.md            # This file
```

## Features

### Data Collection
- **YouTube API Integration**: Collect video metadata using YouTube Data API v3
- **Quota Management**: Automatic tracking and limits on API usage
- **Batch Processing**: Efficient collection from multiple channels
- **Error Handling**: Robust handling of API errors and rate limits

### Feature Engineering
- **Temporal Features**: Upload timing, video age, seasonal patterns
- **Content Features**: Duration, title/description analysis, subject detection
- **Engagement Metrics**: Like/comment ratios, engagement scores
- **Channel Features**: Aggregated channel-level statistics

### Code Quality
- **Type Hints**: Full type annotations for all functions
- **Testing**: Comprehensive unit test coverage with pytest
- **Logging**: Structured logging for debugging and monitoring
- **Documentation**: Google-style docstrings throughout

## API Quota Notes

- YouTube Data API v3: 10,000 quota units per day
- Quota costs:
  - `search.list`: ~100 units per request
  - `videos.list`: ~1 unit per video
- Default limit: 8,000 units (reserves buffer for other operations)

## Data Schema

### Raw Video Metadata (`data/raw/videos_metadata.csv`)
- `video_id`, `title`, `description`, `publish_date`
- `channel_id`, `channel_title`, `category_id`
- `duration_iso`, `duration_sec`
- `view_count`, `like_count`, `comment_count`
- `tags`, `thumbnail_url`

### Engineered Features (`data/processed/videos_engineered.csv`)
Additional features include:
- Temporal: `days_since_publish`, `publish_hour`, `is_evening_upload`, `is_weekday`
- Content: `title_length`, `subject`, `is_exam_focused`, `tag_count`
- Engagement: `like_ratio`, `comment_ratio`, `engagement_score`, `engagement_category`
- Channel: `channel_video_count`, `channel_avg_views`, `channel_age_days`

## Development

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_youtube_collector.py

# Run with verbose output
uv run pytest -v
```

### Code Formatting

```bash
# Format code with black
uv run black src/ tests/

# Lint with ruff
uv run ruff check src/ tests/

# Type checking with mypy
uv run mypy src/
```

## Next Steps

1. **Exploratory Data Analysis**: Use notebooks in `notebooks/` to explore patterns
2. **Model Development**: Build predictive models in `src/models/`
3. **Visualization**: Create insights dashboards in `src/visualization/`
4. **Deployment**: Package models for production use

See `MVP-data-collection-EDA-guide.md` for detailed implementation roadmap.

## Contributing

This project follows data science best practices:
- Use `uv` for all package management
- Write tests for all data transformations
- Use pathlib for file operations
- Never commit secrets (use `.env` file)
- Follow type hints and docstring conventions

## License

MIT
