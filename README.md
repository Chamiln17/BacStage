# Algerian Bac Educational Video Engagement Analysis

End-to-end machine learning project for analyzing YouTube engagement patterns in Algerian Bac educational content.

## Quick Start

### 1. Setup

```bash
# Create virtual environment and install dependencies using uv
uv venv
.venv\Scripts\activate  # On Windows (use source .venv/bin/activate on Linux/Mac)
uv sync --all-extras

# Configure API key
cp .env.example .env
# Edit .env and add your YouTube Data API v3 key
```

### 2. Prepare Channel Data

Create a CSV file with channel IDs in `data/raw/channels.csv`:

```csv
channel_id,channel_name,subjects
UCxxxxxx,Channel Name,Math
```

### 3. Run the Pipeline

```bash
# Full pipeline: collect data + engineer features + analyze
uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv
```

Or run steps individually:

```bash
# Collect data only
uv run python run_pipeline.py collect --channels data/raw/channels.csv

# Engineer features
uv run python run_pipeline.py engineer

# Quick analysis
uv run python run_pipeline.py analyze
```

### 4. Explore Data

```bash
jupyter notebook notebooks/
```

## Project Structure

```
SIC/
├── run_pipeline.py              # Unified CLI entry point
├── README.md                    # This file
├── QUICK_REFERENCE.md           # Quick commands reference
├── pyproject.toml               # Project configuration
├── data/
│   ├── README.md                # Data documentation
│   ├── raw/                     # Raw data from collection
│   │   ├── channels.csv         # Input: channel list
│   │   ├── videos_metadata.csv  # Video data with snapshots
│   │   ├── video_registry.csv   # Known video IDs
│   │   └── channel_statistics.csv
│   └── processed/               # Feature-engineered data
│       └── videos_engineered.csv
├── notebooks/                   # Jupyter notebooks
│   └── 01_data_exploration.ipynb
├── src/                         # Source code
│   ├── data/                    # Data collection
│   │   ├── youtube_collector.py # YouTube API wrapper
│   │   ├── video_registry.py    # Video ID registry
│   │   ├── storage.py           # Raw JSON utilities
│   │   └── collect.py           # Collection CLI
│   └── features/                # Feature engineering
│       ├── engineer.py
│       └── build_features.py
└── tests/                       # Unit tests (pytest)
```

## CLI Commands

All operations use `run_pipeline.py`:

| Command | Description |
|---------|-------------|
| `full-pipeline` | Run everything: collect + engineer + analyze |
| `collect` | Gather video metadata and channel statistics |
| `engineer` | Build ML features from raw data |
| `analyze` | Quick stats on collected data |

### Common Options

```bash
# Limit videos per channel
--max-videos 50

# Set API quota limit
--max-quota 5000

# Skip discovery (only update known videos)
--no-discover

# Keep only latest snapshot per video
--mode dedupe

# Include comment samples
--collect-comments
```

## Features

### Data Collection (Quota-Optimized)

- **Uploads Playlist Discovery**: 1 unit per 50 videos (vs 100 units with search.list)
- **Batched Enrichment**: 50 videos per API request
- **Video Registry**: Tracks known videos for incremental updates
- **Snapshot Tracking**: Enables time-series analysis
- **Channel Statistics**: Subscriber counts, view totals, etc.

### Feature Engineering

- **Temporal**: Upload timing, video age, seasonal patterns
- **Content**: Duration, title analysis, subject detection
- **Engagement**: Like/comment ratios, engagement scores
- **Channel**: Aggregated channel-level statistics

## API Quota Efficiency

YouTube Data API v3: 10,000 quota units per day

| Operation | Old Method | New Method | Savings |
|-----------|-----------|-----------|---------|
| Discover 200 videos | 400 units | 4 units | 100x |
| Enrich 200 videos | 200 units | 4 units | 50x |
| **Total (4 channels)** | **2,400 units** | **24 units** | **100x** |

This enables daily refreshes within the quota limit.

## Data Schema

### videos_metadata.csv
- `video_id`, `title`, `description`, `publish_date`
- `channel_id`, `channel_title`, `category_id`
- `duration_sec`, `view_count`, `like_count`, `comment_count`
- `snapshot_date`: Enables time-series tracking
- `run_id`: Links to collection run

### videos_engineered.csv
Additional ML features:
- `days_since_publish`, `publish_hour`, `is_evening_upload`
- `title_length`, `subject`, `is_exam_focused`, `tag_count`
- `like_ratio`, `engagement_score`, `engagement_category`
- `channel_video_count`, `channel_avg_views`

## Development

### Running Tests

```bash
uv run pytest
uv run pytest -v  # Verbose
uv run pytest --cov=src  # With coverage
```

### Code Quality

```bash
uv run black src/ tests/
uv run ruff check src/ tests/
uv run mypy src/
```

## Documentation

- **README.md** - Project overview (this file)
- **QUICK_REFERENCE.md** - Quick commands and workflows
- **data/README.md** - Data directory documentation

## License

MIT
