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
├── IMPLEMENTATION_GUIDE.md  # Detailed implementation guide
├── QUICK_REFERENCE.md       # Quick command reference
└── README.md                # This file
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

### Environment Setup

```bash
# Clone repository
git clone <repository-url>
cd SIC

# Setup virtual environment
uv venv
.venv\Scripts\activate  # Windows (or source .venv/bin/activate on Linux/Mac)
uv sync --all-extras

# Configure environment
cp .env.example .env
# Edit .env and add your YOUTUBE_API_KEY
```

### Running Tests

```bash
# Run all tests with coverage
uv run pytest

# Run specific test file
uv run pytest tests/test_youtube_collector.py

# Run with verbose output
uv run pytest -v

# Generate coverage report
uv run pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code (required before committing)
uv run black src/ tests/

# Check formatting without changes
uv run black --check src/ tests/

# Lint code
uv run ruff check src/ tests/

# Auto-fix linting issues
uv run ruff check --fix src/ tests/

# Type checking
uv run mypy src/
```

### Coding Standards

This project follows strict data science best practices:

**Package Management:**
- Use `uv` exclusively for all package operations
- Never use `pip` or `conda` directly

**Code Style:**
- **Type Hints**: Required for all function signatures in `src/`
- **Docstrings**: Google-style docstrings for all functions
- **Formatting**: Black (88 char line length)
- **Linting**: Ruff (configured in `pyproject.toml`)
- **Imports**: Use `pathlib.Path` for all file operations

**Configuration:**
- Never hardcode secrets or API keys
- Use `.env` file for all configuration
- Load with `python-dotenv`

**Project Organization:**
- `notebooks/`: Exploration only (name as `01_name.ipynb`)
- `src/`: Production code only
- `tests/`: Unit tests for all `src/` modules
- `data/`: Data files (gitignored)

**Example Code Pattern:**
```python
from pathlib import Path
import os
from dotenv import load_dotenv
import logging

logger = logging.getLogger(__name__)

def process_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    Process raw data and save results.
    
    Args:
        input_path: Path to input CSV file
        output_path: Path to save processed data
        
    Returns:
        Processed DataFrame
        
    Raises:
        FileNotFoundError: If input file doesn't exist
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Processing data from {input_path}")
    # ... processing logic
    return df
```

### Testing Guidelines

- Write tests for all data transformations
- Mock external API calls
- Use fixtures from `tests/conftest.py`
- Aim for >80% code coverage
- Test both success and error cases

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/your-feature-name

# Make changes and test
uv run pytest
uv run black src/ tests/
uv run ruff check src/ tests/

# Commit changes
git add .
git commit -m "Add feature: description"

# Push to remote
git push origin feature/your-feature-name
```

**Commit Message Guidelines:**
- Use clear, descriptive messages
- Focus on "why" not "what"
- Examples: "Add temporal feature engineering", "Fix quota tracking bug"

### Troubleshooting

**Import errors:**
```bash
# Reinstall package in development mode
uv sync --all-extras
```

**API quota exceeded:**
- Quota resets daily at midnight PST
- Check `data_collection.log` for quota usage
- Use `--max-quota` flag to limit usage

**Tests failing:**
```bash
# Run specific test with verbose output
uv run pytest tests/test_youtube_collector.py -v -s

# Check test coverage
uv run pytest --cov=src --cov-report=term-missing
```

## Architecture

### System Overview

The project follows an end-to-end ML pipeline with clear separation of concerns:

```
YouTube API → Data Collection → Feature Engineering → EDA → Modeling → Deployment
```

**Pipeline Flow:**

1. **Data Collection** (`src/data/`)
   - YouTube API integration via `YouTubeCollector` class
   - Channel video enumeration and metadata extraction
   - Automatic quota management and rate limiting
   - Output: `data/raw/videos_metadata.csv`

2. **Feature Engineering** (`src/features/`)
   - Transform raw metadata via `VideoFeatureEngineer` class
   - Create temporal, text, engagement, and channel features
   - Output: `data/processed/videos_engineered.csv`

3. **Exploratory Analysis** (`notebooks/`)
   - Data quality checks and pattern discovery
   - Statistical analysis and visualizations
   - Insight generation for content creators

4. **Model Development** (`src/models/`) - Future
   - Feature selection and model training
   - Performance evaluation and tuning
   - Model persistence and versioning

### Module Design

**`src/data/youtube_collector.py`**
- **Purpose**: YouTube API interaction and data collection
- **Key Class**: `YouTubeCollector`
  - State management (quota tracking)
  - Methods: `get_channel_videos()`, `get_video_metadata()`, `collect_from_channels()`
- **Design**: Class-based for state, separation of API calls from CLI logic

**`src/features/engineer.py`**
- **Purpose**: Transform raw data into ML-ready features
- **Key Class**: `VideoFeatureEngineer`
  - Modular feature groups: temporal, text, engagement, channel
  - Reproducible transformations with collection date parameter
- **Design**: Each feature group in separate method for easy extension

**`src/data/collect.py` & `src/features/build_features.py`**
- **Purpose**: CLI interfaces for data pipeline
- **Features**: Argument parsing, logging, error handling
- **Design**: Thin wrappers around core classes

### Data Flow

```
channels.csv
    ↓
YouTubeCollector.collect_from_channels()
    ↓
data/raw/videos_metadata.csv
    ↓
VideoFeatureEngineer.fit_transform()
    ↓
data/processed/videos_engineered.csv
    ↓
Jupyter Notebooks (EDA)
    ↓
Insights & Models
```

### Design Patterns

**1. Class-Based State Management**
```python
collector = YouTubeCollector(api_key)
# Quota tracking persists across calls
videos_df = collector.collect_from_channels(channels_df)
```

**2. Dependency Injection**
```python
engineer = VideoFeatureEngineer(collection_date=custom_date)
# Reproducible feature engineering
```

**3. Factory Pattern for Features**
```python
# Each feature group has its own creation method
engineer._create_temporal_features()
engineer._create_text_features()
engineer._create_engagement_features()
```

**4. Configuration via Environment**
```python
# No hardcoded secrets
load_dotenv()
api_key = os.getenv('YOUTUBE_API_KEY')
```

### Key Design Decisions

**Package Management**: `uv` instead of `pip`
- Faster dependency resolution
- Better lock file support
- Integrated virtual environment management

**Configuration**: `.env` instead of `config.py`
- Secrets never in code
- Environment-specific settings
- Standard pattern across projects

**Testing**: Comprehensive pytest suite
- Mock external API calls
- Fixtures for test data
- ~85% code coverage on core modules

**Type Safety**: Full type hints throughout `src/`
- Catches errors at development time
- Better IDE support
- Self-documenting code

## Project Evolution

This project was refactored from a monolithic script (`extract_raw_data.py`) to a modular, production-ready codebase following data science best practices.

**Key Improvements:**
- ✅ Modular class-based design (reusable components)
- ✅ Comprehensive testing (19 unit tests)
- ✅ Type safety (100% type hints in `src/`)
- ✅ Modern tooling (`uv`, `black`, `ruff`, `mypy`)
- ✅ Security (`.env` for secrets)
- ✅ Documentation (Google-style docstrings)

## Next Steps

1. **Data Collection**: Configure API key and collect your first dataset
2. **Exploratory Analysis**: Use `notebooks/01_data_exploration.ipynb` to explore patterns
3. **Feature Engineering**: Run the full pipeline to create ML-ready features
4. **Model Development**: Build predictive models in `src/models/`
5. **Deployment**: Package models for production use

## Documentation

- **📖 README.md** (this file) - Complete project overview and architecture
- **📚 IMPLEMENTATION_GUIDE.md** - Step-by-step implementation walkthrough
- **⚡ QUICK_REFERENCE.md** - Quick commands and common workflows
- **🔧 TROUBLESHOOTING.md** - Common errors and solutions
- **📁 data/README.md** - Data directory structure and tips
- **📓 notebooks/README.md** - Notebook usage guidelines

**Note:** All logs are saved to `logs/` directory for better organization

## License

MIT
