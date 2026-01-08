# Implementation Guide
## Algerian Bac Educational Video Engagement Analysis

This guide explains how the project is implemented, from data collection to analysis, following the actual codebase structure.

---

## Overview

This project analyzes YouTube engagement patterns for Algerian Bac educational content through a complete ML pipeline:

- **Goal**: Identify factors that drive video engagement (views, likes, comments)
- **Approach**: Collect metadata → Engineer features → Analyze patterns → Generate insights
- **Tech Stack**: Python, pandas, YouTube Data API, pytest, uv
- **Timeline**: 2-3 weeks for MVP

**Expected Output**: 200-400 videos with 30+ features, ready for modeling

---

## Phase 1: Environment Setup

### 1.1 YouTube API Setup

**Get API Credentials:**

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create new project: "algerian-bac-engagement"
3. Enable "YouTube Data API v3"
4. Create API Key under "Credentials"
5. Save key securely (never commit to git)

**API Quota:** 10,000 units/day
- `search.list`: ~100 units per request
- `videos.list`: ~1 unit per video
- Collection of 300 videos ≈ 3,500 units

### 1.2 Project Installation

```bash
# Clone and setup
git clone <repository-url>
cd SIC

# Create virtual environment with uv
uv venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac

# Install all dependencies
uv sync --all-extras

# Configure API key
cp .env.example .env
# Edit .env and add: YOUTUBE_API_KEY=your_key_here
```

**Verify Installation:**
```bash
# Test imports
uv run python -c "from src.data import YouTubeCollector; from src.features import VideoFeatureEngineer; print('✓ Setup complete')"

# Run tests
uv run pytest --co  # Discover tests
uv run pytest       # Run all tests
```

---

## Phase 2: Data Collection

### 2.1 Channel Curation

**Find Target Channels:**

Search YouTube for Algerian Bac content:
- Keywords: "Bac Algeria", "دروس باك الجزائر", "Bac 2025"
- Subjects: Math, Physics, Science, Arabic, Philosophy
- Look for: Consistent uploads, high engagement, clear subject focus

**Create Channel List:**

Create `data/raw/channels.csv`:
```csv
channel_id,channel_name,subjects
UCxxxxxx,Math Bac Channel,Math
UCyyyyyy,Physics Teacher,Physics
UCzzzzzz,Science DZ,Biology,Chemistry
```

**Finding Channel IDs:**
- Go to channel page on YouTube
- Check URL: `youtube.com/channel/CHANNEL_ID` or `youtube.com/@username`
- For @username, view page source and search for "channelId"

### 2.2 Running Data Collection

**Implementation: `src/data/collect.py`**

The collection script uses the `YouTubeCollector` class to:
1. Load channels from CSV
2. For each channel, enumerate all videos via `search.list` API
3. Fetch detailed metadata via `videos.list` API
4. Track quota usage to avoid exceeding limits
5. Save results to CSV with automatic rate limiting

**Basic Collection:**
```bash
uv run python -m src.data.collect \
    --channels data/raw/channels.csv \
    --output data/raw/videos_metadata.csv \
    --max-videos 50  # Limit per channel for testing
```

**Full Collection:**
```bash
# Collect all videos (respects quota limit)
uv run python -m src.data.collect \
    --channels data/raw/channels.csv \
    --output data/raw/videos_metadata.csv \
    --max-quota 8000  # Reserve buffer
```

**Under the Hood:**

```python
# How YouTubeCollector works (simplified)
from src.data import YouTubeCollector

collector = YouTubeCollector(api_key)

# Step 1: Get video IDs from channel
video_ids = collector.get_channel_videos(channel_id)
# → Uses YouTube search API with pagination

# Step 2: Get metadata for each video
for video_id in video_ids:
    metadata = collector.get_video_metadata(video_id)
    # → Returns dict with title, views, likes, duration, etc.
    
# Step 3: Automatic quota tracking
print(f"Quota used: {collector.quota_used}/10000")
```

**Output Format:**

`data/raw/videos_metadata.csv` contains:
- Basic info: `video_id`, `title`, `description`, `publish_date`
- Channel: `channel_id`, `channel_title`, `category_id`
- Stats: `view_count`, `like_count`, `comment_count`
- Content: `duration_sec`, `tags`, `thumbnail_url`

### 2.3 Monitoring Collection

**Check Logs:**
```bash
# View collection log
tail -f data_collection.log

# Check quota usage
grep "quota" data_collection.log
```

**Troubleshooting:**
- **API errors**: Check API key in `.env` file
- **Quota exceeded**: Wait until next day (resets midnight PST) or reduce `--max-videos`
- **Empty results**: Verify channel IDs are correct

---

## Phase 3: Feature Engineering

### 3.1 Feature Categories

The `VideoFeatureEngineer` class creates four feature groups:

**1. Temporal Features** (when was video uploaded)
- `days_since_publish`: Age of video
- `publish_hour`: Hour of day (0-23)
- `publish_day_of_week`: Monday-Sunday
- `is_evening_upload`: Binary (5-9 PM = 1)
- `is_weekday`: Binary (Mon-Fri = 1)

**2. Text Features** (from title/description)
- `title_length`, `description_length`: Character counts
- `title_word_count`: Number of words
- `tag_count`: Number of tags
- `subject`: Detected subject (Math, Physics, Science, Arabic, etc.)
- `is_exam_focused`: Binary (contains "bac", "exam", "2025", etc.)

**3. Engagement Features** (how viewers interact)
- `like_ratio`: likes / views
- `comment_ratio`: comments / views
- `engagement_score`: weighted combination (likes + 2×comments) / views
- `engagement_category`: Low / Medium / High (tertile split)

**4. Channel Features** (aggregated stats)
- `channel_video_count`: Videos in dataset from this channel
- `channel_avg_views`: Average views for channel
- `channel_avg_engagement`: Average engagement for channel
- `channel_age_days`: Days since oldest video

### 3.2 Running Feature Engineering

**Implementation: `src/features/build_features.py`**

```bash
uv run python -m src.features.build_features \
    --input data/raw/videos_metadata.csv \
    --output data/processed/videos_engineered.csv
```

**What Happens:**

1. **Load raw data**: Read `videos_metadata.csv`
2. **Clean data**: Remove 0-view videos, fill missing values
3. **Create features**: Apply all four feature groups
4. **Select features**: Choose final set for modeling
5. **Save results**: Write to `videos_engineered.csv`

**Under the Hood:**

```python
from src.features import VideoFeatureEngineer

engineer = VideoFeatureEngineer(collection_date=datetime.now())

# Transform raw data
videos_engineered = engineer.fit_transform(videos_raw)
# → Applies all feature groups

# Select final features
videos_final = engineer.select_features(videos_engineered, include_target=True)
# → Returns cleaned dataset ready for modeling
```

**Custom Collection Date:**
```python
# For reproducible analysis
from datetime import datetime
engineer = VideoFeatureEngineer(
    collection_date=datetime(2024, 6, 1)
)
```

### 3.3 Feature Engineering Output

`data/processed/videos_engineered.csv` contains:
- All original metadata
- All engineered features (30+ columns)
- Target variable: `engagement_category`

**Example row:**
```csv
video_id,title,subject,duration_sec,view_count,like_ratio,engagement_score,engagement_category,is_exam_focused,is_evening_upload
vid123,Math Bac 2025,Math,630,5000,0.05,0.068,High,1,1
```

---

## Phase 4: Exploratory Data Analysis

### 4.1 Using Jupyter Notebooks

**Launch Jupyter:**
```bash
jupyter notebook notebooks/
```

**Start with:** `01_data_exploration.ipynb`

This notebook demonstrates:
1. Loading and inspecting data
2. Summary statistics
3. Distribution visualizations
4. Correlation analysis

### 4.2 Key Analysis Questions

**From the engineered features, explore:**

1. **Video Length Impact**
   - Do shorter videos get better engagement?
   - What's the optimal duration for Bac content?

2. **Upload Timing**
   - Which days/hours perform best?
   - Does evening upload improve views?

3. **Subject Variations**
   - Which subjects are most popular?
   - Do different subjects have different engagement patterns?

4. **Exam Content**
   - Do exam-focused videos perform better?
   - Seasonal patterns around exam periods?

5. **Channel Effects**
   - Does channel size correlate with engagement?
   - New vs. established channels?

### 4.3 Example Analysis

**Correlation Analysis:**
```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load engineered data
df = pd.read_csv('data/processed/videos_engineered.csv')

# Check correlations with engagement
correlations = df[['duration_sec', 'title_length', 'is_exam_focused', 
                   'is_evening_upload', 'engagement_score']].corr()

# Visualize
sns.heatmap(correlations, annot=True, cmap='coolwarm')
plt.title('Feature Correlations')
plt.show()
```

**Expected Insights:**
- Shorter videos (6-10 min) → higher completion
- Evening uploads (5-9 PM) → more initial views
- Exam-focused content → 40-60% more engagement
- Math/Physics → higher like ratios than other subjects

---

## Phase 5: Model Development (Future)

### 5.1 Planned Models

**1. Engagement Prediction** (Regression)
- **Input**: Video metadata + temporal features
- **Output**: Predicted engagement score
- **Use**: Help creators estimate video performance

**2. Success Classification** (Classification)
- **Input**: All engineered features
- **Output**: Low / Medium / High engagement
- **Use**: Identify factors of successful videos

**3. Content Recommendation** (Clustering)
- **Input**: Text features + engagement
- **Output**: Video clusters by topic and performance
- **Use**: Find content gaps and opportunities

### 5.2 Implementation Structure

**File Organization:**
```
src/models/
├── __init__.py
├── train.py           # Model training pipeline
├── predict.py         # Inference script
├── evaluate.py        # Metrics and evaluation
└── utils.py           # Helper functions
```

**Training Pattern:**
```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Load features
X = features[feature_columns]
y = features['engagement_score']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
score = model.score(X_test, y_test)
print(f"R² Score: {score:.3f}")
```

---

## Implementation Details

### Code Quality Standards

**Type Hints (Required):**
```python
def process_video(video_id: str, metadata: Dict) -> pd.DataFrame:
    """Process single video metadata."""
    ...
```

**Logging (Not print):**
```python
import logging
logger = logging.getLogger(__name__)

logger.info("Processing started")
logger.warning("Quota limit approaching")
logger.error("API call failed")
```

**Path Management:**
```python
from pathlib import Path

# Good
data_path = Path("data") / "raw" / "videos.csv"

# Bad
data_path = "data/raw/videos.csv"
```

**Configuration:**
```python
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("YOUTUBE_API_KEY")
if not api_key:
    raise ValueError("API key not found")
```

### Testing Strategy

**Unit Tests:**
- Test each feature engineering method independently
- Mock YouTube API calls
- Use fixtures for test data

**Run Tests:**
```bash
# All tests
uv run pytest

# Specific test file
uv run pytest tests/test_youtube_collector.py

# With coverage
uv run pytest --cov=src --cov-report=html
```

**Test Example:**
```python
def test_temporal_features():
    """Test temporal feature creation."""
    engineer = VideoFeatureEngineer(collection_date=datetime(2024, 6, 1))
    result = engineer._create_temporal_features(sample_data)
    
    assert 'publish_hour' in result.columns
    assert result.loc[0, 'is_evening_upload'] == 1  # 18:00
    assert result.loc[1, 'is_evening_upload'] == 0  # 14:30
```

### Performance Optimization

**1. Batch API Calls:**
```python
# videos.list supports up to 50 IDs per request
batch_size = 50
for i in range(0, len(video_ids), batch_size):
    batch = video_ids[i:i+batch_size]
    response = youtube.videos().list(
        part='snippet,statistics',
        id=','.join(batch)
    ).execute()
```

**2. Vectorized Operations:**
```python
# Good (vectorized)
df['like_ratio'] = df['like_count'] / df['view_count']

# Bad (loop)
for idx, row in df.iterrows():
    df.loc[idx, 'like_ratio'] = row['like_count'] / row['view_count']
```

**3. Incremental Updates:**
```python
# For large datasets, process in chunks
for chunk in pd.read_csv('large_file.csv', chunksize=1000):
    processed = engineer.fit_transform(chunk)
    processed.to_csv('output.csv', mode='a', header=False)
```

---

## Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall package
uv sync --all-extras

# Check Python path
uv run python -c "import sys; print(sys.path)"
```

**2. API Quota Exceeded**
- **Solution**: Wait until next day or use `--max-videos` flag
- **Check**: `grep quota data_collection.log`
- **Reset**: Daily at midnight PST

**3. Empty Channel Results**
- **Cause**: Incorrect channel ID or private channel
- **Solution**: Verify channel exists and is public
- **Test**: Visit `youtube.com/channel/CHANNEL_ID`

**4. Feature Engineering Fails**
- **Cause**: Missing required columns in raw data
- **Solution**: Check `videos_metadata.csv` has all expected columns
- **Fix**: Re-run data collection

**5. Tests Failing**
```bash
# Run with verbose output
uv run pytest -v -s

# Run specific failing test
uv run pytest tests/test_feature_engineer.py::test_temporal_features -v
```

---

## Best Practices

### Data Collection
1. Start with small sample (`--max-videos 10`) to test
2. Monitor quota usage in logs
3. Save intermediate results frequently
4. Validate channel IDs before large collection

### Feature Engineering
1. Keep raw data unchanged (read-only)
2. Document feature definitions
3. Version engineered datasets (use timestamps)
4. Test on sample before full dataset

### Development Workflow
1. Create feature branch for changes
2. Run tests before committing
3. Format code with black
4. Check linting with ruff
5. Update documentation

### Version Control
```bash
# Recommended .gitignore already includes:
# - data/*.csv (all data files)
# - .env (secrets)
# - __pycache__/ (Python cache)
# - *.log (log files)
```

---

## Next Steps

1. ✅ **Setup Complete**: Environment configured, tests passing
2. 🎯 **Collect Data**: Run data collection on curated channels
3. 📊 **Engineer Features**: Transform raw data to ML-ready format
4. 🔍 **Analyze**: Explore patterns in Jupyter notebooks
5. 🤖 **Model**: Build predictive models (future work)
6. 🚀 **Deploy**: Create API/dashboard (future work)

---

## Reference

- **YouTube Data API Docs**: https://developers.google.com/youtube/v3
- **Project Repository**: Check README.md for latest updates
- **Questions**: Open an issue or check existing documentation

**Happy analyzing! 🎓📊**
