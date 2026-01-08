# Quick Reference Guide

## Adding New Channels Without Overwriting

### Method 1: Using Incremental Collection (Recommended)

1. **Add new channels** to `data/raw/channels.csv`:
```csv
channel_id,channel_name,subjects
UCuJqXbrblfKeO9fa82a7KHQ,Katfi Charif Zina,Natural Sciences
UC1OxQheqDv1l0l7sj3HHqQQ,Mr.Mansouri,French
UCnewchannel123,New Channel,Math  ← Add this line
```

2. **Run incremental collection**:
```bash
uv run python -m src.data.collect_incremental \
    --channels data/raw/channels.csv
```

3. **What happens**:
   - ✅ Creates backup: `videos_metadata_backup_YYYYMMDD_HHMMSS.csv`
   - ✅ Detects NEW channel (UCnewchannel123)
   - ✅ Collects only from new channel
   - ✅ Merges with existing 318 videos
   - ✅ Removes any duplicates
   - ✅ Saves updated dataset

### Method 2: Manual Merge

If you want more control:

```bash
# Collect from new channels to separate file
uv run python -m src.data.collect \
    --channels data/raw/new_channels.csv \
    --output data/raw/new_videos.csv

# Then merge manually with pandas
```

## Getting More Insights

### Collect Channel Statistics

```bash
uv run python -m src.data.collect_enhanced \
    --channels data/raw/channels.csv \
    --channel-stats-output data/raw/channel_statistics.csv
```

**Provides:**
- Subscriber counts
- Total video counts
- Total channel views
- Channel creation dates
- Channel descriptions

### Collect Comment Samples

```bash
uv run python -m src.data.collect_enhanced \
    --channels data/raw/channels.csv \
    --collect-comments \
    --comments-output data/raw/comments_sample.csv
```

**Provides:**
- Top 10 comments from 50 most-viewed videos
- Comment text, author, likes
- Useful for sentiment analysis

**Note**: Comments use more quota (~100 units per request)

## Understanding Video Counts

**Your current collection:**
- Natural Sciences: 125 videos (not all videos from channel)
- Mr. Mansouri: 193 videos
- Total: 318 videos

**Why not all videos?**
- YouTube API returns ~500 most recent videos max
- Private/unlisted videos excluded
- Very old videos may not be indexed

**To check channel's actual video count:**
```bash
# Check channel statistics
uv run python -m src.data.collect_enhanced \
    --channels data/raw/channels.csv \
    --channel-stats-output data/raw/channel_stats.csv

# View results
cat data/raw/channel_stats.csv
```

## Common Workflows

### Workflow 1: Initial Collection
```bash
# 1. Prepare channels.csv with 2-5 channels
# 2. Collect data
uv run python -m src.data.collect --channels data/raw/channels.csv

# 3. Engineer features
uv run python -m src.features.build_features \
    --input data/raw/videos_metadata.csv \
    --output data/processed/videos_engineered.csv

# 4. Analyze
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Workflow 2: Adding More Channels
```bash
# 1. Add new channels to channels.csv
# 2. Run incremental collection (auto-merges)
uv run python -m src.data.collect_incremental --channels data/raw/channels.csv

# 3. Re-engineer features with new data
uv run python -m src.features.build_features \
    --input data/raw/videos_metadata.csv \
    --output data/processed/videos_engineered.csv

# 4. Re-analyze with expanded dataset
```

### Workflow 3: Getting Deeper Insights
```bash
# 1. Collect enhanced data
uv run python -m src.data.collect_enhanced \
    --channels data/raw/channels.csv \
    --collect-comments \
    --channel-stats-output data/raw/channel_stats.csv \
    --comments-output data/raw/comments.csv

# 2. Analyze channel statistics
# 3. Perform sentiment analysis on comments
# 4. Compare channel performance
```

## Quick Commands

### Check Current Data
```bash
# Count videos
wc -l data/raw/videos_metadata.csv
# Or PowerShell:
Get-Content data/raw/videos_metadata.csv | Measure-Object -Line

# View sample
head data/raw/videos_metadata.csv

# Check channels collected
uv run python -c "import pandas as pd; df = pd.read_csv('data/raw/videos_metadata.csv'); print(df['channel_title'].value_counts())"
```

### Backup Data
```bash
# Manual backup
cp data/raw/videos_metadata.csv data/raw/videos_metadata_backup.csv

# Automatic backup (incremental collection does this)
uv run python -m src.data.collect_incremental --channels data/raw/channels.csv
```

### View Logs
```bash
# View collection log
tail -50 data_collection.log

# View enhanced collection log
tail -50 data_collection_enhanced.log

# View incremental collection log
tail -50 data_collection_incremental.log
```

## File Locations

```
data/
├── raw/
│   ├── channels.csv              ← Your channel list
│   ├── videos_metadata.csv       ← Main data file
│   ├── videos_metadata_backup_*.csv  ← Auto backups
│   ├── channel_statistics.csv    ← Channel stats (optional)
│   └── comments_sample.csv       ← Comments (optional)
└── processed/
    └── videos_engineered.csv     ← ML-ready features
```

## Next Steps After Collection

1. ✅ **Feature Engineering**
   ```bash
   uv run python -m src.features.build_features \
       --input data/raw/videos_metadata.csv \
       --output data/processed/videos_engineered.csv
   ```

2. ✅ **Exploratory Analysis**
   ```bash
   jupyter notebook notebooks/01_data_exploration.ipynb
   ```

3. ✅ **Identify Insights**
   - What video lengths perform best?
   - Optimal upload times?
   - Subject-specific patterns?
   - Exam content impact?

## Troubleshooting

**Problem**: "No new channels to collect"
- **Cause**: All channels in CSV already in dataset
- **Solution**: Either add different channels or use `--force-update`

**Problem**: "Only got 125 videos but channel has 300"
- **Cause**: YouTube API limitation (~500 videos max)
- **Solution**: This is expected - you have the most recent videos

**Problem**: "Quota exceeded"
- **Cause**: Used 10,000 daily API units
- **Solution**: Wait until next day or use `--max-quota` to limit usage

**Problem**: "Want to collect more data from same channel"
- **Cause**: Already have channel in dataset
- **Solution**: Use `--force-update` flag to re-collect everything

## Quick Tips

💡 **Always use incremental collection** when adding channels
💡 **Check quota usage** in logs before large collections
💡 **Start small** (5 channels) then expand
💡 **Collect comments** for deeper insights (but uses more quota)
💡 **Backup important datasets** before re-running collection
💡 **Run feature engineering** after each data update
