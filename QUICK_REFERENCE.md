# Quick Reference Guide

## Unified CLI Commands

All pipeline operations use a single entry point: `run_pipeline.py`

### Full Pipeline (Recommended)

Run everything in one command: collect, filter, engineer features, and analyze.

```bash
# With Bac 3AS filtering (recommended)
uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv --filter

# Without filtering
uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv
```

### Individual Commands

```bash
# Collect data (discovery + enrichment + channel stats)
uv run python run_pipeline.py collect --channels data/raw/channels.csv

# Apply data-driven Bac 3AS filter
uv run python run_pipeline.py filter_data

# Build ML features from collected data
uv run python run_pipeline.py engineer

# Quick analysis of collected data
uv run python run_pipeline.py analyze
```

## Common Workflows

### Workflow 1: Initial Collection with Filtering

```bash
# 1. Prepare channels.csv with your channels
# 2. Run full pipeline with filtering
uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv --filter

# 3. Explore data in notebook
jupyter notebook notebooks/01_data_exploration.ipynb
```

### Workflow 2: Daily Updates

```bash
# Just refresh stats for known videos (skip discovery)
uv run python run_pipeline.py collect --channels data/raw/channels.csv --no-discover
```

### Workflow 3: Adding New Channels

```bash
# 1. Add new channels to data/raw/channels.csv
# 2. Run collection (will discover new videos automatically)
uv run python run_pipeline.py collect --channels data/raw/channels.csv

# 3. Re-run filter and engineer features
uv run python run_pipeline.py filter_data --force-discovery
uv run python run_pipeline.py engineer --input data/processed/videos_bac_only.csv
```

### Workflow 4: Re-run Filtering Only

```bash
# Skip discovery (use cached channel priors and TF-IDF terms)
uv run python run_pipeline.py filter_data --skip-discovery

# Force re-discovery (quarterly maintenance)
uv run python run_pipeline.py filter_data --force-discovery

# Generate validation samples for manual review
uv run python run_pipeline.py filter_data --validate
```

## Collection Options

```bash
# Limit videos per channel
uv run python run_pipeline.py collect --channels data/raw/channels.csv --max-videos 50

# Set quota limit
uv run python run_pipeline.py collect --channels data/raw/channels.csv --max-quota 5000

# Skip discovery (only update known videos)
uv run python run_pipeline.py collect --channels data/raw/channels.csv --no-discover

# Skip channel statistics
uv run python run_pipeline.py collect --channels data/raw/channels.csv --no-channel-stats

# Keep only latest snapshot per video (smaller file)
uv run python run_pipeline.py collect --channels data/raw/channels.csv --mode dedupe

# Include comment samples (uses more quota)
uv run python run_pipeline.py collect --channels data/raw/channels.csv --collect-comments

# Skip backup creation
uv run python run_pipeline.py collect --channels data/raw/channels.csv --no-backup
```

## Filtering Options (filter_data)

```bash
# Run full filter (discovery + filter)
uv run python run_pipeline.py filter_data

# Skip discovery if artifacts already exist
uv run python run_pipeline.py filter_data --skip-discovery

# Force re-run discovery even if artifacts exist
uv run python run_pipeline.py filter_data --force-discovery

# Generate validation samples after filtering
uv run python run_pipeline.py filter_data --validate

# Use custom config file
uv run python run_pipeline.py filter_data --config config/my_config.yaml

# Specify custom input/output
uv run python run_pipeline.py filter_data --input data/raw/videos_metadata.csv --output-dir data/processed
```

The filter automatically:
- Builds channel priors (which channels are Bac-heavy)
- Discovers TF-IDF keywords (high-signal Bac terms)
- Applies balanced filtering rules
- Generates validation samples (with `--validate`)

## File Locations

```
data/
├── raw/
│   ├── channels.csv              <- Your channel list (input)
│   ├── videos_metadata.csv       <- Video data with snapshots
│   ├── video_registry.csv        <- Known video IDs
│   ├── channel_statistics.csv    <- Channel stats
│   └── comments_sample.csv       <- Comments (optional)
├── processed/
│   ├── channel_priors.csv        <- Auto-computed channel Bac ratios
│   ├── tfidf_bac_terms.json      <- Auto-discovered Bac keywords
│   ├── videos_bac_balanced.csv   <- All videos with filter columns
│   ├── videos_bac_only.csv       <- Bac 3AS videos only (for ML)
│   ├── videos_rejected.csv       <- Rejected videos (for analysis)
│   └── videos_engineered.csv     <- ML-ready features
├── validation/
│   └── sample_for_manual_review.csv <- Validation samples
└── config/
    └── filter_config.yaml        <- Filter tuning parameters
```

## Quick Analysis

```bash
# Get quick stats on your data
uv run python run_pipeline.py analyze
```

Output shows:
- Total videos and unique count
- Videos per channel
- Date range
- Snapshot count
- View statistics
- Registry status
- Feature engineering status

## Quota Efficiency

The pipeline is optimized for quota efficiency:

| Operation | Quota Cost |
|-----------|-----------|
| Discover 50 videos | 1 unit |
| Enrich 50 videos | 1 unit |
| Channel stats | 1 unit per channel |
| Comments | ~1 unit per video |

**Example**: 3 channels with 500 total videos = ~15 quota units

## Troubleshooting

**"No videos to enrich"**
- Run with discovery: remove `--no-discover` flag

**"Quota exceeded"**
- Wait until next day or use `--max-quota` to limit usage

**"Only got X videos but channel has Y"**
- YouTube API returns most recent ~500 videos max
- Private/unlisted videos are excluded

## Tips

- Use `full-pipeline` for first-time setup
- Use `collect --no-discover` for daily stat refreshes
- Use `--mode dedupe` if you don't need time-series tracking
- Check `analyze` output to verify collection worked
- Back up your data before major re-collections
