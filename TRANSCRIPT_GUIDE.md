# YouTube Transcript Collection - Quick Start Guide

## 🎯 Your Mission
Collect transcripts for **4,316 YouTube videos** (Part 2 of split dataset).

## 📋 Prerequisites

1. **Install Tor Browser** and ensure it's running
2. **Python environment** with required packages
3. **Files received:**
   - `videos_part2_lahcen.csv` (in `data/processed/`)
   - Entire project codebase

## 🚀 How to Run

### Step 1: Navigate to project directory
```bash
cd path/to/SIC
```

### Step 2: Activate environment
```bash
# Windows PowerShell
& .venv/Scripts/Activate.ps1

# OR if using uv (recommended)
# No activation needed, just use 'uv run'
```

### Step 3: Start collection
```bash
uv run python src/data/transcript_collector.py \
  --input data/processed/videos_part2_lahcen.csv \
  --output data/processed/transcripts_part2.csv \
  --proxy "socks5h://127.0.0.1:9150" \
  --workers 3
```

**Important flags:**
- `--workers 3`: Uses 3 parallel workers (faster, recommended)
- `--proxy`: Routes through Tor for anonymity

## ⏱️ Expected Duration
- **With 3 workers:** ~10-15 hours
- **With 5 workers:** ~8-10 hours (riskier, more bot blocks)

## 📊 Monitor Progress

The script will:
- Show progress every 50 videos
- Save checkpoints automatically
- Resume from last checkpoint if stopped

**Check current stats:**
```bash
python scripts/check_stats.py
```

## 🔄 If Script Crashes

**No worries!** Just re-run the same command. It will resume automatically from the last checkpoint.

## ✅ When Complete

You'll have a file: `data/processed/transcripts_part2.csv`

**Send this file back** along with the checkpoint file:
- `transcripts_part2.csv` (results)
- `transcripts_part2_checkpoint.csv` (backup)

## ⚠️ Troubleshooting

### Script stuck?
- Tor is slow (~30-60s per video is normal)
- Check that Tor Browser is running

### Too many errors?
- Reduce workers: `--workers 2` or `--workers 1`
- Check Tor connection
