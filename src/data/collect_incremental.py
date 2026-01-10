"""
Incremental data collection with snapshot tracking.

This script:
1. Loads existing videos_metadata.csv
2. Uses video registry for efficient discovery
3. Collects/updates data using batched API calls
4. Appends new snapshots (enables time-series analysis)
5. Saves updated dataset

Supports two modes:
- append: Keep all snapshots (time-series tracking)
- dedupe: Keep only latest per video_id (smaller storage)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv

from src.data.storage import get_run_id, save_raw_response
from src.data.video_registry import VideoRegistry
from src.data.youtube_collector import YouTubeCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/data_collection_incremental.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)

# Fix console encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def get_api_key() -> str:
    """Retrieve YouTube API key from environment variable."""
    load_dotenv()
    api_key = os.getenv("YOUTUBE_API_KEY")

    if not api_key or api_key == "your_api_key_here":
        raise ValueError(
            "YOUTUBE_API_KEY not found or invalid. Please set it in your .env file."
        )

    return api_key


def load_existing_data(file_path: Path) -> pd.DataFrame:
    """
    Load existing video metadata if it exists.

    Returns:
        DataFrame with existing data, or empty DataFrame if file doesn't exist
    """
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            unique_channels = df["channel_id"].nunique() if "channel_id" in df.columns else 0
            logger.info(
                f"Loaded existing data: {len(df)} records from {unique_channels} channels"
            )
            
            # Check for snapshot_date column
            if "snapshot_date" in df.columns:
                unique_snapshots = df["snapshot_date"].nunique()
                logger.info(f"  Existing snapshots: {unique_snapshots}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return pd.DataFrame()
    else:
        logger.info("No existing data found. Starting fresh collection.")
        return pd.DataFrame()


def merge_data(
    existing_df: pd.DataFrame,
    new_df: pd.DataFrame,
    mode: str = "append",
) -> pd.DataFrame:
    """
    Merge new data with existing data.

    Args:
        existing_df: Existing video data
        new_df: Newly collected video data
        mode: Merge mode
            - 'append': Keep all snapshots (for time-series analysis)
            - 'dedupe': Keep only latest per video_id (smaller storage)

    Returns:
        Merged DataFrame
    """
    if existing_df.empty:
        return new_df

    if new_df.empty:
        return existing_df

    # Combine dataframes
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    if mode == "append":
        # Keep all records - this enables time-series analysis
        logger.info(f"Append mode: keeping all {len(combined)} records")
        
        # Ensure snapshot_date exists and is datetime
        if "snapshot_date" in combined.columns:
            combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
            
    elif mode == "dedupe":
        # Keep only latest record per video_id
        initial_count = len(combined)
        
        if "snapshot_date" in combined.columns:
            combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
            # Sort by snapshot_date descending, keep first (latest)
            combined = combined.sort_values("snapshot_date", ascending=False)
            combined = combined.drop_duplicates(subset=["video_id"], keep="first")
            combined = combined.sort_values("snapshot_date", ascending=True)
        else:
            # No snapshot_date, just dedupe keeping last
            combined = combined.drop_duplicates(subset=["video_id"], keep="last")
        
        duplicates_removed = initial_count - len(combined)
        logger.info(f"Dedupe mode: removed {duplicates_removed} older snapshots")

    logger.info(f"Merged dataset: {len(combined)} total records")
    return combined


def create_backup(file_path: Path) -> Optional[Path]:
    """Create a timestamped backup of existing data."""
    if file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}.csv"

        try:
            df = pd.read_csv(file_path)
            df.to_csv(backup_path, index=False)
            logger.info(f"Created backup: {backup_path}")
            return backup_path
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")
            return None
    return None


def main() -> None:
    """Main entry point for incremental collection."""
    parser = argparse.ArgumentParser(
        description="Incremental data collection with snapshot tracking",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Refresh stats for all known videos (daily run)
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv

  # Discover new videos and refresh all stats
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv --discover

  # Keep only latest snapshot per video (smaller file)
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv --mode dedupe

  # Save raw JSON responses for auditing
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv --save-raw
        """,
    )

    parser.add_argument(
        "--channels", type=Path, required=True, help="Path to channels CSV file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/videos_metadata.csv"),
        help="Output CSV file path (default: data/raw/videos_metadata.csv)",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/raw/video_registry.csv"),
        help="Path to video registry CSV (default: data/raw/video_registry.csv)",
    )

    parser.add_argument(
        "--max-videos", type=int, default=None, help="Max videos per channel"
    )

    parser.add_argument(
        "--max-quota", type=int, default=8000, help="Maximum API quota to use"
    )

    parser.add_argument(
        "--discover",
        action="store_true",
        help="Run discovery to find new videos (uses uploads playlist)",
    )

    parser.add_argument(
        "--mode",
        choices=["append", "dedupe"],
        default="append",
        help="Merge mode: 'append' keeps all snapshots, 'dedupe' keeps only latest (default: append)",
    )

    parser.add_argument(
        "--save-raw",
        action="store_true",
        help="Save raw API responses to data/raw/api_responses/",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup of existing data",
    )

    # Legacy flag for compatibility
    parser.add_argument(
        "--force-update",
        action="store_true",
        help="(Legacy) Same as --discover - re-discover all videos",
    )

    args = parser.parse_args()

    # Handle legacy flag
    if args.force_update:
        args.discover = True

    logger.info("=" * 60)
    logger.info("Incremental YouTube Data Collection")
    logger.info("=" * 60)

    # Generate run ID for this collection
    run_id = get_run_id()
    snapshot_date = datetime.now()
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Snapshot date: {snapshot_date.isoformat()}")

    # Get API key
    try:
        api_key = get_api_key()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Load existing data
    existing_df = load_existing_data(args.output)

    # Create backup if data exists and backup not disabled
    if not args.no_backup and not existing_df.empty:
        create_backup(args.output)

    # Load channels
    try:
        if not args.channels.exists():
            raise FileNotFoundError(f"Channel file not found: {args.channels}")

        channels_df = pd.read_csv(args.channels)
        logger.info(f"Loaded {len(channels_df)} channels from {args.channels}")

        if "channel_id" not in channels_df.columns:
            raise ValueError("CSV must contain 'channel_id' column")

    except Exception as e:
        logger.error(f"Error loading channels: {e}")
        sys.exit(1)

    # Initialize collector and registry
    try:
        collector = YouTubeCollector(api_key)
        registry = VideoRegistry(args.registry)
        registry.load()
        
        logger.info(f"Registry: {len(registry)} known videos")

    except Exception as e:
        logger.error(f"Error initializing: {e}")
        sys.exit(1)

    # Phase 1: Discovery (if requested or registry is empty)
    if args.discover or len(registry) == 0:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: DISCOVERY")
        logger.info("=" * 60)

        for idx, row in channels_df.iterrows():
            channel_id = row["channel_id"]
            channel_name = row.get("channel_name", channel_id)

            if collector.quota_used >= args.max_quota:
                logger.warning("Quota limit reached during discovery")
                break

            logger.info(f"[{idx+1}/{len(channels_df)}] Discovering videos for {channel_name}")

            try:
                video_ids = collector.get_channel_videos(
                    channel_id, max_videos=args.max_videos
                )
                
                if video_ids:
                    added = registry.add_videos(video_ids, channel_id, source="uploads_playlist")
                    logger.info(f"  Found {len(video_ids)} videos, {added} new")
                else:
                    logger.warning(f"  No videos found for {channel_name}")

            except Exception as e:
                logger.error(f"  Discovery failed for {channel_name}: {e}")
                continue

        # Save registry after discovery
        registry.save()
        logger.info(f"Registry updated: {len(registry)} total videos")

    # Phase 2: Enrichment
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: ENRICHMENT")
    logger.info("=" * 60)

    # Get video IDs to enrich
    channel_ids = channels_df["channel_id"].tolist()
    video_ids = registry.get_ids_for_enrichment(channel_ids=channel_ids)

    if not video_ids:
        logger.warning("No videos to enrich. Run with --discover to find videos.")
        sys.exit(0)

    logger.info(f"Enriching {len(video_ids)} videos...")

    # Calculate quota needed
    batches_needed = (len(video_ids) + 49) // 50
    quota_needed = batches_needed
    logger.info(f"Estimated quota: {quota_needed} units for enrichment")

    if collector.quota_used + quota_needed > args.max_quota:
        # Limit to what we can afford
        remaining_quota = args.max_quota - collector.quota_used
        max_videos_affordable = remaining_quota * 50
        video_ids = video_ids[:max_videos_affordable]
        logger.warning(f"Limiting to {len(video_ids)} videos due to quota constraints")

    # Fetch metadata
    try:
        new_df = pd.DataFrame(
            collector.get_videos_metadata_batch(
                video_ids,
                snapshot_date=snapshot_date,
                run_id=run_id,
            )
        )
    except Exception as e:
        logger.error(f"Error during enrichment: {e}")
        sys.exit(1)

    # Phase 3: Merge and Save
    if not new_df.empty:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: MERGE & SAVE")
        logger.info("=" * 60)

        merged_df = merge_data(existing_df, new_df, mode=args.mode)

        # Save merged data
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(args.output, index=False)

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Previous records: {len(existing_df)}")
        logger.info(f"New records collected: {len(new_df)}")
        logger.info(f"Total records in dataset: {len(merged_df)}")
        
        if "video_id" in merged_df.columns:
            logger.info(f"Unique videos: {merged_df['video_id'].nunique()}")
        if "channel_id" in merged_df.columns:
            logger.info(f"Unique channels: {merged_df['channel_id'].nunique()}")
        if "snapshot_date" in merged_df.columns:
            logger.info(f"Unique snapshots: {merged_df['snapshot_date'].nunique()}")
            
        logger.info(f"Saved to: {args.output}")
        logger.info(f"Quota used: {collector.quota_used}")

    else:
        logger.warning("No new data collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
