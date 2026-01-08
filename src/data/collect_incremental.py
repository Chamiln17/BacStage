"""
Incremental data collection - merge new channels without overwriting existing data.

This script:
1. Loads existing videos_metadata.csv
2. Collects data from new channels
3. Merges without duplicates
4. Saves updated dataset
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

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
            "YOUTUBE_API_KEY not found or invalid. " "Please set it in your .env file."
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
            logger.info(
                f"✓ Loaded existing data: {len(df)} videos from {len(df['channel_id'].nunique())} channels"
            )
            logger.info(f"  Existing channels: {df['channel_title'].unique().tolist()}")
            return df
        except Exception as e:
            logger.error(f"Error loading existing data: {e}")
            return pd.DataFrame()
    else:
        logger.info("No existing data found. Starting fresh collection.")
        return pd.DataFrame()


def identify_new_channels(
    channels_df: pd.DataFrame, existing_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Identify channels that are not in existing data.

    Args:
        channels_df: DataFrame with channels to collect
        existing_df: DataFrame with existing video data

    Returns:
        DataFrame with only new channels to collect
    """
    if existing_df.empty:
        logger.info("All channels are new (no existing data)")
        return channels_df

    existing_channel_ids = existing_df["channel_id"].unique()
    new_channels = channels_df[~channels_df["channel_id"].isin(existing_channel_ids)]

    if len(new_channels) == 0:
        logger.info("⚠️  No new channels to collect. All channels already in dataset.")
        logger.info(
            "   If you want to update existing channels, use --force-update flag"
        )
    else:
        logger.info(f"✓ Found {len(new_channels)} new channels to collect")
        logger.info(
            f"  New channels: {new_channels['channel_name'].tolist() if 'channel_name' in new_channels.columns else new_channels['channel_id'].tolist()}"
        )

    return new_channels


def merge_data(existing_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge new data with existing data, removing duplicates.

    Args:
        existing_df: Existing video data
        new_df: Newly collected video data

    Returns:
        Merged DataFrame without duplicates
    """
    if existing_df.empty:
        return new_df

    if new_df.empty:
        return existing_df

    # Combine dataframes
    combined = pd.concat([existing_df, new_df], ignore_index=True)

    # Remove duplicates based on video_id (keep first occurrence)
    initial_count = len(combined)
    combined = combined.drop_duplicates(subset=["video_id"], keep="first")
    duplicates_removed = initial_count - len(combined)

    if duplicates_removed > 0:
        logger.info(f"✓ Removed {duplicates_removed} duplicate videos")

    logger.info(f"✓ Merged dataset: {len(combined)} total videos")

    return combined


def create_backup(file_path: Path) -> None:
    """Create a timestamped backup of existing data."""
    if file_path.exists():
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}.csv"

        try:
            df = pd.read_csv(file_path)
            df.to_csv(backup_path, index=False)
            logger.info(f"✓ Created backup: {backup_path}")
        except Exception as e:
            logger.warning(f"Could not create backup: {e}")


def main() -> None:
    """Main entry point for incremental collection."""
    parser = argparse.ArgumentParser(
        description="Incremental data collection - merge new channels with existing data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add new channels to existing data
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv

  # Force update all channels (re-collect everything)
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv --force-update

  # Merge with custom output location
  uv run python -m src.data.collect_incremental --channels data/raw/channels.csv --output data/raw/my_videos.csv
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
        "--max-videos", type=int, default=None, help="Max videos per channel"
    )

    parser.add_argument(
        "--max-quota", type=int, default=8000, help="Maximum API quota to use"
    )

    parser.add_argument(
        "--force-update",
        action="store_true",
        help="Re-collect all channels (not just new ones)",
    )

    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Skip creating backup of existing data",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("Incremental YouTube Data Collection")
    logger.info("=" * 60)

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

    # Load channels to collect
    try:
        if not args.channels.exists():
            raise FileNotFoundError(f"Channel file not found: {args.channels}")

        channels_df = pd.read_csv(args.channels)
        logger.info(f"✓ Loaded {len(channels_df)} channels from {args.channels}")

        if "channel_id" not in channels_df.columns:
            raise ValueError("CSV must contain 'channel_id' column")

    except Exception as e:
        logger.error(f"Error loading channels: {e}")
        sys.exit(1)

    # Identify channels to collect
    if args.force_update:
        logger.info("--force-update flag set: collecting all channels")
        channels_to_collect = channels_df
    else:
        channels_to_collect = identify_new_channels(channels_df, existing_df)

    if channels_to_collect.empty:
        logger.info("Nothing to collect. Exiting.")
        sys.exit(0)

    # Initialize collector and collect data
    try:
        collector = YouTubeCollector(api_key)
        logger.info(f"\nCollecting data from {len(channels_to_collect)} channels...")

        new_df = collector.collect_from_channels(
            channels_to_collect,
            max_videos_per_channel=args.max_videos,
            max_quota=args.max_quota,
        )

    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)

    # Merge with existing data
    if not new_df.empty:
        logger.info("\n" + "=" * 60)
        logger.info("MERGING DATA")
        logger.info("=" * 60)

        merged_df = merge_data(existing_df, new_df)

        # Save merged data
        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(args.output, index=False)

        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"✓ Previous videos: {len(existing_df)}")
        logger.info(f"✓ New videos collected: {len(new_df)}")
        logger.info(f"✓ Total videos in dataset: {len(merged_df)}")
        logger.info(f"✓ Total channels: {merged_df['channel_id'].nunique()}")
        logger.info(f"✓ Saved to: {args.output}")
        logger.info(f"✓ Quota used: {collector.quota_used}")

    else:
        logger.warning("No new data collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
