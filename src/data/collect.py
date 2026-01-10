"""
Unified YouTube data collection with incremental updates and channel statistics.

This script combines all collection functionality:
- Video discovery via uploads playlist (quota-efficient)
- Batched video metadata enrichment
- Channel statistics collection
- Optional comment sampling
- Incremental updates with snapshot tracking
- Video registry for tracking known videos

Usage:
    uv run python -m src.data.collect --channels data/raw/channels.csv
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from dotenv import load_dotenv

from src.data.storage import get_run_id
from src.data.video_registry import VideoRegistry
from src.data.youtube_collector import YouTubeCollector

# Setup logging (console only - no file logging by default)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
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
            "YOUTUBE_API_KEY not found or invalid. "
            "Please set it in your .env file."
        )

    return api_key


def load_existing_data(file_path: Path) -> pd.DataFrame:
    """Load existing video metadata if it exists."""
    if file_path.exists():
        try:
            df = pd.read_csv(file_path)
            unique_channels = df["channel_id"].nunique() if "channel_id" in df.columns else 0
            logger.info(f"Loaded existing data: {len(df)} records from {unique_channels} channels")
            
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
        mode: 'append' keeps all snapshots, 'dedupe' keeps only latest

    Returns:
        Merged DataFrame
    """
    if existing_df.empty:
        return new_df

    if new_df.empty:
        return existing_df

    combined = pd.concat([existing_df, new_df], ignore_index=True)

    if mode == "append":
        logger.info(f"Append mode: keeping all {len(combined)} records")
        if "snapshot_date" in combined.columns:
            combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
            
    elif mode == "dedupe":
        initial_count = len(combined)
        
        if "snapshot_date" in combined.columns:
            combined["snapshot_date"] = pd.to_datetime(combined["snapshot_date"])
            combined = combined.sort_values("snapshot_date", ascending=False)
            combined = combined.drop_duplicates(subset=["video_id"], keep="first")
            combined = combined.sort_values("snapshot_date", ascending=True)
        else:
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


def get_video_comments_sample(
    youtube_client, video_id: str, max_comments: int = 10
) -> List[Dict]:
    """Get sample of top comments from a video."""
    comments = []
    try:
        request = youtube_client.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_comments,
            order="relevance",
            textFormat="plainText",
        )
        response = request.execute()

        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "video_id": video_id,
                "comment_text": comment["textDisplay"],
                "author": comment["authorDisplayName"],
                "like_count": comment["likeCount"],
                "published_at": comment["publishedAt"],
            })
    except Exception as e:
        logger.debug(f"Could not fetch comments for {video_id}: {e}")

    return comments


def main() -> None:
    """Main entry point for unified data collection."""
    parser = argparse.ArgumentParser(
        description="YouTube data collection with incremental updates and channel statistics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic collection (discover + enrich + channel stats)
  uv run python -m src.data.collect --channels data/raw/channels.csv

  # Skip discovery (only refresh stats for known videos)
  uv run python -m src.data.collect --channels data/raw/channels.csv --no-discover

  # Keep only latest snapshot per video
  uv run python -m src.data.collect --channels data/raw/channels.csv --mode dedupe

  # Include comment samples
  uv run python -m src.data.collect --channels data/raw/channels.csv --collect-comments

  # Limit videos and quota
  uv run python -m src.data.collect --channels data/raw/channels.csv --max-videos 50 --max-quota 5000
        """,
    )

    parser.add_argument(
        "--channels", type=Path, required=True,
        help="Path to channels CSV file (must have channel_id column)"
    )

    parser.add_argument(
        "--output", type=Path, default=Path("data/raw/videos_metadata.csv"),
        help="Output path for video metadata (default: data/raw/videos_metadata.csv)"
    )

    parser.add_argument(
        "--channel-stats-output", type=Path, default=Path("data/raw/channel_statistics.csv"),
        help="Output path for channel statistics (default: data/raw/channel_statistics.csv)"
    )

    parser.add_argument(
        "--comments-output", type=Path, default=Path("data/raw/comments_sample.csv"),
        help="Output path for comment samples (default: data/raw/comments_sample.csv)"
    )

    parser.add_argument(
        "--registry", type=Path, default=Path("data/raw/video_registry.csv"),
        help="Path to video registry CSV (default: data/raw/video_registry.csv)"
    )

    parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max videos per channel (default: all)"
    )

    parser.add_argument(
        "--max-quota", type=int, default=8000,
        help="Maximum API quota to use (default: 8000)"
    )

    parser.add_argument(
        "--no-discover", action="store_true",
        help="Skip discovery phase (only enrich known videos from registry)"
    )

    parser.add_argument(
        "--no-channel-stats", action="store_true",
        help="Skip channel statistics collection"
    )

    parser.add_argument(
        "--mode", choices=["append", "dedupe"], default="append",
        help="Merge mode: 'append' keeps all snapshots, 'dedupe' keeps only latest (default: append)"
    )

    parser.add_argument(
        "--collect-comments", action="store_true",
        help="Also collect comment samples from top videos"
    )

    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip creating backup of existing data"
    )

    args = parser.parse_args()

    # Generate run metadata
    run_id = get_run_id()
    snapshot_date = datetime.now()

    logger.info("=" * 60)
    logger.info("YouTube Data Collection Pipeline")
    logger.info("=" * 60)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Snapshot date: {snapshot_date.isoformat()}")

    # Get API key
    try:
        api_key = get_api_key()
        logger.info("API key loaded successfully")
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        sys.exit(1)

    # Load existing data
    existing_df = load_existing_data(args.output)

    # Create backup if data exists
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
        youtube = collector.youtube
        registry = VideoRegistry(args.registry)
        registry.load()
        logger.info(f"Registry: {len(registry)} known videos")

    except Exception as e:
        logger.error(f"Error initializing: {e}")
        sys.exit(1)

    # ===== PHASE 1: CHANNEL STATISTICS =====
    channel_stats_list = []
    if not args.no_channel_stats:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: CHANNEL STATISTICS")
        logger.info("=" * 60)

        for idx, row in channels_df.iterrows():
            channel_id = row["channel_id"]
            channel_name = row.get("channel_name", channel_id)
            logger.info(f"[{idx+1}/{len(channels_df)}] Getting stats for {channel_name}")
            
            stats = collector.get_channel_info(channel_id)
            if stats:
                stats["snapshot_date"] = snapshot_date.isoformat()
                channel_stats_list.append(stats)

        if channel_stats_list:
            channel_stats_df = pd.DataFrame(channel_stats_list)
            args.channel_stats_output.parent.mkdir(parents=True, exist_ok=True)
            channel_stats_df.to_csv(args.channel_stats_output, index=False)
            logger.info(f"Saved channel statistics to: {args.channel_stats_output}")

    # ===== PHASE 2: DISCOVERY =====
    if not args.no_discover:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 2: DISCOVERY")
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

        registry.save()
        logger.info(f"Registry updated: {len(registry)} total videos")

    # ===== PHASE 3: ENRICHMENT =====
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 3: ENRICHMENT")
    logger.info("=" * 60)

    channel_ids = channels_df["channel_id"].tolist()
    video_ids = registry.get_ids_for_enrichment(channel_ids=channel_ids)

    if not video_ids:
        logger.warning("No videos to enrich. Make sure discovery ran or registry has videos.")
        sys.exit(0)

    logger.info(f"Enriching {len(video_ids)} videos...")

    # Calculate quota
    batches_needed = (len(video_ids) + 49) // 50
    quota_needed = batches_needed
    logger.info(f"Estimated quota: {quota_needed} units for enrichment")

    if collector.quota_used + quota_needed > args.max_quota:
        remaining_quota = args.max_quota - collector.quota_used
        max_videos_affordable = remaining_quota * 50
        video_ids = video_ids[:max_videos_affordable]
        logger.warning(f"Limiting to {len(video_ids)} videos due to quota constraints")

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

    # ===== PHASE 4: COMMENTS (optional) =====
    all_comments = []
    if args.collect_comments and not new_df.empty and collector.quota_used < args.max_quota - 500:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 4: COMMENT SAMPLES")
        logger.info("=" * 60)

        top_videos = new_df.nlargest(50, "view_count")

        for idx, (_, video_row) in enumerate(top_videos.iterrows()):
            if collector.quota_used >= args.max_quota - 100:
                logger.warning("Approaching quota limit, stopping comment collection")
                break

            video_id = video_row["video_id"]
            comments = get_video_comments_sample(youtube, video_id, max_comments=10)
            
            for comment in comments:
                comment["snapshot_date"] = snapshot_date.isoformat()
            
            all_comments.extend(comments)
            collector.quota_used += 1

            if (idx + 1) % 10 == 0:
                logger.info(f"Collected comments from {idx+1}/50 top videos")

        if all_comments:
            comments_df = pd.DataFrame(all_comments)
            args.comments_output.parent.mkdir(parents=True, exist_ok=True)
            comments_df.to_csv(args.comments_output, index=False)
            logger.info(f"Saved {len(comments_df)} comments to: {args.comments_output}")

    # ===== PHASE 5: MERGE & SAVE =====
    if not new_df.empty:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 5: MERGE & SAVE")
        logger.info("=" * 60)

        merged_df = merge_data(existing_df, new_df, mode=args.mode)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(args.output, index=False)

        # ===== SUMMARY =====
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Channels analyzed: {len(channel_stats_list)}")
        logger.info(f"Previous records: {len(existing_df)}")
        logger.info(f"New records collected: {len(new_df)}")
        logger.info(f"Total records in dataset: {len(merged_df)}")
        
        if "video_id" in merged_df.columns:
            logger.info(f"Unique videos: {merged_df['video_id'].nunique()}")
        if "channel_id" in merged_df.columns:
            logger.info(f"Unique channels: {merged_df['channel_id'].nunique()}")
        if "snapshot_date" in merged_df.columns:
            logger.info(f"Unique snapshots: {merged_df['snapshot_date'].nunique()}")
        if args.collect_comments:
            logger.info(f"Comments collected: {len(all_comments)}")
            
        logger.info(f"Saved to: {args.output}")
        logger.info(f"Total quota used: {collector.quota_used}")

    else:
        logger.warning("No new data collected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
