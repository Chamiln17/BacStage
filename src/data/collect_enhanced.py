"""
Enhanced data collection with additional insights.

Collects:
- Video metadata (using batched enrichment)
- Channel statistics
- Comment samples (optional)

Uses optimized quota-efficient methods:
- Uploads playlist for discovery (1 unit per 50 videos)
- Batched videos.list for enrichment (1 unit per 50 videos)
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd
from dotenv import load_dotenv

from src.data.storage import get_run_id
from src.data.video_registry import VideoRegistry
from src.data.youtube_collector import YouTubeCollector

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/data_collection_enhanced.log", encoding="utf-8"),
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
            "YOUTUBE_API_KEY not found or invalid. "
            "Please set it in your .env file. "
            "See .env.example for template."
        )

    return api_key


def get_video_comments_sample(
    youtube_client, video_id: str, max_comments: int = 10
) -> List[Dict]:
    """
    Get sample of top comments from a video.

    Args:
        youtube_client: YouTube API client
        video_id: Video ID
        max_comments: Number of comments to retrieve

    Returns:
        List of comment dictionaries

    Note:
        API quota cost: ~1 unit per request
    """
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
            comments.append(
                {
                    "video_id": video_id,
                    "comment_text": comment["textDisplay"],
                    "author": comment["authorDisplayName"],
                    "like_count": comment["likeCount"],
                    "published_at": comment["publishedAt"],
                }
            )
    except Exception as e:
        logger.debug(f"Could not fetch comments for {video_id}: {e}")
        # Comments might be disabled

    return comments


def main() -> None:
    """Main entry point for enhanced data collection CLI."""
    parser = argparse.ArgumentParser(
        description="Enhanced YouTube data collection with additional insights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic collection with channel stats
  uv run python -m src.data.collect_enhanced --channels data/raw/channels.csv

  # Include comment samples from top videos
  uv run python -m src.data.collect_enhanced --channels data/raw/channels.csv --collect-comments

  # Update registry for future incremental runs
  uv run python -m src.data.collect_enhanced --channels data/raw/channels.csv --update-registry
        """,
    )

    parser.add_argument(
        "--channels", type=Path, required=True, help="Path to channels CSV file"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw/videos_metadata.csv"),
        help="Output path for video metadata",
    )

    parser.add_argument(
        "--channel-stats-output",
        type=Path,
        default=Path("data/raw/channel_statistics.csv"),
        help="Output path for channel statistics",
    )

    parser.add_argument(
        "--comments-output",
        type=Path,
        default=Path("data/raw/comments_sample.csv"),
        help="Output path for comment samples",
    )

    parser.add_argument(
        "--registry",
        type=Path,
        default=Path("data/raw/video_registry.csv"),
        help="Path to video registry CSV",
    )

    parser.add_argument(
        "--max-videos", type=int, default=None, help="Max videos per channel"
    )

    parser.add_argument(
        "--max-quota", type=int, default=8000, help="Maximum API quota to use"
    )

    parser.add_argument(
        "--collect-comments",
        action="store_true",
        help="Also collect comment samples (uses more quota)",
    )

    parser.add_argument(
        "--update-registry",
        action="store_true",
        help="Update video registry with discovered video IDs",
    )

    args = parser.parse_args()

    # Generate run metadata
    run_id = get_run_id()
    snapshot_date = datetime.now()

    logger.info("=" * 60)
    logger.info("Enhanced YouTube Data Collection Pipeline")
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

    # Initialize collector
    try:
        collector = YouTubeCollector(api_key)
        youtube = collector.youtube
    except Exception as e:
        logger.error(f"Failed to initialize YouTube collector: {e}")
        sys.exit(1)

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

    # Initialize registry if requested
    registry = None
    if args.update_registry:
        registry = VideoRegistry(args.registry)
        registry.load()
        logger.info(f"Registry loaded: {len(registry)} existing videos")

    # Phase 1: Channel Statistics
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 1: CHANNEL STATISTICS")
    logger.info("=" * 60)

    channel_stats_list = []
    for idx, row in channels_df.iterrows():
        channel_id = row["channel_id"]
        channel_name = row.get("channel_name", channel_id)
        logger.info(f"[{idx+1}/{len(channels_df)}] Getting stats for {channel_name}")
        
        # Use the collector's get_channel_info method
        stats = collector.get_channel_info(channel_id)
        if stats:
            # Add snapshot metadata
            stats["snapshot_date"] = snapshot_date.isoformat()
            channel_stats_list.append(stats)

    if channel_stats_list:
        channel_stats_df = pd.DataFrame(channel_stats_list)
        args.channel_stats_output.parent.mkdir(parents=True, exist_ok=True)
        channel_stats_df.to_csv(args.channel_stats_output, index=False)
        logger.info(f"Saved channel statistics to: {args.channel_stats_output}")

    # Phase 2: Video Metadata (using optimized batched collection)
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: VIDEO METADATA")
    logger.info("=" * 60)

    try:
        videos_df = collector.collect_from_channels(
            channels_df,
            max_videos_per_channel=args.max_videos,
            max_quota=args.max_quota,
            snapshot_date=snapshot_date,
            run_id=run_id,
        )
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        sys.exit(1)

    # Update registry if requested
    if registry and not videos_df.empty:
        for channel_id in videos_df["channel_id"].unique():
            channel_videos = videos_df[videos_df["channel_id"] == channel_id]["video_id"].tolist()
            registry.add_videos(channel_videos, channel_id, source="enhanced_collect")
        registry.save()
        logger.info(f"Registry updated: {len(registry)} total videos")

    # Save video metadata
    all_comments = []
    if not videos_df.empty:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        videos_df.to_csv(args.output, index=False)
        logger.info(f"Saved {len(videos_df)} videos to: {args.output}")

        # Phase 3: Comment samples (if requested)
        if args.collect_comments and collector.quota_used < args.max_quota - 500:
            logger.info("\n" + "=" * 60)
            logger.info("PHASE 3: COMMENT SAMPLES")
            logger.info("=" * 60)

            # Sample top videos by views
            top_videos = videos_df.nlargest(50, "view_count")

            for idx, (_, video_row) in enumerate(top_videos.iterrows()):
                if collector.quota_used >= args.max_quota - 100:
                    logger.warning(
                        "Approaching quota limit, stopping comment collection"
                    )
                    break

                video_id = video_row["video_id"]
                comments = get_video_comments_sample(youtube, video_id, max_comments=10)
                
                # Add snapshot metadata to comments
                for comment in comments:
                    comment["snapshot_date"] = snapshot_date.isoformat()
                
                all_comments.extend(comments)
                collector.quota_used += 1  # Approximate quota cost

                if (idx + 1) % 10 == 0:
                    logger.info(f"Collected comments from {idx+1}/50 top videos")

            if all_comments:
                comments_df = pd.DataFrame(all_comments)
                args.comments_output.parent.mkdir(parents=True, exist_ok=True)
                comments_df.to_csv(args.comments_output, index=False)
                logger.info(
                    f"Saved {len(comments_df)} comments to: {args.comments_output}"
                )

        # Final summary
        logger.info("\n" + "=" * 60)
        logger.info("COLLECTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Videos collected: {len(videos_df)}")
        logger.info(f"Channels analyzed: {len(channel_stats_list)}")
        if args.collect_comments:
            logger.info(f"Comments collected: {len(all_comments)}")
        logger.info(f"Total quota used: {collector.quota_used}")

    else:
        logger.warning("No data collected. Check logs for errors.")
        sys.exit(1)


if __name__ == "__main__":
    main()
