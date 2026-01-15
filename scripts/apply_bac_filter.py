#!/usr/bin/env python
"""
Apply Bac 3AS filter to video metadata.

This script filters the collected YouTube videos to identify Bac (3AS) content
using keyword-based classification. It generates:
1. A full dataset with filter columns added
2. A Bac-only subset for ML modeling

Usage:
    uv run python scripts/apply_bac_filter.py
    uv run python scripts/apply_bac_filter.py --input data/raw/videos_metadata.csv
    uv run python scripts/apply_bac_filter.py --dedupe  # Deduplicate by video_id first
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.features.bac_filter import filter_videos_dataframe, get_filter_statistics

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Fix console encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def main() -> None:
    """Main entry point for Bac filtering."""
    parser = argparse.ArgumentParser(
        description="Apply Bac 3AS filter to video metadata",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "videos_metadata.csv",
        help="Path to input video metadata CSV",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
        help="Directory for output files",
    )

    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Deduplicate by video_id (keep latest snapshot) before filtering",
    )

    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable progress bar",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bac 3AS Video Filter")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    logger.info("\nLoading video metadata...")
    try:
        df = pd.read_csv(args.input)
        logger.info(f"Loaded {len(df):,} records")
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)

    # Optional deduplication
    if args.dedupe:
        logger.info("\nDeduplicating by video_id (keeping latest snapshot)...")
        original_count = len(df)

        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="ISO8601")
            df = df.sort_values("snapshot_date", ascending=False)

        df = df.drop_duplicates(subset=["video_id"], keep="first")
        removed = original_count - len(df)
        logger.info(f"Removed {removed:,} duplicate snapshots, {len(df):,} unique videos remain")

    # Show data info
    logger.info(f"\nData columns: {list(df.columns)}")
    if "channel_id" in df.columns:
        logger.info(f"Unique channels: {df['channel_id'].nunique()}")
    if "video_id" in df.columns:
        logger.info(f"Unique videos: {df['video_id'].nunique()}")

    # Apply filter
    logger.info("\nApplying Bac 3AS filter...")
    df_filtered = filter_videos_dataframe(
        df,
        title_col="title",
        description_col="description",
        tags_col="tags",
        show_progress=not args.no_progress,
    )

    # Get statistics
    stats = get_filter_statistics(df_filtered)

    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING RESULTS")
    logger.info("=" * 60)
    logger.info(
        f"Bac 3AS videos: {stats['bac_3as_count']:,} / {stats['total_videos']:,} "
        f"({stats['bac_percentage']:.1f}%)"
    )
    logger.info(f"Non-Bac videos: {stats['non_bac_count']:,}")

    if "subject_distribution" in stats:
        logger.info("\nBy Subject:")
        for subject, count in sorted(
            stats["subject_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            logger.info(f"  {subject:25s} {count:,}")

    if "confidence_stats" in stats:
        conf = stats["confidence_stats"]
        logger.info("\nConfidence Distribution (Bac videos):")
        logger.info(f"  Mean:  {conf['mean']:.3f}")
        logger.info(f"  Std:   {conf['std']:.3f}")
        logger.info(f"  Min:   {conf['min']:.3f}")
        logger.info(f"  Max:   {conf['max']:.3f}")

    # Save outputs
    logger.info("\n" + "=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)

    # Full dataset with filter columns
    full_output = args.output_dir / "videos_with_bac_filter.csv"
    df_filtered.to_csv(full_output, index=False)
    logger.info(f"Full dataset: {full_output}")
    logger.info(f"  {len(df_filtered):,} records, {len(df_filtered.columns)} columns")

    # Bac-only subset
    df_bac_only = df_filtered[df_filtered["is_bac_3as"]].copy()
    bac_output = args.output_dir / "videos_bac_3as_only.csv"
    df_bac_only.to_csv(bac_output, index=False)
    logger.info(f"Bac-only dataset: {bac_output}")
    logger.info(f"  {len(df_bac_only):,} videos")

    # Non-Bac subset (for reference/validation)
    df_non_bac = df_filtered[~df_filtered["is_bac_3as"]].copy()
    non_bac_output = args.output_dir / "videos_non_bac.csv"
    df_non_bac.to_csv(non_bac_output, index=False)
    logger.info(f"Non-Bac dataset: {non_bac_output}")
    logger.info(f"  {len(df_non_bac):,} videos")

    # Summary by filter reason (for debugging)
    logger.info("\nFilter Reason Distribution:")
    reason_dist = df_filtered["filter_reason"].value_counts().head(10)
    for reason, count in reason_dist.items():
        logger.info(f"  {reason[:50]:50s} {count:,}")

    logger.info("\n" + "=" * 60)
    logger.info("FILTERING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Ready for ML modeling with {len(df_bac_only):,} Bac 3AS videos")


if __name__ == "__main__":
    main()
