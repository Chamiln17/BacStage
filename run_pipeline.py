#!/usr/bin/env python
"""
Unified CLI for the YouTube Analytics Pipeline.

This is the main entry point for all pipeline operations:
- collect: Gather video metadata and channel statistics
- engineer: Build ML features from raw data
- analyze: Quick stats on collected data
- full-pipeline: Run everything in sequence

Usage:
    uv run python run_pipeline.py collect --channels data/raw/channels.csv
    uv run python run_pipeline.py engineer
    uv run python run_pipeline.py analyze
    uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv
"""

import argparse
import sys
from pathlib import Path


def cmd_collect(args: argparse.Namespace) -> int:
    """Run data collection."""
    from src.data.collect import main as collect_main
    
    # Build sys.argv for the collect module
    argv = ["collect", "--channels", str(args.channels)]
    
    if args.output:
        argv.extend(["--output", str(args.output)])
    if args.max_videos:
        argv.extend(["--max-videos", str(args.max_videos)])
    if args.max_quota:
        argv.extend(["--max-quota", str(args.max_quota)])
    if args.no_discover:
        argv.append("--no-discover")
    if args.no_channel_stats:
        argv.append("--no-channel-stats")
    if args.mode:
        argv.extend(["--mode", args.mode])
    if args.collect_comments:
        argv.append("--collect-comments")
    if args.no_backup:
        argv.append("--no-backup")
    
    sys.argv = argv
    try:
        collect_main()
        return 0
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 1


def cmd_engineer(args: argparse.Namespace) -> int:
    """Run feature engineering."""
    import logging
    import pandas as pd
    from src.features.engineer import VideoFeatureEngineer
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    input_path = args.input
    output_path = args.output
    
    logger.info("=" * 60)
    logger.info("Feature Engineering Pipeline")
    logger.info("=" * 60)
    
    # Load raw data
    try:
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        logger.info(f"Loading raw data from {input_path}")
        videos_df = pd.read_csv(input_path)
        logger.info(f"Loaded {len(videos_df)} videos with {len(videos_df.columns)} columns")
        
    except Exception as e:
        logger.error(f"Error loading input file: {e}")
        return 1
    
    # Engineer features
    try:
        engineer = VideoFeatureEngineer()
        videos_engineered = engineer.fit_transform(videos_df)
        videos_final = engineer.select_features(videos_engineered, include_target=True)
        logger.info(f"Feature engineering complete: {videos_final.shape}")
        
    except Exception as e:
        logger.error(f"Error during feature engineering: {e}")
        return 1
    
    # Save results
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        videos_final.to_csv(output_path, index=False)
        
        logger.info(f"Saved engineered features to: {output_path}")
        logger.info(f"Final dataset shape: {videos_final.shape}")
        logger.info(f"Features: {', '.join(videos_final.columns.tolist())}")
        
        # Summary
        logger.info("\nDataset Summary:")
        logger.info(f"  Date range: {videos_final['publish_date'].min()} to {videos_final['publish_date'].max()}")
        logger.info(f"  Channels: {videos_final['channel_id'].nunique()}")
        logger.info(f"  Subjects: {videos_final['subject'].value_counts().to_dict()}")
        logger.info(f"  Engagement: {videos_final['engagement_category'].value_counts().to_dict()}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error saving output: {e}")
        return 1


def cmd_analyze(args: argparse.Namespace) -> int:
    """Quick analysis of collected data."""
    import pandas as pd
    import sys
    
    # Fix Windows console encoding for Unicode
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    
    videos_path = args.videos
    channels_path = args.channels
    
    print("=" * 60)
    print("COLLECTION ANALYSIS")
    print("=" * 60)
    
    # Analyze videos
    if videos_path.exists():
        df = pd.read_csv(videos_path)
        print(f"\n[Videos] {videos_path}")
        print(f"   Total records: {len(df)}")
        
        if "video_id" in df.columns:
            print(f"   Unique videos: {df['video_id'].nunique()}")
        
        if "channel_title" in df.columns:
            print(f"\n   Videos per channel:")
            for channel, count in df["channel_title"].value_counts().items():
                print(f"     - {channel}: {count}")
        
        if "publish_date" in df.columns:
            df["publish_date"] = pd.to_datetime(df["publish_date"], format="ISO8601", utc=True)
            print(f"\n   Date range:")
            print(f"     Oldest: {df['publish_date'].min()}")
            print(f"     Newest: {df['publish_date'].max()}")
        
        if "snapshot_date" in df.columns:
            print(f"\n   Snapshots: {df['snapshot_date'].nunique()}")
        
        if "view_count" in df.columns:
            print(f"\n   View statistics:")
            print(f"     Total views: {df['view_count'].sum():,}")
            print(f"     Average views: {df['view_count'].mean():,.0f}")
            print(f"     Max views: {df['view_count'].max():,}")
    else:
        print(f"\n[!] Videos file not found: {videos_path}")
    
    # Analyze channel stats
    if channels_path.exists():
        ch_df = pd.read_csv(channels_path)
        print(f"\n[Channel Stats] {channels_path}")
        print(f"   Channels: {len(ch_df)}")
        
        if "subscriber_count" in ch_df.columns:
            print(f"   Total subscribers: {ch_df['subscriber_count'].sum():,}")
    else:
        print(f"\n[!] Channel stats file not found: {channels_path}")
    
    # Check registry
    registry_path = Path("data/raw/video_registry.csv")
    if registry_path.exists():
        reg_df = pd.read_csv(registry_path)
        print(f"\n[Registry] {registry_path}")
        print(f"   Known videos: {len(reg_df)}")
        if "channel_id" in reg_df.columns:
            print(f"   Channels tracked: {reg_df['channel_id'].nunique()}")
    
    # Check engineered features
    features_path = Path("data/processed/videos_engineered.csv")
    if features_path.exists():
        feat_df = pd.read_csv(features_path)
        print(f"\n[Engineered Features] {features_path}")
        print(f"   Records: {len(feat_df)}")
        print(f"   Features: {len(feat_df.columns)}")
    
    print("\n" + "=" * 60)
    return 0


def cmd_full_pipeline(args: argparse.Namespace) -> int:
    """Run the full pipeline: collect -> engineer."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("FULL PIPELINE EXECUTION")
    logger.info("=" * 60)
    
    # Step 1: Collect
    logger.info("\n>>> STEP 1: DATA COLLECTION <<<\n")
    
    collect_args = argparse.Namespace(
        channels=args.channels,
        output=Path("data/raw/videos_metadata.csv"),
        max_videos=args.max_videos,
        max_quota=args.max_quota,
        no_discover=False,
        no_channel_stats=False,
        mode=args.mode,
        collect_comments=args.collect_comments,
        no_backup=False,
    )
    
    result = cmd_collect(collect_args)
    if result != 0:
        logger.error("Collection failed. Aborting pipeline.")
        return result
    
    # Step 2: Engineer
    logger.info("\n>>> STEP 2: FEATURE ENGINEERING <<<\n")
    
    engineer_args = argparse.Namespace(
        input=Path("data/raw/videos_metadata.csv"),
        output=Path("data/processed/videos_engineered.csv"),
    )
    
    result = cmd_engineer(engineer_args)
    if result != 0:
        logger.error("Feature engineering failed.")
        return result
    
    # Step 3: Analyze
    logger.info("\n>>> STEP 3: ANALYSIS <<<\n")
    
    analyze_args = argparse.Namespace(
        videos=Path("data/raw/videos_metadata.csv"),
        channels=Path("data/raw/channel_statistics.csv"),
    )
    
    cmd_analyze(analyze_args)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Analytics Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect        Collect video metadata and channel statistics
  engineer       Build ML features from raw data  
  analyze        Quick stats on collected data
  full-pipeline  Run everything (collect -> engineer -> analyze)

Examples:
  # Full pipeline
  uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv

  # Just collect data
  uv run python run_pipeline.py collect --channels data/raw/channels.csv

  # Just build features
  uv run python run_pipeline.py engineer

  # Quick analysis
  uv run python run_pipeline.py analyze
        """,
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # ===== COLLECT =====
    collect_parser = subparsers.add_parser(
        "collect", help="Collect video metadata and channel statistics"
    )
    collect_parser.add_argument(
        "--channels", type=Path, required=True,
        help="Path to channels CSV file"
    )
    collect_parser.add_argument(
        "--output", type=Path, default=None,
        help="Output path for video metadata"
    )
    collect_parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max videos per channel"
    )
    collect_parser.add_argument(
        "--max-quota", type=int, default=8000,
        help="Maximum API quota to use"
    )
    collect_parser.add_argument(
        "--no-discover", action="store_true",
        help="Skip discovery (only enrich known videos)"
    )
    collect_parser.add_argument(
        "--no-channel-stats", action="store_true",
        help="Skip channel statistics"
    )
    collect_parser.add_argument(
        "--mode", choices=["append", "dedupe"], default="append",
        help="Merge mode for snapshots"
    )
    collect_parser.add_argument(
        "--collect-comments", action="store_true",
        help="Also collect comment samples"
    )
    collect_parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip backup creation"
    )
    collect_parser.set_defaults(func=cmd_collect)
    
    # ===== ENGINEER =====
    engineer_parser = subparsers.add_parser(
        "engineer", help="Build ML features from raw data"
    )
    engineer_parser.add_argument(
        "--input", type=Path, default=Path("data/raw/videos_metadata.csv"),
        help="Input video metadata CSV"
    )
    engineer_parser.add_argument(
        "--output", type=Path, default=Path("data/processed/videos_engineered.csv"),
        help="Output engineered features CSV"
    )
    engineer_parser.set_defaults(func=cmd_engineer)
    
    # ===== ANALYZE =====
    analyze_parser = subparsers.add_parser(
        "analyze", help="Quick stats on collected data"
    )
    analyze_parser.add_argument(
        "--videos", type=Path, default=Path("data/raw/videos_metadata.csv"),
        help="Path to videos metadata CSV"
    )
    analyze_parser.add_argument(
        "--channels", type=Path, default=Path("data/raw/channel_statistics.csv"),
        help="Path to channel statistics CSV"
    )
    analyze_parser.set_defaults(func=cmd_analyze)
    
    # ===== FULL-PIPELINE =====
    full_parser = subparsers.add_parser(
        "full-pipeline", help="Run collect -> engineer -> analyze"
    )
    full_parser.add_argument(
        "--channels", type=Path, required=True,
        help="Path to channels CSV file"
    )
    full_parser.add_argument(
        "--max-videos", type=int, default=None,
        help="Max videos per channel"
    )
    full_parser.add_argument(
        "--max-quota", type=int, default=8000,
        help="Maximum API quota to use"
    )
    full_parser.add_argument(
        "--mode", choices=["append", "dedupe"], default="append",
        help="Merge mode for snapshots"
    )
    full_parser.add_argument(
        "--collect-comments", action="store_true",
        help="Also collect comment samples"
    )
    full_parser.set_defaults(func=cmd_full_pipeline)
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
