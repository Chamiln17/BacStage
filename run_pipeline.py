#!/usr/bin/env python
"""
Unified CLI for the YouTube Analytics Pipeline.

This is the main entry point for all pipeline operations:
- collect: Gather video metadata and channel statistics
- filter_data: Apply data-driven Bac 3AS filter (discovery + filtering)
- engineer: Build ML features from raw data
- analyze: Quick stats on collected data
- full-pipeline: Run everything in sequence

Usage:
    uv run python run_pipeline.py collect --channels data/raw/channels.csv
    uv run python run_pipeline.py filter_data
    uv run python run_pipeline.py filter_data --skip-discovery --validate
    uv run python run_pipeline.py engineer
    uv run python run_pipeline.py analyze
    uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv --filter
"""

import argparse
import sys
from pathlib import Path
from typing import Optional


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


def _load_videos_with_transcripts(videos_path: Path, transcripts_path: Optional[Path]):
    """Load videos and attach `transcript_text` by video_id when a transcripts CSV exists."""
    import logging
    import pandas as pd

    logger = logging.getLogger(__name__)
    if not videos_path.exists():
        raise FileNotFoundError(f"Input file not found: {videos_path}")
    videos = pd.read_csv(videos_path)
    logger.info(f"Loaded {len(videos):,} videos from {videos_path}")
    if transcripts_path is not None and transcripts_path.exists() and "transcript_text" not in videos:
        transcripts = pd.read_csv(transcripts_path, usecols=["video_id", "transcript_text"])
        videos = videos.merge(transcripts.drop_duplicates("video_id"), on="video_id", how="left")
        logger.info(f"Attached {videos['transcript_text'].notna().sum():,} transcripts from {transcripts_path}")
    elif transcripts_path is not None and not transcripts_path.exists():
        logger.warning(f"Transcripts file not found: {transcripts_path}; transcript features will be 0")
    return videos


def cmd_engineer(args: argparse.Namespace) -> int:
    """Export per-video engineered features (for notebooks and analysis)."""
    import logging
    from src.features.engineer import engineered_export

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    try:
        videos = _load_videos_with_transcripts(args.input, args.transcripts)
        engineered = engineered_export(videos)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        engineered.to_csv(args.output, index=False)
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return 1
    logger.info(f"Saved {engineered.shape[1]} columns for {len(engineered):,} videos to {args.output}")
    logger.info(f"Engagement: {engineered['engagement_category'].value_counts().to_dict()}")
    return 0


def cmd_transcripts(args: argparse.Namespace) -> int:
    """Collect YouTube transcripts."""
    from src.data.transcript_collector import collect_transcripts_cli
    return collect_transcripts_cli(args.input, args.output, args.delay)


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


def cmd_filter_data(args: argparse.Namespace) -> int:
    """Apply the data-driven Bac 3AS filter; file I/O around `run_bac_filter`."""
    import json
    import logging
    from datetime import datetime
    import pandas as pd

    from src.features.bac_filter_balanced import (
        get_filter_statistics,
        latest_snapshots,
        load_channel_subjects,
        load_filter_config,
        run_bac_filter,
        score_against_labels,
        validation_sample,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    output_dir = args.output_dir
    priors_path = output_dir / "channel_priors.csv"
    terms_path = output_dir / "tfidf_bac_terms.json"
    try:
        config = load_filter_config(args.config)
        if not args.input.exists():
            raise FileNotFoundError(f"Input file not found: {args.input}")
        videos = pd.read_csv(args.input)
        if args.dedupe:
            videos = latest_snapshots(videos)
        logger.info(f"Loaded {len(videos):,} videos from {args.input}")

        cached = not args.force_discovery and priors_path.exists() and terms_path.exists()
        if args.skip_discovery and not cached:
            logger.warning("--skip-discovery given but cached artifacts are missing; discovering")
        use_cache = cached
        priors = pd.read_csv(priors_path) if use_cache else None
        terms = json.loads(terms_path.read_text(encoding="utf-8")) if use_cache else None
        logger.info("Using cached channel priors and terms" if use_cache else "Discovering channel priors and terms")

        result = run_bac_filter(videos, config, load_channel_subjects(args.channels),
                                priors=priors, terms=terms, show_progress=not args.no_progress)
    except (FileNotFoundError, ValueError) as e:
        logger.error(str(e))
        return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    if not use_cache:
        result.priors.to_csv(priors_path, index=False)
        terms_path.write_text(json.dumps(result.terms, ensure_ascii=False, indent=2), encoding="utf-8")
    filtered = result.videos
    filtered.to_csv(output_dir / "videos_bac_balanced.csv", index=False)
    filtered[filtered["is_bac_3as"]].to_csv(output_dir / "videos_bac_only.csv", index=False)
    filtered[~filtered["is_bac_3as"]].to_csv(output_dir / "videos_rejected.csv", index=False)

    stats = get_filter_statistics(filtered)
    logger.info(f"Bac-heavy channels: {int(result.priors['is_bac_heavy'].sum())}/{len(result.priors)}, "
                f"TF-IDF terms: {len(result.terms)}")
    logger.info(f"Bac 3AS videos: {stats['bac_3as_count']:,} / {stats['total_videos']:,} "
                f"({stats['bac_percentage']:.1f}%)")
    logger.info(f"By category: {stats.get('category_distribution', {})}")
    logger.info(f"Outputs written to {output_dir}")

    validation_dir = Path("data/validation")
    labels_path = validation_dir / "sample_for_manual_review.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path, usecols=["video_id", "manual_is_bac"])
        scored = labels.merge(filtered[["video_id", "is_bac_3as"]], on="video_id")
        score = score_against_labels(scored)
        logger.info(f"Against {score['labelled']} hand labels: precision={score['precision']:.3f} "
                    f"recall={score['recall']:.3f} accuracy={score['accuracy']:.3f}")

    if args.validate:
        sample = validation_sample(filtered, config)
        validation_dir.mkdir(parents=True, exist_ok=True)
        # Never overwrite a file that may hold hand labels.
        target = labels_path if not labels_path.exists() else (
            validation_dir / f"sample_for_manual_review_{datetime.now():%Y%m%d_%H%M%S}.csv")
        sample.to_csv(target, index=False)
        logger.info(f"Validation sample of {len(sample)} videos written to {target}")
    return 0


def cmd_clean(args: argparse.Namespace) -> int:
    """Run data cleaning pipeline."""
    from src.data.clean_data import clean_pipeline
    
    return clean_pipeline(
        videos_path=args.videos,
        transcripts_path=args.transcripts,
        output_dir=args.output_dir,
    )

def cmd_full_pipeline(args: argparse.Namespace) -> int:
    """Run the full pipeline: collect -> [filter] -> engineer -> analyze."""
    import logging
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 60)
    logger.info("FULL PIPELINE EXECUTION")
    logger.info("=" * 60)
    
    step_num = 1
    
    # Step 1: Collect
    logger.info(f"\n>>> STEP {step_num}: DATA COLLECTION <<<\n")
    step_num += 1
    
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
    
    # Step 2: Filter (optional)
    if getattr(args, "filter", False):
        logger.info(f"\n>>> STEP {step_num}: DATA-DRIVEN FILTERING <<<\n")
        step_num += 1
        
        filter_args = argparse.Namespace(
            input=Path("data/raw/videos_metadata.csv"),
            channels=args.channels,
            output_dir=Path("data/processed"),
            config=Path("config/filter_config.yaml"),
            skip_discovery=False,
            force_discovery=False,
            validate=False,
            dedupe=True,
            no_progress=False,
        )
        
        result = cmd_filter_data(filter_args)
        if result != 0:
            logger.error("Filtering failed. Continuing with unfiltered data.")
    
    # Step 3: Engineer
    logger.info(f"\n>>> STEP {step_num}: FEATURE ENGINEERING <<<\n")
    step_num += 1
    
    # Use filtered data if filtering was done
    if getattr(args, "filter", False):
        input_for_engineer = Path("data/cleaned/videos_cleaned.csv")
    else:
        input_for_engineer = Path("data/cleaned/videos_cleaned.csv") # Default to cleaned
    
    engineer_args = argparse.Namespace(
        input=input_for_engineer,
        output=Path("data/processed/videos_engineered.csv"),
        transcripts=Path("data/cleaned/transcripts_clean.csv"),
    )
    
    result = cmd_engineer(engineer_args)
    if result != 0:
        logger.error("Feature engineering failed.")
        return result
    
    # Step 4: Analyze
    logger.info(f"\n>>> STEP {step_num}: ANALYSIS <<<\n")
    
    analyze_args = argparse.Namespace(
        videos=Path("data/raw/videos_metadata.csv"),
        channels=Path("data/raw/channel_statistics.csv"),
    )
    
    cmd_analyze(analyze_args)
    
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    
    return 0



def cmd_train(args: argparse.Namespace) -> int:
    """Train the engagement model and save models/model.joblib."""
    import logging
    import pandas as pd
    from src.models.train_model import train

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    try:
        videos = _load_videos_with_transcripts(args.input, args.transcripts)
        test_ids = None
        if args.split_file is not None:
            splits = pd.read_csv(args.split_file)
            test_ids = splits.loc[splits["split"] == "test", "video_id"]
        embedder, embedding_model = None, None
        if not args.no_arabert:
            from src.features.text_embeddings import TextEmbeddingExtractor
            embedding_model = "aubmindlab/bert-base-arabertv2"
            embedder = TextEmbeddingExtractor(embedding_model)
        metadata = train(
            videos,
            args.output_dir,
            model_type=args.model_type,
            embedder=embedder,
            embedding_model=embedding_model,
            test_ids=test_ids,
            random_seed=args.random_seed,
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
    m = metadata["test_metrics"]
    logger.info(f"Test R2={m['r2']:.4f}  MAE={m['mae']:.4f}  RMSE={m['rmse']:.4f}  ({metadata['split']})")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    """Score planned videos from a JSON object, a JSON list, or a CSV."""
    import json
    import logging
    import pandas as pd
    from src.models.predict_model import EngagementPredictor

    logger = logging.getLogger(__name__)
    try:
        predictor = EngagementPredictor(model_dir=args.model_dir)
        if str(args.input).endswith(".json"):
            data = json.loads(Path(args.input).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                output = predictor.predict(data)
            else:
                output = predictor.predict_batch(pd.DataFrame(data)).to_dict(orient="records")
        else:
            output = predictor.predict_batch(pd.read_csv(args.input)).to_dict(orient="records")

        if args.output == "stdout":
            print(json.dumps(output, indent=2, ensure_ascii=False, default=str))
        elif args.format == "csv" or args.output.endswith(".csv"):
            rows = output if isinstance(output, list) else [output]
            pd.DataFrame(rows).to_csv(args.output, index=False)
        else:
            Path(args.output).write_text(
                json.dumps(output, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
            )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return 1
    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="YouTube Analytics Pipeline CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect        Collect video metadata and channel statistics
  filter_data    Apply data-driven Bac 3AS filter
  engineer       Build ML features from raw data  
  analyze        Quick stats on collected data
  full-pipeline  Run everything (collect -> [filter] -> engineer -> analyze)

Examples:
  # Full pipeline with filtering
  uv run python run_pipeline.py full-pipeline --channels data/raw/channels.csv --filter

  # Just collect data
  uv run python run_pipeline.py collect --channels data/raw/channels.csv

  # Apply data-driven filter (discovery + filter)
  uv run python run_pipeline.py filter_data

  # Filter with validation samples
  uv run python run_pipeline.py filter_data --validate

  # Skip discovery (use cached artifacts)
  uv run python run_pipeline.py filter_data --skip-discovery

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
        "--input", type=Path, default=Path("data/cleaned/videos_cleaned.csv"),
        help="Input video metadata CSV"
    )
    engineer_parser.add_argument(
        "--output", type=Path, default=Path("data/processed/videos_engineered.csv"),
        help="Output engineered features CSV"
    )
    engineer_parser.add_argument(
        "--transcripts", type=Path, default=Path("data/cleaned/transcripts_clean.csv"),
        help="Transcripts CSV joined by video_id (default: data/cleaned/transcripts_clean.csv)"
    )
    engineer_parser.set_defaults(func=cmd_engineer)
    
    # ===== TRANSCRIPTS =====
    transcripts_parser = subparsers.add_parser(
        "transcripts", help="Collect YouTube transcripts"
    )
    transcripts_parser.add_argument(
        "--input", type=Path, required=True,
        help="Input CSV with video IDs"
    )
    transcripts_parser.add_argument(
        "--output", type=Path, required=True,
        help="Output CSV for transcripts"
    )
    transcripts_parser.add_argument(
        "--delay", type=float, default=2.0,
        help="Seconds to wait between videos (default: 2)"
    )
    transcripts_parser.set_defaults(func=cmd_transcripts)
    
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
    
    # ===== FILTER_DATA =====
    filter_parser = subparsers.add_parser(
        "filter_data", help="Apply data-driven Bac 3AS filter"
    )
    filter_parser.add_argument(
        "--input", type=Path, default=Path("data/raw/videos_metadata.csv"),
        help="Input video metadata CSV"
    )
    filter_parser.add_argument(
        "--channels", type=Path, default=Path("data/raw/channels.csv"),
        help="Channels CSV with subject mapping"
    )
    filter_parser.add_argument(
        "--output-dir", type=Path, default=Path("data/processed"),
        help="Output directory for filtered data"
    )
    filter_parser.add_argument(
        "--config", type=Path, default=Path("config/filter_config.yaml"),
        help="Filter configuration YAML"
    )
    filter_parser.add_argument(
        "--skip-discovery", action="store_true",
        help="Skip Phase 1 discovery if artifacts exist"
    )
    filter_parser.add_argument(
        "--force-discovery", action="store_true",
        help="Force re-run Phase 1 even if artifacts exist"
    )
    filter_parser.add_argument(
        "--validate", action="store_true",
        help="Generate validation samples after filtering"
    )
    filter_parser.add_argument(
        "--dedupe", action="store_true", default=True,
        help="Deduplicate by video_id before filtering (default: True)"
    )
    filter_parser.add_argument(
        "--no-progress", action="store_true",
        help="Disable progress bar"
    )
    filter_parser.set_defaults(func=cmd_filter_data)
    
    # ===== CLEAN =====
    clean_parser = subparsers.add_parser(
        "clean", help="Clean and prepare data for feature engineering"
    )
    clean_parser.add_argument(
        "--videos", type=Path, default=Path("data/processed/videos_bac_only.csv"),
        help="Input video metadata CSV"
    )
    clean_parser.add_argument(
        "--transcripts", type=Path, default=Path("data/processed/transcripts_merged.csv"),
        help="Input transcripts CSV"
    )
    clean_parser.add_argument(
        "--output-dir", type=Path, default=Path("data/cleaned"),
        help="Output directory for cleaned data"
    )
    clean_parser.set_defaults(func=cmd_clean)
    
    # ===== FULL-PIPELINE =====
    full_parser = subparsers.add_parser(
        "full-pipeline", help="Run collect -> filter -> engineer -> analyze"
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
    full_parser.add_argument(
        "--filter", action="store_true",
        help="Include Bac 3AS filtering step"
    )
    full_parser.set_defaults(func=cmd_full_pipeline)
    
    # ===== TRAIN =====
    train_parser = subparsers.add_parser(
        "train", help="Train engagement prediction model"
    )
    train_parser.add_argument(
        "--input", type=Path, default=Path("data/cleaned/videos_cleaned.csv"),
        help="Cleaned videos CSV (default: data/cleaned/videos_cleaned.csv)"
    )
    train_parser.add_argument(
        "--transcripts", type=Path, default=Path("data/cleaned/transcripts_clean.csv"),
        help="Transcripts CSV joined by video_id (default: data/cleaned/transcripts_clean.csv)"
    )
    train_parser.add_argument(
        "--output-dir", type=Path, default=Path("models"),
        help="Model output directory (default: models/)"
    )
    train_parser.add_argument(
        "--model-type", type=str, default="rf",
        choices=["catboost", "xgboost", "lightgbm", "rf"],
        help="Model type (default: rf, the best model from the notebook comparison)"
    )
    train_parser.add_argument(
        "--split-file", type=Path, default=None,
        help="CSV with video_id,split; rows with split=test are held out (default: random 20%%)"
    )
    train_parser.add_argument(
        "--no-arabert", action="store_true",
        help="Skip AraBERT embeddings (faster training)"
    )
    train_parser.add_argument(
        "--random-seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    train_parser.set_defaults(func=cmd_train)

    # ===== PREDICT =====
    predict_parser = subparsers.add_parser(
        "predict", help="Predict engagement for videos"
    )
    predict_parser.add_argument(
        "--input", type=Path, required=True,
        help="Input video data (CSV or JSON)"
    )
    predict_parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Model directory (default: models/)"
    )
    predict_parser.add_argument(
        "--output", type=str, default="stdout",
        help="Output path or 'stdout' (default: stdout)"
    )
    predict_parser.add_argument(
        "--format", type=str, default="json",
        choices=["json", "csv"],
        help="Output format (default: json)"
    )
    predict_parser.set_defaults(func=cmd_predict)
    
    # Parse and execute

    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
