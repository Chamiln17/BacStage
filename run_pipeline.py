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


def cmd_filter_data(args: argparse.Namespace) -> int:
    """Apply data-driven Bac 3AS filter with automatic discovery."""
    import json
    import logging
    import yaml
    import pandas as pd
    
    from src.features.bac_filter_balanced import (
        BalancedBacFilter,
        filter_videos_dataframe,
        get_filter_statistics,
        load_channel_priors,
        load_tfidf_terms,
        load_channel_subjects,
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    
    # Fix console encoding for Windows
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    
    # Resolve paths
    input_path = args.input
    channels_path = args.channels
    output_dir = args.output_dir
    config_path = args.config
    
    priors_path = output_dir / "channel_priors.csv"
    tfidf_path = output_dir / "tfidf_bac_terms.json"
    
    logger.info("=" * 60)
    logger.info("DATA-DRIVEN BAC 3AS FILTER")
    logger.info("=" * 60)
    
    # Load config
    config = {}
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        logger.warning(f"Config not found: {config_path}, using defaults")
    
    # Get config values with defaults
    channel_prior_config = config.get("channel_prior", {})
    soft_positives_config = config.get("soft_positives", {})
    markers_config = config.get("markers", {})
    validation_config = config.get("validation", {})
    
    bac_threshold = channel_prior_config.get("bac_threshold", 0.7)
    non_bac_max = channel_prior_config.get("non_bac_max", 0.1)
    duration_min = soft_positives_config.get("duration_min", 300)
    require_tfidf = soft_positives_config.get("require_tfidf", False)
    require_duration = soft_positives_config.get("require_duration", False)
    allow_channel_subject = soft_positives_config.get("allow_channel_subject", True)
    
    bac_markers = markers_config.get("bac", [
        "bac", "3as", "بكالوريا", "باك", "ثالثة ثانوي",
        "السنة الثالثة ثانوي", "terminale"
    ])
    non_bac_markers = markers_config.get("non_bac", [
        "1as", "2as", "سنة أولى ثانوي", "سنة ثانية ثانوي",
        "أولى ثانوي", "ثانية ثانوي", "متوسط", "bem",
        "1am", "2am", "3am", "4am", "ابتدائي"
    ])
    strong_bac_intent = markers_config.get("strong_bac_intent", [
        "مراجعة بكالوريا", "تحضير بكالوريا", "تصحيح بكالوريا",
        "موضوع بكالوريا", "حل موضوع بكالوريا",
        "bac blanc", "révision bac", "corrigé bac", "sujet bac"
    ])
    
    # =========================================================================
    # PHASE 1: Data Discovery
    # =========================================================================
    
    need_discovery = args.force_discovery or (
        not args.skip_discovery and (
            not priors_path.exists() or not tfidf_path.exists()
        )
    )
    
    if need_discovery:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 1: DATA DISCOVERY")
        logger.info("=" * 60)
        
        # Validate input
        if not input_path.exists():
            logger.error(f"Input file not found: {input_path}")
            return 1
        
        # Load and dedupe data for discovery
        logger.info(f"\nLoading video metadata from {input_path}...")
        df_raw = pd.read_csv(input_path)
        logger.info(f"Loaded {len(df_raw):,} records")
        
        if "snapshot_date" in df_raw.columns:
            df_raw["snapshot_date"] = pd.to_datetime(
                df_raw["snapshot_date"], format="ISO8601", errors="coerce"
            )
            df_raw = df_raw.sort_values("snapshot_date", ascending=False)
        
        if "video_id" in df_raw.columns:
            original = len(df_raw)
            df_raw = df_raw.drop_duplicates(subset=["video_id"], keep="first")
            logger.info(f"Deduplicated: {original:,} → {len(df_raw):,} unique videos")
        
        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # --- Build channel priors ---
        logger.info("\n>>> Building channel priors...")
        
        import re
        
        def has_markers(text_series: pd.Series, markers: list) -> pd.Series:
            pattern = "|".join([re.escape(m) for m in markers])
            return text_series.str.contains(pattern, regex=True, case=False, na=False)
        
        text = (
            df_raw["title"].fillna("").astype(str) + " "
            + df_raw["description"].fillna("").astype(str) + " "
            + df_raw["tags"].fillna("").astype(str)
        ).str.lower()
        
        df_raw["has_bac"] = has_markers(text, bac_markers)
        df_raw["has_non_bac"] = has_markers(text, non_bac_markers)
        df_raw["has_any_grade"] = df_raw["has_bac"] | df_raw["has_non_bac"]
        
        channel_stats = df_raw.groupby("channel_id").agg({
            "has_bac": "sum",
            "has_non_bac": "sum",
            "has_any_grade": "sum",
            "video_id": "count",
        }).rename(columns={
            "has_bac": "count_bac_marked",
            "has_non_bac": "count_non_bac_marked",
            "has_any_grade": "count_any_grade_marked",
            "video_id": "total_videos",
        })
        
        channel_stats["p_bac"] = (
            channel_stats["count_bac_marked"] / 
            channel_stats["count_any_grade_marked"].replace(0, 1)
        )
        channel_stats["p_non"] = (
            channel_stats["count_non_bac_marked"] / 
            channel_stats["count_any_grade_marked"].replace(0, 1)
        )
        channel_stats["is_bac_heavy"] = (
            (channel_stats["p_bac"] >= bac_threshold) & 
            (channel_stats["p_non"] <= non_bac_max)
        )
        
        channel_stats = channel_stats.reset_index().sort_values("p_bac", ascending=False)
        channel_stats.to_csv(priors_path, index=False)
        
        bac_heavy_count = channel_stats["is_bac_heavy"].sum()
        logger.info(f"Channel priors saved: {priors_path}")
        logger.info(f"Bac-heavy channels: {bac_heavy_count}/{len(channel_stats)}")
        
        # --- TF-IDF discovery ---
        logger.info("\n>>> Running TF-IDF keyword discovery...")
        
        from sklearn.feature_extraction.text import TfidfVectorizer
        import numpy as np
        
        # Create pseudo-labels
        bac_mask = df_raw["has_bac"] & ~df_raw["has_non_bac"]
        non_bac_mask = df_raw["has_non_bac"] & ~df_raw["has_bac"]
        
        bac_titles = df_raw.loc[bac_mask, "title"].fillna("").astype(str).tolist()
        non_bac_titles = df_raw.loc[non_bac_mask, "title"].fillna("").astype(str).tolist()
        
        logger.info(f"Pseudo-labeled: {len(bac_titles):,} Bac, {len(non_bac_titles):,} non-Bac")
        
        if len(bac_titles) >= 50 and len(non_bac_titles) >= 20:
            tfidf_config = config.get("tfidf", {})
            top_n = tfidf_config.get("top_n_terms", 100)
            min_df = tfidf_config.get("min_df", 5)
            max_df = tfidf_config.get("max_df", 0.8)
            ngram_range = tuple(tfidf_config.get("ngram_range", [1, 2]))
            analyzer = tfidf_config.get("analyzer", "char_wb")
            
            all_titles = bac_titles + non_bac_titles
            
            vectorizer = TfidfVectorizer(
                ngram_range=ngram_range,
                analyzer=analyzer,
                min_df=min_df,
                max_df=max_df,
            )
            
            try:
                X = vectorizer.fit_transform(all_titles)
                feature_names = vectorizer.get_feature_names_out()
                
                n_bac = len(bac_titles)
                bac_tfidf = X[:n_bac].mean(axis=0).A1
                non_bac_tfidf = X[n_bac:].mean(axis=0).A1
                diff = bac_tfidf - non_bac_tfidf
                
                top_idx = np.argsort(diff)[::-1][:top_n]
                top_terms = [feature_names[i] for i in top_idx if diff[i] > 0]
                
                with open(tfidf_path, "w", encoding="utf-8") as f:
                    json.dump(top_terms, f, ensure_ascii=False, indent=2)
                
                logger.info(f"TF-IDF terms saved: {tfidf_path} ({len(top_terms)} terms)")
                
            except Exception as e:
                logger.warning(f"TF-IDF discovery failed: {e}")
                top_terms = []
                with open(tfidf_path, "w", encoding="utf-8") as f:
                    json.dump(top_terms, f)
        else:
            logger.warning("Not enough pseudo-labeled data for TF-IDF, skipping")
            with open(tfidf_path, "w", encoding="utf-8") as f:
                json.dump([], f)
    else:
        logger.info("\nSkipping discovery (artifacts exist or --skip-discovery)")
    
    # =========================================================================
    # PHASE 2: Apply Balanced Filter
    # =========================================================================
    
    logger.info("\n" + "=" * 60)
    logger.info("PHASE 2: APPLY BALANCED FILTER")
    logger.info("=" * 60)
    
    # Validate inputs
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    # Load data-driven components
    logger.info("\nLoading data-driven components...")
    
    bac_heavy_channels = load_channel_priors(priors_path)
    logger.info(f"  Channel priors: {len(bac_heavy_channels)} Bac-heavy channels")
    
    tfidf_terms = load_tfidf_terms(tfidf_path)
    logger.info(f"  TF-IDF terms: {len(tfidf_terms)} terms")
    
    channel_subjects = load_channel_subjects(channels_path)
    logger.info(f"  Channel subjects: {len(channel_subjects)} channels")
    
    # Load video data
    logger.info(f"\nLoading video metadata from {input_path}...")
    df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(df):,} records")
    
    # Deduplicate if requested
    if args.dedupe:
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(
                df["snapshot_date"], format="ISO8601", errors="coerce"
            )
            df = df.sort_values("snapshot_date", ascending=False)
        
        if "video_id" in df.columns:
            original = len(df)
            df = df.drop_duplicates(subset=["video_id"], keep="first")
            logger.info(f"Deduplicated: {original:,} → {len(df):,} unique videos")
    
    # Create filter
    bac_filter = BalancedBacFilter(
        bac_markers=bac_markers,
        non_bac_markers=non_bac_markers,
        strong_bac_intent=strong_bac_intent,
        bac_heavy_channels=bac_heavy_channels,
        tfidf_terms=tfidf_terms,
        channel_subjects=channel_subjects,
        duration_min=duration_min,
        require_tfidf=require_tfidf,
        require_duration=require_duration,
        allow_channel_subject=allow_channel_subject,
    )
    
    # Apply filter
    logger.info("\nApplying balanced Bac filter...")
    df_filtered = filter_videos_dataframe(
        df,
        bac_filter,
        show_progress=not args.no_progress,
    )
    
    # Get statistics
    stats = get_filter_statistics(df_filtered)
    
    # Display results
    logger.info("\n" + "=" * 60)
    logger.info("FILTERING RESULTS")
    logger.info("=" * 60)
    
    logger.info(
        f"\nBac 3AS videos: {stats['bac_3as_count']:,} / {stats['total_videos']:,} "
        f"({stats['bac_percentage']:.1f}%)"
    )
    
    if "category_distribution" in stats:
        logger.info("\nBy Category:")
        for cat, count in sorted(stats["category_distribution"].items(), key=lambda x: -x[1]):
            logger.info(f"  {cat:25s} {count:,}")
    
    if "subject_distribution" in stats:
        logger.info("\nBy Subject (Bac videos):")
        for subject, count in sorted(stats["subject_distribution"].items(), key=lambda x: -x[1]):
            logger.info(f"  {subject:25s} {count:,}")
    
    if "confidence_stats" in stats:
        conf = stats["confidence_stats"]
        logger.info("\nConfidence Distribution (Bac videos):")
        logger.info(f"  Mean:  {conf['mean']:.3f}")
        logger.info(f"  Std:   {conf['std']:.3f}")
    
    # Save outputs
    logger.info("\n" + "=" * 60)
    logger.info("SAVING OUTPUTS")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Full dataset
    balanced_path = output_dir / "videos_bac_balanced.csv"
    df_filtered.to_csv(balanced_path, index=False)
    logger.info(f"\nFull dataset: {balanced_path}")
    logger.info(f"  {len(df_filtered):,} records, {len(df_filtered.columns)} columns")
    
    # Bac-only subset
    df_bac_only = df_filtered[df_filtered["is_bac_3as"]].copy()
    bac_only_path = output_dir / "videos_bac_only.csv"
    df_bac_only.to_csv(bac_only_path, index=False)
    logger.info(f"\nBac-only dataset: {bac_only_path}")
    logger.info(f"  {len(df_bac_only):,} videos")
    
    # Rejected subset
    df_rejected = df_filtered[~df_filtered["is_bac_3as"]].copy()
    rejected_path = output_dir / "videos_rejected.csv"
    df_rejected.to_csv(rejected_path, index=False)
    logger.info(f"\nRejected dataset: {rejected_path}")
    logger.info(f"  {len(df_rejected):,} videos")
    
    # =========================================================================
    # PHASE 3: Validation Sampling (Optional)
    # =========================================================================
    
    if args.validate:
        logger.info("\n" + "=" * 60)
        logger.info("PHASE 3: VALIDATION SAMPLING")
        logger.info("=" * 60)
        
        validation_dir = Path("data/validation")
        validation_dir.mkdir(parents=True, exist_ok=True)
        
        samples = []
        
        # Sample from bac_3as (explicit markers)
        bac_3as_mask = df_filtered["filter_category"] == "bac_3as"
        n_bac_3as = validation_config.get("sample_bac_3as", 100)
        if bac_3as_mask.sum() > 0:
            sample_bac = df_filtered[bac_3as_mask].sample(
                n=min(n_bac_3as, bac_3as_mask.sum()), random_state=42
            )
            sample_bac["sample_type"] = "bac_3as"
            samples.append(sample_bac)
            logger.info(f"Sampled {len(sample_bac)} from bac_3as")
        
        # Sample from bac_3as_ambiguous (soft positives)
        ambiguous_mask = df_filtered["filter_category"] == "bac_3as_ambiguous"
        n_ambiguous = validation_config.get("sample_bac_ambiguous", 100)
        if ambiguous_mask.sum() > 0:
            sample_amb = df_filtered[ambiguous_mask].sample(
                n=min(n_ambiguous, ambiguous_mask.sum()), random_state=42
            )
            sample_amb["sample_type"] = "bac_3as_ambiguous"
            samples.append(sample_amb)
            logger.info(f"Sampled {len(sample_amb)} from bac_3as_ambiguous")
        
        # Sample from non_bac (false negative check)
        non_bac_mask = df_filtered["filter_category"] == "non_bac"
        n_non_bac = validation_config.get("sample_non_bac", 50)
        if non_bac_mask.sum() > 0:
            sample_non = df_filtered[non_bac_mask].sample(
                n=min(n_non_bac, non_bac_mask.sum()), random_state=42
            )
            sample_non["sample_type"] = "non_bac"
            samples.append(sample_non)
            logger.info(f"Sampled {len(sample_non)} from non_bac")
        
        # Sample from unknown
        unknown_mask = df_filtered["filter_category"] == "unknown"
        n_conflict = validation_config.get("sample_conflict", 50)
        if unknown_mask.sum() > 0:
            sample_unk = df_filtered[unknown_mask].sample(
                n=min(n_conflict, unknown_mask.sum()), random_state=42
            )
            sample_unk["sample_type"] = "unknown"
            samples.append(sample_unk)
            logger.info(f"Sampled {len(sample_unk)} from unknown")
        
        # Combine samples
        if samples:
            combined_sample = pd.concat(samples, ignore_index=True)
            combined_sample["manual_is_bac"] = ""
            combined_sample["notes"] = ""
            
            sample_path = validation_dir / "sample_for_manual_review.csv"
            combined_sample.to_csv(sample_path, index=False)
            logger.info(f"\nValidation sample saved: {sample_path}")
            logger.info(f"Total samples: {len(combined_sample)}")
    
    logger.info("\n" + "=" * 60)
    logger.info("FILTER COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Ready for ML modeling with {len(df_bac_only):,} Bac 3AS videos")
    
    return 0


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
        input_for_engineer = Path("data/processed/videos_bac_only.csv")
    else:
        input_for_engineer = Path("data/raw/videos_metadata.csv")
    
    engineer_args = argparse.Namespace(
        input=input_for_engineer,
        output=Path("data/processed/videos_engineered.csv"),
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
    
    # Parse and execute
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 0
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
