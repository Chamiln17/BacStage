#!/usr/bin/env python
"""
Generate validation samples for manual review of Bac filter accuracy.

This script creates stratified samples from the balanced filter output to enable
manual accuracy validation. It samples from:
1. bac_3as: Explicit Bac markers (precision check)
2. bac_3as_ambiguous: Soft positives from Bac-heavy channels
3. non_bac: Rejected videos (false negative check)
4. unknown: Videos without grade markers from non Bac-heavy channels

Usage:
    uv run python scripts/validate_filter.py
    uv run python scripts/validate_filter.py --input data/processed/videos_bac_balanced.csv
    uv run python scripts/validate_filter.py --sample-bac 150 --sample-ambiguous 100
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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


def sample_by_category(
    df: pd.DataFrame,
    category: str,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample n videos from a specific filter category.
    
    Args:
        df: Filtered DataFrame with filter_category column
        category: Category to sample from
        n_samples: Number of samples to take
        random_state: Random seed for reproducibility
    
    Returns:
        Sampled DataFrame
    """
    category_df = df[df["filter_category"] == category]
    n_sample = min(n_samples, len(category_df))
    
    if n_sample > 0:
        sample = category_df.sample(n=n_sample, random_state=random_state)
        sample = sample.copy()
        sample["sample_type"] = category
        return sample
    
    return pd.DataFrame()


def sample_by_confidence(
    df: pd.DataFrame,
    n_per_band: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample videos from different confidence bands.
    
    Args:
        df: Filtered DataFrame with filter_confidence column
        n_per_band: Number of samples per confidence band
        random_state: Random seed
    
    Returns:
        Sampled DataFrame with confidence band labels
    """
    bands = [
        ("very_low", 0.0, 0.4),
        ("low", 0.4, 0.6),
        ("medium", 0.6, 0.75),
        ("high", 0.75, 0.9),
        ("very_high", 0.9, 1.01),
    ]
    
    samples = []
    for band_name, low, high in bands:
        band_df = df[
            (df["filter_confidence"] >= low) & 
            (df["filter_confidence"] < high)
        ]
        n_sample = min(n_per_band, len(band_df))
        if n_sample > 0:
            sample = band_df.sample(n=n_sample, random_state=random_state)
            sample = sample.copy()
            sample["sample_type"] = f"confidence_{band_name}"
            sample["confidence_band"] = band_name
            samples.append(sample)
    
    if samples:
        return pd.concat(samples, ignore_index=True)
    return pd.DataFrame()


def sample_by_subject(
    df: pd.DataFrame,
    n_per_subject: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample n videos per subject (from channel mapping).
    
    Args:
        df: Filtered DataFrame with subject column
        n_per_subject: Number of samples per subject
        random_state: Random seed
    
    Returns:
        Sampled DataFrame
    """
    if "subject" not in df.columns:
        return pd.DataFrame()
    
    # Only sample from Bac videos
    bac_df = df[df["is_bac_3as"]]
    
    samples = []
    for subject in bac_df["subject"].unique():
        if pd.isna(subject) or subject == "Unknown":
            continue
        subject_df = bac_df[bac_df["subject"] == subject]
        n_sample = min(n_per_subject, len(subject_df))
        if n_sample > 0:
            sample = subject_df.sample(n=n_sample, random_state=random_state)
            sample = sample.copy()
            sample["sample_type"] = f"subject_{subject.replace(' ', '_').lower()}"
            samples.append(sample)
    
    if samples:
        return pd.concat(samples, ignore_index=True)
    return pd.DataFrame()


def sample_conflicts(
    df: pd.DataFrame,
    n_samples: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Sample videos where both Bac and non-Bac markers were present.
    
    Args:
        df: Filtered DataFrame
        n_samples: Number of samples
        random_state: Random seed
    
    Returns:
        Sampled conflict cases
    """
    # Look for conflict resolution in filter_reason
    conflict_df = df[
        df["filter_reason"].fillna("").str.contains("conflict|both", case=False)
    ]
    
    n_sample = min(n_samples, len(conflict_df))
    if n_sample > 0:
        sample = conflict_df.sample(n=n_sample, random_state=random_state)
        sample = sample.copy()
        sample["sample_type"] = "conflict"
        return sample
    
    return pd.DataFrame()


def main() -> None:
    """Main entry point for validation sampling."""
    parser = argparse.ArgumentParser(
        description="Generate validation samples for balanced Bac filter review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "videos_bac_balanced.csv",
        help="Path to filtered video metadata CSV (default: videos_bac_balanced.csv)",
    )
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation",
        help="Directory for validation samples",
    )
    
    parser.add_argument(
        "--sample-bac",
        type=int,
        default=100,
        help="Number of samples from bac_3as category (default: 100)",
    )
    
    parser.add_argument(
        "--sample-ambiguous",
        type=int,
        default=100,
        help="Number of samples from bac_3as_ambiguous category (default: 100)",
    )
    
    parser.add_argument(
        "--sample-non-bac",
        type=int,
        default=50,
        help="Number of samples from non_bac category (default: 50)",
    )
    
    parser.add_argument(
        "--sample-unknown",
        type=int,
        default=50,
        help="Number of samples from unknown category (default: 50)",
    )
    
    parser.add_argument(
        "--sample-per-confidence",
        type=int,
        default=20,
        help="Number of samples per confidence band (default: 20)",
    )
    
    parser.add_argument(
        "--sample-per-subject",
        type=int,
        default=20,
        help="Number of samples per subject (default: 20)",
    )
    
    parser.add_argument(
        "--sample-conflicts",
        type=int,
        default=30,
        help="Number of conflict case samples (default: 30)",
    )
    
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        # Try fallback paths
        fallback_paths = [
            PROJECT_ROOT / "data" / "processed" / "videos_with_bac_filter.csv",
            PROJECT_ROOT / "data" / "processed" / "videos_bac_only.csv",
        ]
        
        for fallback in fallback_paths:
            if fallback.exists():
                logger.warning(f"Primary input not found, using: {fallback}")
                args.input = fallback
                break
        else:
            logger.error(f"Input file not found: {args.input}")
            logger.info("Run 'python run_pipeline.py filter_data' first to generate filtered data.")
            sys.exit(1)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=" * 60)
    logger.info("Balanced Bac Filter Validation Sampling")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Load data
    logger.info("\nLoading filtered data...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} records")
    
    # Check for required columns
    has_filter_category = "filter_category" in df.columns
    has_filter_confidence = "filter_confidence" in df.columns
    has_is_bac = "is_bac_3as" in df.columns
    
    if not has_is_bac:
        logger.error("Missing required column: is_bac_3as")
        sys.exit(1)
    
    # Summary
    df_bac = df[df["is_bac_3as"]].copy()
    df_non_bac = df[~df["is_bac_3as"]].copy()
    logger.info(f"Bac videos: {len(df_bac):,}")
    logger.info(f"Non-Bac videos: {len(df_non_bac):,}")
    
    if has_filter_category:
        logger.info("\nCategory distribution:")
        for cat, count in df["filter_category"].value_counts().items():
            logger.info(f"  {cat}: {count:,}")
    
    all_samples = []
    
    # Sample 1: By category
    if has_filter_category:
        logger.info(f"\n1. Sampling by filter category...")
        
        sample_bac = sample_by_category(df, "bac_3as", args.sample_bac)
        if len(sample_bac) > 0:
            logger.info(f"   bac_3as: {len(sample_bac)} samples")
            all_samples.append(sample_bac)
        
        sample_ambig = sample_by_category(df, "bac_3as_ambiguous", args.sample_ambiguous)
        if len(sample_ambig) > 0:
            logger.info(f"   bac_3as_ambiguous: {len(sample_ambig)} samples")
            all_samples.append(sample_ambig)
        
        sample_non = sample_by_category(df, "non_bac", args.sample_non_bac)
        if len(sample_non) > 0:
            logger.info(f"   non_bac: {len(sample_non)} samples")
            all_samples.append(sample_non)
        
        sample_unk = sample_by_category(df, "unknown", args.sample_unknown)
        if len(sample_unk) > 0:
            logger.info(f"   unknown: {len(sample_unk)} samples")
            all_samples.append(sample_unk)
    else:
        # Fallback: sample from Bac and non-Bac
        logger.info(f"\n1. Sampling from Bac/non-Bac (no category column)...")
        
        n_bac_sample = min(args.sample_bac + args.sample_ambiguous, len(df_bac))
        if n_bac_sample > 0:
            sample_bac = df_bac.sample(n=n_bac_sample, random_state=42)
            sample_bac = sample_bac.copy()
            sample_bac["sample_type"] = "bac"
            all_samples.append(sample_bac)
            logger.info(f"   bac: {len(sample_bac)} samples")
        
        n_non_sample = min(args.sample_non_bac, len(df_non_bac))
        if n_non_sample > 0:
            sample_non = df_non_bac.sample(n=n_non_sample, random_state=42)
            sample_non = sample_non.copy()
            sample_non["sample_type"] = "non_bac"
            all_samples.append(sample_non)
            logger.info(f"   non_bac: {len(sample_non)} samples")
    
    # Sample 2: By confidence
    if has_filter_confidence:
        logger.info(f"\n2. Sampling by confidence band ({args.sample_per_confidence} per band)...")
        sample_conf = sample_by_confidence(df, args.sample_per_confidence)
        if len(sample_conf) > 0:
            logger.info(f"   Total confidence samples: {len(sample_conf)}")
            all_samples.append(sample_conf)
    
    # Sample 3: By subject
    if "subject" in df.columns:
        logger.info(f"\n3. Sampling by subject ({args.sample_per_subject} per subject)...")
        sample_subj = sample_by_subject(df, args.sample_per_subject)
        if len(sample_subj) > 0:
            logger.info(f"   Total subject samples: {len(sample_subj)}")
            all_samples.append(sample_subj)
    
    # Sample 4: Conflicts
    if "filter_reason" in df.columns:
        logger.info(f"\n4. Sampling conflict cases...")
        sample_conflict = sample_conflicts(df, args.sample_conflicts)
        if len(sample_conflict) > 0:
            logger.info(f"   Conflict samples: {len(sample_conflict)}")
            all_samples.append(sample_conflict)
    
    # Combine all samples
    if not all_samples:
        logger.error("No samples collected!")
        sys.exit(1)
    
    combined = pd.concat(all_samples, ignore_index=True)
    
    # Remove duplicates
    initial_count = len(combined)
    if "video_id" in combined.columns:
        combined = combined.drop_duplicates(subset=["video_id"], keep="first")
    logger.info(f"\nRemoved {initial_count - len(combined)} duplicate samples")
    logger.info(f"Total unique samples: {len(combined)}")
    
    # Select columns for review
    priority_columns = [
        "video_id",
        "title",
        "description",
        "channel_title",
        "channel_id",
        "subject",
        "filter_category",
        "filter_confidence",
        "filter_reason",
        "is_bac_3as",
        "sample_type",
    ]
    
    # Keep available columns in priority order, then add any extras
    available = [c for c in priority_columns if c in combined.columns]
    extra_cols = [c for c in combined.columns if c not in available and c != "sample_type"]
    
    # Limit extra columns to avoid huge CSV
    extra_cols = extra_cols[:10]
    
    review_df = combined[available + extra_cols].copy()
    
    # Add manual review columns
    review_df["manual_is_bac"] = ""
    review_df["notes"] = ""
    
    # Save outputs
    logger.info("\n" + "=" * 60)
    logger.info("SAVING VALIDATION SAMPLES")
    logger.info("=" * 60)
    
    # Main review file
    review_output = args.output_dir / "sample_for_manual_review.csv"
    review_df.to_csv(review_output, index=False)
    logger.info(f"\nMain review file: {review_output}")
    logger.info(f"  {len(review_df)} samples for manual review")
    
    # Separate files by sample type
    for sample_type in review_df["sample_type"].unique():
        type_df = review_df[review_df["sample_type"] == sample_type]
        # Clean filename
        safe_name = str(sample_type).replace("/", "_").replace(" ", "_")
        type_output = args.output_dir / f"sample_{safe_name}.csv"
        type_df.to_csv(type_output, index=False)
        logger.info(f"  {sample_type}: {len(type_df)} samples -> {type_output.name}")
    
    # Print review instructions
    logger.info("\n" + "=" * 60)
    logger.info("MANUAL REVIEW INSTRUCTIONS")
    logger.info("=" * 60)
    logger.info("""
For each video in the sample, please fill in:

1. manual_is_bac: TRUE if video is actually Bac 3AS content, FALSE otherwise
   - Check if content targets Bac-level students (not 1AS, 2AS, or middle school)
   - Verify depth matches Bac curriculum requirements

2. notes: Any observations about why classification was wrong

After review, calculate metrics:

   Precision (bac_3as) = (correct in bac_3as) / (total bac_3as reviewed)
   Precision (ambiguous) = (correct in bac_3as_ambiguous) / (total ambiguous reviewed)
   False Negative Rate = (FN in non_bac) / (total non_bac reviewed)

If precision < 80%:
   - For bac_3as: Review Bac marker list, may be too broad
   - For ambiguous: Tighten soft positives in config/filter_config.yaml
   
If false negative rate > 20%:
   - Relax soft positives or add missing Bac markers
   - Consider lowering channel prior thresholds

Re-run filter after adjustments:
   uv run python run_pipeline.py filter_data --force-discovery
""")
    
    # Sample type distribution
    logger.info("\nSample Type Distribution:")
    for sample_type, count in review_df["sample_type"].value_counts().items():
        logger.info(f"  {sample_type}: {count}")
    
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION SAMPLING COMPLETE")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
