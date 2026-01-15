#!/usr/bin/env python
"""
Generate validation samples for manual review of Bac filter accuracy.

This script creates stratified samples from the filtered dataset to enable
manual accuracy validation. It samples:
1. By subject (equal representation)
2. By confidence level (including edge cases near threshold)
3. From rejected videos (to find false negatives)

Usage:
    uv run python scripts/validate_filter.py
    uv run python scripts/validate_filter.py --per-subject 30
    uv run python scripts/validate_filter.py --rejected-sample 100
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


def sample_by_subject(
    df: pd.DataFrame, n_per_subject: int, random_state: int = 42
) -> pd.DataFrame:
    """
    Sample n videos per detected subject.

    Args:
        df: Filtered DataFrame (Bac videos only)
        n_per_subject: Number of samples per subject
        random_state: Random seed for reproducibility

    Returns:
        Sampled DataFrame
    """
    samples = []
    for subject in df["detected_subject"].unique():
        subject_df = df[df["detected_subject"] == subject]
        n_sample = min(n_per_subject, len(subject_df))
        if n_sample > 0:
            sample = subject_df.sample(n=n_sample, random_state=random_state)
            samples.append(sample)

    if samples:
        return pd.concat(samples, ignore_index=True)
    return pd.DataFrame()


def sample_by_confidence(
    df: pd.DataFrame, n_per_band: int, random_state: int = 42
) -> pd.DataFrame:
    """
    Sample videos from different confidence bands.

    Args:
        df: Filtered DataFrame
        n_per_band: Number of samples per confidence band
        random_state: Random seed

    Returns:
        Sampled DataFrame with confidence band labels
    """
    # Define confidence bands
    bands = [
        ("low", 0.6, 0.7),
        ("medium", 0.7, 0.8),
        ("high", 0.8, 0.9),
        ("very_high", 0.9, 1.01),
    ]

    samples = []
    for band_name, low, high in bands:
        band_df = df[(df["filter_confidence"] >= low) & (df["filter_confidence"] < high)]
        n_sample = min(n_per_band, len(band_df))
        if n_sample > 0:
            sample = band_df.sample(n=n_sample, random_state=random_state)
            sample = sample.copy()
            sample["confidence_band"] = band_name
            samples.append(sample)

    if samples:
        return pd.concat(samples, ignore_index=True)
    return pd.DataFrame()


def sample_edge_cases(
    df: pd.DataFrame, n_samples: int, threshold_range: float = 0.05, random_state: int = 42
) -> pd.DataFrame:
    """
    Sample videos near the decision threshold (edge cases).

    Args:
        df: Full filtered DataFrame (both Bac and non-Bac)
        n_samples: Total number of edge case samples
        threshold_range: Range around threshold to consider (e.g., 0.05 = ±0.05)
        random_state: Random seed

    Returns:
        Sampled edge cases
    """
    # Get videos near typical thresholds (0.65-0.75)
    edge_df = df[
        (df["filter_confidence"] >= 0.60) & (df["filter_confidence"] <= 0.75)
    ]

    n_sample = min(n_samples, len(edge_df))
    if n_sample > 0:
        sample = edge_df.sample(n=n_sample, random_state=random_state)
        sample = sample.copy()
        sample["sample_type"] = "edge_case"
        return sample

    return pd.DataFrame()


def main() -> None:
    """Main entry point for validation sampling."""
    parser = argparse.ArgumentParser(
        description="Generate validation samples for Bac filter review",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "videos_with_bac_filter.csv",
        help="Path to filtered video metadata CSV",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "validation",
        help="Directory for validation samples",
    )

    parser.add_argument(
        "--per-subject",
        type=int,
        default=50,
        help="Number of samples per subject (default: 50)",
    )

    parser.add_argument(
        "--per-confidence-band",
        type=int,
        default=25,
        help="Number of samples per confidence band (default: 25)",
    )

    parser.add_argument(
        "--edge-cases",
        type=int,
        default=50,
        help="Number of edge case samples near threshold (default: 50)",
    )

    parser.add_argument(
        "--rejected-sample",
        type=int,
        default=100,
        help="Number of rejected (non-Bac) videos to sample for false negative check (default: 100)",
    )

    args = parser.parse_args()

    # Validate input
    if not args.input.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.info("Run apply_bac_filter.py first to generate filtered data.")
        sys.exit(1)

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Bac Filter Validation Sampling")
    logger.info("=" * 60)
    logger.info(f"Input: {args.input}")
    logger.info(f"Output directory: {args.output_dir}")

    # Load data
    logger.info("\nLoading filtered data...")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df):,} records")

    df_bac = df[df["is_bac_3as"]].copy()
    df_non_bac = df[~df["is_bac_3as"]].copy()
    logger.info(f"Bac videos: {len(df_bac):,}")
    logger.info(f"Non-Bac videos: {len(df_non_bac):,}")

    # Sample 1: By subject (stratified)
    logger.info(f"\n1. Sampling {args.per_subject} videos per subject...")
    sample_subject = sample_by_subject(df_bac, args.per_subject)
    sample_subject["sample_type"] = "by_subject"
    logger.info(f"   Collected {len(sample_subject)} samples from {sample_subject['detected_subject'].nunique()} subjects")

    # Sample 2: By confidence band
    logger.info(f"\n2. Sampling {args.per_confidence_band} videos per confidence band...")
    sample_confidence = sample_by_confidence(df_bac, args.per_confidence_band)
    if "sample_type" not in sample_confidence.columns:
        sample_confidence["sample_type"] = "by_confidence"
    logger.info(f"   Collected {len(sample_confidence)} samples")

    # Sample 3: Edge cases
    logger.info(f"\n3. Sampling {args.edge_cases} edge cases near threshold...")
    sample_edges = sample_edge_cases(df, args.edge_cases)
    logger.info(f"   Collected {len(sample_edges)} edge case samples")

    # Sample 4: Rejected videos (for false negative detection)
    logger.info(f"\n4. Sampling {args.rejected_sample} rejected videos for false negative check...")
    n_rejected_sample = min(args.rejected_sample, len(df_non_bac))
    if n_rejected_sample > 0:
        sample_rejected = df_non_bac.sample(n=n_rejected_sample, random_state=42)
        sample_rejected = sample_rejected.copy()
        sample_rejected["sample_type"] = "rejected_check"
    else:
        sample_rejected = pd.DataFrame()
    logger.info(f"   Collected {len(sample_rejected)} rejected samples")

    # Combine all samples
    all_samples = pd.concat(
        [sample_subject, sample_confidence, sample_edges, sample_rejected],
        ignore_index=True,
    )

    # Remove duplicates (videos may appear in multiple samples)
    initial_count = len(all_samples)
    all_samples = all_samples.drop_duplicates(subset=["video_id"], keep="first")
    logger.info(f"\nRemoved {initial_count - len(all_samples)} duplicate samples")
    logger.info(f"Total unique samples: {len(all_samples)}")

    # Select columns for review
    review_columns = [
        "video_id",
        "title",
        "description",
        "channel_title",
        "detected_subject",
        "filter_confidence",
        "is_bac_3as",
        "bac_keyword_count",
        "exclude_keyword_count",
        "filter_reason",
        "sample_type",
    ]

    # Filter to available columns
    available_columns = [c for c in review_columns if c in all_samples.columns]
    review_df = all_samples[available_columns].copy()

    # Add manual review columns
    review_df["manual_is_bac"] = ""  # For reviewer to fill: TRUE/FALSE
    review_df["manual_subject_correct"] = ""  # For reviewer: TRUE/FALSE
    review_df["notes"] = ""  # For reviewer notes

    # Save outputs
    logger.info("\n" + "=" * 60)
    logger.info("SAVING VALIDATION SAMPLES")
    logger.info("=" * 60)

    # Main review file
    review_output = args.output_dir / "sample_for_manual_review.csv"
    review_df.to_csv(review_output, index=False)
    logger.info(f"Main review file: {review_output}")
    logger.info(f"  {len(review_df)} samples for manual review")

    # Separate files by sample type for easier review
    for sample_type in review_df["sample_type"].unique():
        type_df = review_df[review_df["sample_type"] == sample_type]
        type_output = args.output_dir / f"sample_{sample_type}.csv"
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

2. manual_subject_correct: TRUE if detected_subject is correct, FALSE otherwise
   - Leave empty if is_bac_3as is FALSE

3. notes: Any observations about why classification was wrong

After review, calculate accuracy:
   accuracy = (correct classifications) / (total reviewed)

If accuracy < 80%:
   - Identify common failure patterns
   - Adjust thresholds in src/features/bac_keywords.py
   - Add missing keywords or exclusion terms
   - Re-run: uv run python scripts/apply_bac_filter.py
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
