#!/usr/bin/env python
"""
Build Channel Priors: Compute per-channel Bac vs non-Bac ratios.

This script automatically computes channel-level grade evidence rates from
actual video content, enabling data-driven "Bac-heavy" channel classification.

Output: data/processed/channel_priors.csv

Usage:
    uv run python scripts/02_build_channel_priors.py
    uv run python scripts/02_build_channel_priors.py --bac-threshold 0.8 --non-bac-max 0.05
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Fix console encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def load_config() -> dict:
    """Load filter configuration from YAML."""
    import yaml
    
    config_path = PROJECT_ROOT / "config" / "filter_config.yaml"
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    
    # Default config
    return {
        "channel_prior": {
            "bac_threshold": 0.7,
            "non_bac_max": 0.1,
        },
        "markers": {
            "bac": ["bac", "3as", "بكالوريا", "باك", "ثالثة ثانوي", "السنة الثالثة ثانوي", "terminale"],
            "non_bac": ["1as", "2as", "سنة أولى ثانوي", "سنة ثانية ثانوي", "أولى ثانوي", "ثانية ثانوي", "متوسط", "bem", "1am", "2am", "3am", "4am", "ابتدائي"],
        },
        "paths": {
            "input": "data/raw/videos_metadata.csv",
            "output_dir": "data/processed",
        }
    }


def has_markers(text_series: pd.Series, markers: list) -> pd.Series:
    """Check if text contains any of the markers."""
    pattern = "|".join([re.escape(m) for m in markers])
    return text_series.str.contains(pattern, regex=True, case=False, na=False)


def build_channel_priors(
    input_path: Path,
    output_path: Path,
    bac_markers: list,
    non_bac_markers: list,
    bac_threshold: float = 0.7,
    non_bac_max: float = 0.1,
) -> pd.DataFrame:
    """
    Build channel priors from video metadata.
    
    Args:
        input_path: Path to videos_metadata.csv
        output_path: Path to save channel_priors.csv
        bac_markers: List of Bac markers
        non_bac_markers: List of non-Bac markers
        bac_threshold: Threshold for p_bac to mark as Bac-heavy
        non_bac_max: Maximum p_non to mark as Bac-heavy
    
    Returns:
        DataFrame with channel priors
    """
    print("=" * 60)
    print("BUILD CHANNEL PRIORS")
    print("=" * 60)
    
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Bac threshold: {bac_threshold}")
    print(f"Non-Bac max: {non_bac_max}")
    
    # Load data
    print("\nLoading video metadata...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records")
    
    # Deduplicate to latest snapshot
    if "snapshot_date" in df.columns:
        print("Deduplicating to latest snapshot per video...")
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="ISO8601", errors="coerce")
        df = df.sort_values("snapshot_date", ascending=False)
    
    if "video_id" in df.columns:
        original = len(df)
        df = df.drop_duplicates(subset=["video_id"], keep="first")
        print(f"Deduplicated: {original:,} → {len(df):,} unique videos")
    
    # Combine text fields
    text = (
        df["title"].fillna("").astype(str) + " "
        + df["description"].fillna("").astype(str) + " "
        + df["tags"].fillna("").astype(str)
    ).str.lower()
    
    # Check for markers
    df["has_bac"] = has_markers(text, bac_markers)
    df["has_non_bac"] = has_markers(text, non_bac_markers)
    df["has_any_grade"] = df["has_bac"] | df["has_non_bac"]
    
    # Aggregate by channel
    print("\nComputing per-channel statistics...")
    channel_stats = df.groupby("channel_id").agg({
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
    
    # Compute ratios
    channel_stats["p_bac"] = (
        channel_stats["count_bac_marked"] / 
        channel_stats["count_any_grade_marked"].replace(0, 1)
    )
    channel_stats["p_non"] = (
        channel_stats["count_non_bac_marked"] / 
        channel_stats["count_any_grade_marked"].replace(0, 1)
    )
    
    # Mark Bac-heavy channels
    channel_stats["is_bac_heavy"] = (
        (channel_stats["p_bac"] >= bac_threshold) & 
        (channel_stats["p_non"] <= non_bac_max)
    )
    
    # Reset index
    channel_stats = channel_stats.reset_index()
    
    # Sort by p_bac descending
    channel_stats = channel_stats.sort_values("p_bac", ascending=False)
    
    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    channel_stats.to_csv(output_path, index=False)
    print(f"\nSaved channel priors to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CHANNEL PRIOR SUMMARY")
    print("=" * 60)
    
    total_channels = len(channel_stats)
    bac_heavy = channel_stats["is_bac_heavy"].sum()
    
    print(f"\nTotal channels: {total_channels}")
    print(f"Bac-heavy channels: {bac_heavy} ({bac_heavy/total_channels*100:.0f}%)")
    print(f"Non Bac-heavy channels: {total_channels - bac_heavy}")
    
    # p_bac distribution
    print("\np_bac distribution:")
    print(f"  Mean:   {channel_stats['p_bac'].mean():.3f}")
    print(f"  Median: {channel_stats['p_bac'].median():.3f}")
    print(f"  Std:    {channel_stats['p_bac'].std():.3f}")
    print(f"  Min:    {channel_stats['p_bac'].min():.3f}")
    print(f"  Max:    {channel_stats['p_bac'].max():.3f}")
    
    # Show all channels
    print("\nAll channels (sorted by p_bac):")
    print("-" * 80)
    for _, row in channel_stats.iterrows():
        heavy = "BAC-HEAVY" if row["is_bac_heavy"] else ""
        ch_id = str(row["channel_id"])[:24]
        print(
            f"  {ch_id:24s} "
            f"p_bac={row['p_bac']:.2f} "
            f"p_non={row['p_non']:.2f} "
            f"(bac={int(row['count_bac_marked']):3d} "
            f"non={int(row['count_non_bac_marked']):3d} "
            f"total={int(row['total_videos']):4d}) "
            f"{heavy}"
        )
    
    return channel_stats


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Build channel priors from video metadata"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input video metadata CSV"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output channel priors CSV"
    )
    parser.add_argument(
        "--bac-threshold", type=float, default=None,
        help="p_bac threshold for Bac-heavy (default: 0.7)"
    )
    parser.add_argument(
        "--non-bac-max", type=float, default=None,
        help="Max p_non for Bac-heavy (default: 0.1)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    
    # Resolve paths
    input_path = args.input or PROJECT_ROOT / config["paths"]["input"]
    output_path = args.output or PROJECT_ROOT / config["paths"]["output_dir"] / "channel_priors.csv"
    
    # Resolve thresholds
    bac_threshold = args.bac_threshold or config["channel_prior"]["bac_threshold"]
    non_bac_max = args.non_bac_max or config["channel_prior"]["non_bac_max"]
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Build priors
    try:
        build_channel_priors(
            input_path=input_path,
            output_path=output_path,
            bac_markers=config["markers"]["bac"],
            non_bac_markers=config["markers"]["non_bac"],
            bac_threshold=bac_threshold,
            non_bac_max=non_bac_max,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
