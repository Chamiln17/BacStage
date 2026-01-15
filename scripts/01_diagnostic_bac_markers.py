#!/usr/bin/env python
"""
Diagnostic Script: Quantify the "No Subject + Bac Marker" Opportunity

This script analyzes how many videos were rejected for "No subject detected"
but actually contain explicit Bac markers, helping quantify the improvement
potential of the new data-driven filtering approach.

Usage:
    uv run python scripts/01_diagnostic_bac_markers.py
"""

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
    
    # Default markers if config doesn't exist
    return {
        "markers": {
            "bac": ["bac", "3as", "بكالوريا", "باك", "ثالثة ثانوي", "السنة الثالثة ثانوي", "terminale"],
            "non_bac": ["1as", "2as", "سنة أولى ثانوي", "سنة ثانية ثانوي", "أولى ثانوي", "ثانية ثانوي", "متوسط", "bem", "1am", "2am", "3am", "4am", "ابتدائي"],
        }
    }


def has_markers(text_series: pd.Series, markers: list) -> pd.Series:
    """Check if text contains any of the markers."""
    pattern = "|".join([re.escape(m) for m in markers])
    return text_series.str.contains(pattern, regex=True, case=False, na=False)


def run_diagnostic() -> None:
    """Run the diagnostic analysis."""
    print("=" * 60)
    print("DIAGNOSTIC: No Subject + Bac Marker Analysis")
    print("=" * 60)
    
    # Load config
    config = load_config()
    bac_markers = config["markers"]["bac"]
    non_bac_markers = config["markers"]["non_bac"]
    
    print(f"\nBac markers ({len(bac_markers)}): {bac_markers[:5]}...")
    print(f"Non-Bac markers ({len(non_bac_markers)}): {non_bac_markers[:5]}...")
    
    # Try to load the existing filtered data
    filtered_path = PROJECT_ROOT / "data" / "processed" / "videos_with_bac_filter.csv"
    raw_path = PROJECT_ROOT / "data" / "raw" / "videos_metadata.csv"
    
    if filtered_path.exists():
        print(f"\nLoading filtered data from {filtered_path}...")
        df = pd.read_csv(filtered_path)
        has_filter_reason = "filter_reason" in df.columns
    else:
        print(f"\nFiltered data not found. Loading raw data from {raw_path}...")
        df = pd.read_csv(raw_path)
        has_filter_reason = False
    
    print(f"Loaded {len(df):,} records")
    
    # Deduplicate if needed
    if "video_id" in df.columns:
        original_count = len(df)
        if "snapshot_date" in df.columns:
            df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="ISO8601", errors="coerce")
            df = df.sort_values("snapshot_date", ascending=False)
        df = df.drop_duplicates(subset=["video_id"], keep="first")
        if len(df) < original_count:
            print(f"Deduplicated: {original_count:,} → {len(df):,} unique videos")
    
    # Combine text fields
    text = (
        df["title"].fillna("").astype(str) + " "
        + df["description"].fillna("").astype(str) + " "
        + df["tags"].fillna("").astype(str)
    ).str.lower()
    
    # Check for markers
    df["has_bac_marker"] = has_markers(text, bac_markers)
    df["has_non_bac_marker"] = has_markers(text, non_bac_markers)
    
    # Overall statistics
    print("\n" + "=" * 60)
    print("OVERALL MARKER STATISTICS")
    print("=" * 60)
    
    total = len(df)
    has_bac = df["has_bac_marker"].sum()
    has_non_bac = df["has_non_bac_marker"].sum()
    has_both = (df["has_bac_marker"] & df["has_non_bac_marker"]).sum()
    has_neither = (~df["has_bac_marker"] & ~df["has_non_bac_marker"]).sum()
    
    print(f"\nTotal unique videos: {total:,}")
    print(f"Has Bac markers:     {has_bac:,} ({has_bac/total*100:.1f}%)")
    print(f"Has non-Bac markers: {has_non_bac:,} ({has_non_bac/total*100:.1f}%)")
    print(f"Has both:            {has_both:,} ({has_both/total*100:.1f}%)")
    print(f"Has neither:         {has_neither:,} ({has_neither/total*100:.1f}%)")
    
    # If we have filter_reason from old filter, analyze "No subject detected"
    if has_filter_reason:
        print("\n" + "=" * 60)
        print("'NO SUBJECT DETECTED' ANALYSIS")
        print("=" * 60)
        
        no_subject = df["filter_reason"].fillna("").str.contains("No subject detected", case=False)
        no_subject_count = no_subject.sum()
        
        if no_subject_count > 0:
            no_subject_with_bac = (no_subject & df["has_bac_marker"]).sum()
            no_subject_with_non_bac = (no_subject & df["has_non_bac_marker"]).sum()
            no_subject_bac_only = (no_subject & df["has_bac_marker"] & ~df["has_non_bac_marker"]).sum()
            
            print(f"\nVideos with 'No subject detected': {no_subject_count:,}")
            print(f"  ...with Bac markers:     {no_subject_with_bac:,} ({no_subject_with_bac/no_subject_count*100:.1f}%)")
            print(f"  ...with non-Bac markers: {no_subject_with_non_bac:,} ({no_subject_with_non_bac/no_subject_count*100:.1f}%)")
            print(f"  ...Bac ONLY (no non-Bac): {no_subject_bac_only:,} ({no_subject_bac_only/no_subject_count*100:.1f}%)")
            
            if no_subject_with_bac / no_subject_count > 0.3:
                print(f"\n>>> INSIGHT: {no_subject_with_bac/no_subject_count*100:.0f}% of 'No subject' videos have Bac markers!")
                print(">>> 'Explicit Bac => keep' will unlock significant data.")
            else:
                print(f"\n>>> Share with Bac markers is {no_subject_with_bac/no_subject_count*100:.0f}%")
        
        # Filter reason distribution
        print("\n" + "=" * 60)
        print("FILTER REASON DISTRIBUTION")
        print("=" * 60)
        
        reason_counts = df["filter_reason"].value_counts().head(10)
        for reason, count in reason_counts.items():
            pct = count / total * 100
            reason_short = str(reason)[:50] if pd.notna(reason) else "N/A"
            print(f"  {reason_short:50s} {count:>6,} ({pct:5.1f}%)")
    
    # Analyze "ambiguous" videos (no grade markers)
    print("\n" + "=" * 60)
    print("AMBIGUOUS VIDEOS (NO GRADE MARKERS)")
    print("=" * 60)
    
    ambiguous = ~df["has_bac_marker"] & ~df["has_non_bac_marker"]
    ambiguous_count = ambiguous.sum()
    print(f"\nAmbiguous videos: {ambiguous_count:,} ({ambiguous_count/total*100:.1f}%)")
    
    # Check duration distribution for ambiguous videos
    if "duration_sec" in df.columns:
        ambiguous_long = (ambiguous & (df["duration_sec"] >= 300)).sum()
        print(f"  ...with duration >= 5 min: {ambiguous_long:,} ({ambiguous_long/ambiguous_count*100:.1f}% of ambiguous)")
    
    # Channel distribution for ambiguous
    if "channel_id" in df.columns:
        channels_with_ambiguous = df[ambiguous]["channel_id"].nunique()
        print(f"  ...across {channels_with_ambiguous} channels")
    
    # By-channel Bac marker rates (preview of channel priors)
    print("\n" + "=" * 60)
    print("PER-CHANNEL BAC MARKER RATES (Preview)")
    print("=" * 60)
    
    if "channel_id" in df.columns:
        channel_stats = df.groupby("channel_id").agg({
            "has_bac_marker": ["sum", "count"],
            "has_non_bac_marker": "sum",
        })
        channel_stats.columns = ["bac_count", "total", "non_bac_count"]
        channel_stats["any_grade"] = channel_stats["bac_count"] + channel_stats["non_bac_count"]
        channel_stats["p_bac"] = channel_stats["bac_count"] / channel_stats["any_grade"].replace(0, 1)
        channel_stats["p_non"] = channel_stats["non_bac_count"] / channel_stats["any_grade"].replace(0, 1)
        
        # Mark Bac-heavy
        channel_stats["is_bac_heavy"] = (channel_stats["p_bac"] >= 0.7) & (channel_stats["p_non"] <= 0.1)
        
        bac_heavy_count = channel_stats["is_bac_heavy"].sum()
        total_channels = len(channel_stats)
        
        print(f"\nTotal channels: {total_channels}")
        print(f"Bac-heavy channels (p_bac >= 0.7, p_non <= 0.1): {bac_heavy_count} ({bac_heavy_count/total_channels*100:.0f}%)")
        
        # Show top channels
        print("\nTop 10 channels by p_bac:")
        top_channels = channel_stats.sort_values("p_bac", ascending=False).head(10)
        for ch_id, row in top_channels.iterrows():
            heavy = "BAC-HEAVY" if row["is_bac_heavy"] else ""
            print(f"  {ch_id[:20]:20s} p_bac={row['p_bac']:.2f} p_non={row['p_non']:.2f} (n={int(row['total'])}) {heavy}")
    
    # Summary recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    bac_only = (df["has_bac_marker"] & ~df["has_non_bac_marker"]).sum()
    print(f"\n1. Videos with Bac markers only (clean Bac): {bac_only:,}")
    print(f"   → These should be INCLUDED immediately")
    
    print(f"\n2. Ambiguous videos (no markers): {ambiguous_count:,}")
    print(f"   → Use channel prior + soft positives to decide")
    
    print(f"\n3. Videos with both markers (conflicts): {has_both:,}")
    print(f"   → Use strong Bac intent phrases to resolve")
    
    non_bac_only = (df["has_non_bac_marker"] & ~df["has_bac_marker"]).sum()
    print(f"\n4. Videos with non-Bac markers only: {non_bac_only:,}")
    print(f"   → These should be EXCLUDED")
    
    print("\n" + "=" * 60)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    run_diagnostic()
