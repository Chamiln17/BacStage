#!/usr/bin/env python
"""
TF-IDF Keyword Discovery: Automatically discover Bac-associated terms.

This script uses TF-IDF to discover high-signal "Bac-ish" terms from video titles,
using pseudo-labels derived from explicit grade markers.

Output: data/processed/tfidf_bac_terms.json

Usage:
    uv run python scripts/03_tfidf_keyword_discovery.py
    uv run python scripts/03_tfidf_keyword_discovery.py --top-n 50 --min-df 10
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
        "tfidf": {
            "top_n_terms": 100,
            "min_df": 5,
            "max_df": 0.8,
            "ngram_range": [1, 2],
            "analyzer": "char_wb",
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


def discover_tfidf_terms(
    input_path: Path,
    output_path: Path,
    bac_markers: list,
    non_bac_markers: list,
    top_n: int = 100,
    min_df: int = 5,
    max_df: float = 0.8,
    ngram_range: tuple = (1, 2),
    analyzer: str = "char_wb",
) -> list:
    """
    Discover Bac-associated terms using TF-IDF.
    
    Args:
        input_path: Path to videos_metadata.csv
        output_path: Path to save tfidf_bac_terms.json
        bac_markers: List of Bac markers for pseudo-labeling
        non_bac_markers: List of non-Bac markers for pseudo-labeling
        top_n: Number of top terms to extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency
        ngram_range: N-gram range for vectorizer
        analyzer: Analyzer type for vectorizer
    
    Returns:
        List of discovered Bac-associated terms
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    print("=" * 60)
    print("TF-IDF KEYWORD DISCOVERY")
    print("=" * 60)
    
    print(f"\nInput: {input_path}")
    print(f"Output: {output_path}")
    print(f"Top N terms: {top_n}")
    print(f"Min DF: {min_df}, Max DF: {max_df}")
    print(f"N-gram range: {ngram_range}")
    print(f"Analyzer: {analyzer}")
    
    # Load data
    print("\nLoading video metadata...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df):,} records")
    
    # Deduplicate to latest snapshot
    if "snapshot_date" in df.columns:
        df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], format="ISO8601", errors="coerce")
        df = df.sort_values("snapshot_date", ascending=False)
    
    if "video_id" in df.columns:
        original = len(df)
        df = df.drop_duplicates(subset=["video_id"], keep="first")
        print(f"Deduplicated: {original:,} → {len(df):,} unique videos")
    
    # Combine text for marker detection
    text = (
        df["title"].fillna("").astype(str) + " "
        + df["description"].fillna("").astype(str) + " "
        + df["tags"].fillna("").astype(str)
    ).str.lower()
    
    # Create pseudo-labels (high-precision seeds)
    has_bac = has_markers(text, bac_markers)
    has_non_bac = has_markers(text, non_bac_markers)
    
    # High-precision sets: clear Bac OR clear non-Bac
    bac_mask = has_bac & ~has_non_bac
    non_bac_mask = has_non_bac & ~has_bac
    
    bac_titles = df.loc[bac_mask, "title"].fillna("").astype(str).tolist()
    non_bac_titles = df.loc[non_bac_mask, "title"].fillna("").astype(str).tolist()
    
    print(f"\nPseudo-labeled sets:")
    print(f"  Bac (clean):     {len(bac_titles):,} titles")
    print(f"  Non-Bac (clean): {len(non_bac_titles):,} titles")
    
    if len(bac_titles) < 100 or len(non_bac_titles) < 50:
        print("\nWarning: Small pseudo-labeled sets may produce unreliable results")
    
    # Combine for TF-IDF
    all_titles = bac_titles + non_bac_titles
    labels = [1] * len(bac_titles) + [0] * len(non_bac_titles)
    
    if len(all_titles) == 0:
        print("Error: No titles found for TF-IDF")
        return []
    
    # Fit TF-IDF vectorizer
    print(f"\nFitting TF-IDF vectorizer on {len(all_titles):,} titles...")
    
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(ngram_range),
        analyzer=analyzer,
        min_df=min_df,
        max_df=max_df,
    )
    
    try:
        X = vectorizer.fit_transform(all_titles)
        feature_names = vectorizer.get_feature_names_out()
        print(f"Extracted {len(feature_names):,} features")
    except ValueError as e:
        print(f"Error fitting vectorizer: {e}")
        return []
    
    # Compute discriminative scores: mean(Bac) - mean(non-Bac)
    print("\nComputing discriminative scores...")
    
    n_bac = len(bac_titles)
    bac_tfidf = X[:n_bac].mean(axis=0).A1
    non_bac_tfidf = X[n_bac:].mean(axis=0).A1
    
    diff = bac_tfidf - non_bac_tfidf
    
    # Get top N Bac-associated terms
    top_idx = np.argsort(diff)[::-1][:top_n]
    top_terms = [(feature_names[i], float(diff[i])) for i in top_idx]
    
    # Filter out terms that are just noise
    # Keep terms with positive discriminative score
    top_terms = [(t, s) for t, s in top_terms if s > 0]
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    term_list = [t for t, s in top_terms]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(term_list, f, ensure_ascii=False, indent=2)
    
    print(f"\nSaved {len(term_list)} terms to: {output_path}")
    
    # Print top terms for manual review
    print("\n" + "=" * 60)
    print(f"TOP {min(50, len(top_terms))} BAC-ASSOCIATED TERMS (for manual review)")
    print("=" * 60)
    print("\nThese terms are more common in Bac-marked titles than non-Bac titles.")
    print("Review and remove any obvious junk terms from the JSON file.\n")
    
    for i, (term, score) in enumerate(top_terms[:50], 1):
        # Clean up display
        term_display = term.replace("\n", "\\n").replace("\t", "\\t")
        print(f"  {i:3d}. {term_display:30s} (score: {score:.4f})")
    
    if len(top_terms) > 50:
        print(f"\n  ... and {len(top_terms) - 50} more terms in {output_path}")
    
    # Also show bottom terms (most non-Bac associated) for reference
    print("\n" + "=" * 60)
    print("TOP 20 NON-BAC-ASSOCIATED TERMS (for reference)")
    print("=" * 60)
    print("\nThese terms are more common in non-Bac titles.\n")
    
    bottom_idx = np.argsort(diff)[:20]
    for i, idx in enumerate(bottom_idx, 1):
        term = feature_names[idx]
        score = diff[idx]
        term_display = term.replace("\n", "\\n").replace("\t", "\\t")
        print(f"  {i:3d}. {term_display:30s} (score: {score:.4f})")
    
    return term_list


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Discover Bac-associated terms using TF-IDF"
    )
    parser.add_argument(
        "--input", type=Path, default=None,
        help="Input video metadata CSV"
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output TF-IDF terms JSON"
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Number of top terms to extract (default: 100)"
    )
    parser.add_argument(
        "--min-df", type=int, default=None,
        help="Minimum document frequency (default: 5)"
    )
    parser.add_argument(
        "--max-df", type=float, default=None,
        help="Maximum document frequency (default: 0.8)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config()
    tfidf_config = config.get("tfidf", {})
    
    # Resolve paths
    input_path = args.input or PROJECT_ROOT / config["paths"]["input"]
    output_path = args.output or PROJECT_ROOT / config["paths"]["output_dir"] / "tfidf_bac_terms.json"
    
    # Resolve parameters
    top_n = args.top_n or tfidf_config.get("top_n_terms", 100)
    min_df = args.min_df or tfidf_config.get("min_df", 5)
    max_df = args.max_df or tfidf_config.get("max_df", 0.8)
    ngram_range = tfidf_config.get("ngram_range", [1, 2])
    analyzer = tfidf_config.get("analyzer", "char_wb")
    
    # Validate input
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1
    
    # Discover terms
    try:
        discover_tfidf_terms(
            input_path=input_path,
            output_path=output_path,
            bac_markers=config["markers"]["bac"],
            non_bac_markers=config["markers"]["non_bac"],
            top_n=top_n,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            analyzer=analyzer,
        )
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
