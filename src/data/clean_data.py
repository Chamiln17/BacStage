#!/usr/bin/env python
"""
Data Cleaning Module
====================
Prepares raw data for feature engineering by handling missing values,
creating derived flags, and ensuring data type consistency.

Usage:
    uv run python run_pipeline.py clean
    
    # Or directly:
    uv run python src/data/clean_data.py

Outputs:
    data/cleaned/videos_cleaned.csv    - Cleaned video metadata
    data/cleaned/transcripts_clean.csv - Cleaned transcripts (valid only)
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def clean_videos(
    videos_df: pd.DataFrame,
    transcripts_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Clean video metadata DataFrame.
    
    Cleaning steps:
    1. Impute missing text fields (description, tags) with empty strings
    2. Add `has_transcript` flag based on transcript availability
    3. Impute transcript-derived features with 0 for missing transcripts
    4. Ensure correct data types
    
    Args:
        videos_df: Raw video metadata DataFrame
        transcripts_df: Optional transcripts DataFrame for has_transcript flag
        
    Returns:
        Cleaned DataFrame ready for feature engineering
    """
    logger.info("=" * 60)
    logger.info("CLEANING VIDEO METADATA")
    logger.info("=" * 60)
    
    df = videos_df.copy()
    initial_count = len(df)
    logger.info(f"Input: {initial_count:,} videos")
    
    # =========================================================================
    # 1. MISSING VALUE IMPUTATION
    # =========================================================================
    logger.info("\n[1/4] Imputing missing values...")
    
    # Text fields -> empty string
    text_fields = ['description', 'tags']
    for field in text_fields:
        if field in df.columns:
            missing = df[field].isna().sum()
            df[field] = df[field].fillna("")
            if missing > 0:
                logger.info(f"  - {field}: {missing:,} missing -> filled with ''")
    
    # =========================================================================
    # 2. HAS_TRANSCRIPT FLAG
    # =========================================================================
    logger.info("\n[2/4] Creating has_transcript flag...")
    
    if transcripts_df is not None and 'video_id' in df.columns:
        # Get set of video_ids with valid transcripts
        valid_transcript_ids = set(
            transcripts_df[
                (transcripts_df['transcript_available'] == True) &
                (transcripts_df['transcript_text'].notna()) &
                (transcripts_df['transcript_text'].str.strip() != "")
            ]['video_id'].unique()
        )
        df['has_transcript'] = df['video_id'].isin(valid_transcript_ids)
        with_transcript = df['has_transcript'].sum()
        logger.info(f"  - Videos with transcripts: {with_transcript:,} ({with_transcript/len(df)*100:.1f}%)")
    else:
        # If no transcript data, default to False
        df['has_transcript'] = False
        logger.info("  - No transcript data provided, defaulting to False")
    
    # =========================================================================
    # 3. TRANSCRIPT FEATURE IMPUTATION
    # =========================================================================
    logger.info("\n[3/4] Imputing transcript-derived features...")
    
    transcript_numeric_features = [
        'transcript_word_count',
        'transcript_char_count', 
        'transcript_sentence_count',
        'segment_count',
        'question_count',
        'example_count',
        'explanation_count',
        'contrast_count',
        'technical_term_count',
        'subject_keyword_count',
        'unique_word_count',
    ]
    
    transcript_ratio_features = [
        'lexical_diversity',
        'avg_words_per_sentence',
        'question_density',
        'example_density',
        'explanation_density',
        'contrast_density',
        'technical_term_density',
        'subject_keyword_density',
    ]
    
    for feature in transcript_numeric_features:
        if feature in df.columns:
            missing = df[feature].isna().sum()
            if missing > 0:
                df[feature] = df[feature].fillna(0)
                logger.info(f"  - {feature}: {missing:,} NaN -> 0")
    
    for feature in transcript_ratio_features:
        if feature in df.columns:
            missing = df[feature].isna().sum()
            if missing > 0:
                df[feature] = df[feature].fillna(0.0)
                logger.info(f"  - {feature}: {missing:,} NaN -> 0.0")
    
    # =========================================================================
    # 4. DATA TYPE VALIDATION
    # =========================================================================
    logger.info("\n[4/4] Validating data types...")
    
    # Ensure numeric columns are numeric
    numeric_cols = ['view_count', 'like_count', 'comment_count', 'duration_sec']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
    
    # Ensure datetime columns are datetime
    if 'publish_date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['publish_date']):
            df['publish_date'] = pd.to_datetime(df['publish_date'], format='ISO8601', utc=True)
            logger.info("  - publish_date: converted to datetime")
    
    logger.info(f"\n✅ Cleaning complete: {len(df):,} videos")
    
    return df


def clean_transcripts(
    input_path: Path,
    output_path: Path,
) -> pd.DataFrame:
    """
    Load transcript data and filter to keep only valid transcripts.
    
    Args:
        input_path: Path to the merged transcripts CSV.
        output_path: Path to save the cleaned transcripts.
        
    Returns:
        Cleaned DataFrame with only valid transcripts.
    """
    logger.info(f"Loading transcripts from: {input_path}")
    df = pd.read_csv(input_path)
    
    total_count = len(df)
    logger.info(f"Total rows loaded: {total_count}")
    
    # Filter: keep only rows where transcript_available is True
    # and transcript_text is not empty/null
    mask_valid = (
        (df["transcript_available"] == True) & 
        (df["transcript_text"].notna()) &
        (df["transcript_text"].str.strip() != "")
    )
    
    # improved: Check for content corruption (JS/HTML artifacts)
    # Some transcripts are just YouTube internal code dumps
    corruption_markers = [
        'window.', 'ytcfg', 'u003d', 'u0026', 'javascript', 
        'var ', 'function(', '<html', 'innertubeapi'
    ]
    
    def is_corrupted(text):
        if not isinstance(text, str): return False
        text_lower = text.lower()
        score = sum(1 for m in corruption_markers if m in text_lower)
        return score >= 2
        
    mask_clean = mask_valid & ~df["transcript_text"].apply(is_corrupted)
    
    df_clean = df[mask_clean].copy()
    
    corrupted_count = mask_valid.sum() - mask_clean.sum()
    if corrupted_count > 0:
        logger.info(f"Removed {corrupted_count} corrupted transcripts (JS/JSON artifacts)")
    
    clean_count = len(df_clean)
    removed_count = total_count - clean_count
    
    logger.info(f"Removed {removed_count} invalid transcripts ({removed_count/total_count*100:.1f}%)")
    logger.info(f"Keeping {clean_count} valid transcripts ({clean_count/total_count*100:.1f}%)")
    
    # Drop the failure_reason column since all are valid now
    if "failure_reason" in df_clean.columns:
        df_clean = df_clean.drop(columns=["failure_reason"])
    
    # Impute transcript_text for the full dataset (useful for later merging)
    df['transcript_text'] = df['transcript_text'].fillna("")
    
    # Save to output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_clean.to_csv(output_path, index=False)
    logger.info(f"Saved cleaned transcripts to: {output_path}")
    
    # Print language distribution
    if "transcript_language" in df_clean.columns:
        lang_dist = df_clean["transcript_language"].value_counts()
        logger.info(f"\nLanguage distribution:\n{lang_dist.to_string()}")
    
    return df_clean


def clean_pipeline(
    videos_path: Path,
    transcripts_path: Path,
    output_dir: Path,
) -> int:
    """
    Run the full cleaning pipeline.
    
    Args:
        videos_path: Path to videos CSV
        transcripts_path: Path to transcripts CSV
        output_dir: Directory to save cleaned outputs
        
    Returns:
        Exit code (0 for success)
    """
    logger.info("=" * 60)
    logger.info("DATA CLEANING PIPELINE")
    logger.info("=" * 60)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    if not videos_path.exists():
        logger.error(f"Videos file not found: {videos_path}")
        return 1
    
    videos_df = pd.read_csv(videos_path)
    logger.info(f"Loaded {len(videos_df):,} videos from {videos_path.name}")
    
    transcripts_df = None
    if transcripts_path.exists():
        transcripts_df = pd.read_csv(transcripts_path)
        logger.info(f"Loaded {len(transcripts_df):,} transcripts from {transcripts_path.name}")
    else:
        logger.warning(f"Transcripts file not found: {transcripts_path}")
    
    # Clean videos
    videos_cleaned = clean_videos(videos_df, transcripts_df)
    videos_output = output_dir / "videos_cleaned.csv"
    videos_cleaned.to_csv(videos_output, index=False)
    logger.info(f"\nSaved cleaned videos to: {videos_output}")
    
    # Clean transcripts (valid only)
    if transcripts_df is not None:
        transcripts_output = output_dir / "transcripts_clean.csv"
        clean_transcripts(transcripts_path, transcripts_output)
    
    logger.info("\n" + "=" * 60)
    logger.info("✅ CLEANING COMPLETE")
    logger.info("=" * 60)
    
    return 0


def main():
    """Main entry point."""
    project_root = Path(__file__).parent.parent.parent
    
    videos_path = project_root / "data" / "processed" / "videos_bac_only.csv"
    transcripts_path = project_root / "data" / "processed" / "transcripts_merged.csv"
    output_dir = project_root / "data" / "cleaned"
    
    return clean_pipeline(videos_path, transcripts_path, output_dir)


if __name__ == "__main__":
    import sys
    sys.exit(main())
