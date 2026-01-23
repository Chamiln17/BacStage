"""
Deduplicate transcript files and regenerate the merged file.

This script:
1. Creates backups of all source files
2. Removes duplicates from transcripts_part1.csv and transcripts_checkpoint.csv
3. Regenerates transcripts_merged.csv from clean source files
4. Validates the final output

Usage:
    python scripts/deduplicate_transcripts.py
"""

import pandas as pd
from pathlib import Path
from datetime import datetime
import shutil

def create_backup(file_path: Path) -> Path:
    """Create timestamped backup of a file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = file_path.parent / f"{file_path.stem}_backup_{timestamp}{file_path.suffix}"
    shutil.copy2(file_path, backup_path)
    return backup_path

def deduplicate_file(file_path: Path, subset=['video_id'], keep='first') -> pd.DataFrame:
    """
    Remove duplicates from a CSV file.
    
    Args:
        file_path: Path to CSV file
        subset: Columns to identify duplicates
        keep: Which duplicate to keep ('first', 'last', False)
    
    Returns:
        Deduplicated DataFrame
    """
    df = pd.read_csv(file_path)
    original_count = len(df)
    
    df_clean = df.drop_duplicates(subset=subset, keep=keep)
    duplicates_removed = original_count - len(df_clean)
    
    return df_clean, original_count, duplicates_removed

def merge_transcript_files(part1_path: Path, part2_path: Path, checkpoint_path: Path) -> pd.DataFrame:
    """
    Merge transcript files from multiple sources.
    
    Args:
        part1_path: Path to transcripts_part1.csv
        part2_path: Path to transcripts_part2.csv
        checkpoint_path: Path to transcripts_checkpoint.csv
    
    Returns:
        Merged and deduplicated DataFrame
    """
    dfs = []
    
    # Load all files that exist
    for path, name in [(part1_path, 'part1'), (part2_path, 'part2'), (checkpoint_path, 'checkpoint')]:
        if path.exists():
            df = pd.read_csv(path)
            print(f"  Loaded {name}: {len(df):,} rows, {df['video_id'].nunique():,} unique IDs")
            dfs.append(df)
        else:
            print(f"  ⚠️  Skipping {name}: File not found")
    
    if not dfs:
        raise ValueError("No transcript files found to merge")
    
    # Concatenate all dataframes
    df_merged = pd.concat(dfs, ignore_index=True)
    print(f"\n  Combined total: {len(df_merged):,} rows")
    
    # Remove duplicates (keep first occurrence)
    df_merged = df_merged.drop_duplicates(subset=['video_id'], keep='last')
    print(f"  After deduplication: {len(df_merged):,} unique video IDs")
    
    # Sort by video_id for consistency
    df_merged = df_merged.sort_values('video_id').reset_index(drop=True)
    
    return df_merged

def main():
    """Main execution function."""
    print("=" * 80)
    print("🔧 TRANSCRIPT FILES DEDUPLICATION")
    print("=" * 80)
    
    # Define paths
    base_path = Path("e:/programming/SIC/data/processed")
    
    files_to_clean = {
        'part1': base_path / "transcripts_part1.csv",
        'part2': base_path / "transcripts_part2.csv",
        'checkpoint': base_path / "transcripts_checkpoint.csv"
    }
    
    merged_path = base_path / "transcripts_merged.csv"
    
    # Step 1: Create backups
    print("\n📦 Step 1: Creating Backups")
    print("-" * 80)
    for name, path in files_to_clean.items():
        if path.exists():
            backup_path = create_backup(path)
            print(f"✓ Backed up {name}: {backup_path.name}")
        else:
            print(f"⚠️  Skipping {name}: File not found")
    
    # Also backup merged file if it exists
    if merged_path.exists():
        backup_path = create_backup(merged_path)
        print(f"✓ Backed up merged: {backup_path.name}")
    
    # Step 2: Deduplicate source files
    print("\n🧹 Step 2: Deduplicating Source Files")
    print("-" * 80)
    
    cleaned_files = {}
    for name, path in files_to_clean.items():
        if not path.exists():
            print(f"⚠️  Skipping {name}: File not found")
            continue
        
        print(f"\nProcessing {name}...")
        df_clean, original, removed = deduplicate_file(path)
        
        # Save cleaned file
        df_clean.to_csv(path, index=False)
        cleaned_files[name] = path
        
        print(f"  Original: {original:,} rows")
        print(f"  Cleaned: {len(df_clean):,} rows")
        print(f"  Removed: {removed:,} duplicates")
        print(f"  ✓ Saved to: {path.name}")
    
    # Step 3: Regenerate merged file
    print("\n🔗 Step 3: Regenerating Merged File")
    print("-" * 80)
    
    df_merged = merge_transcript_files(
        files_to_clean['part1'],
        files_to_clean['part2'],
        files_to_clean['checkpoint']
    )
    
    # Save merged file
    df_merged.to_csv(merged_path, index=False)
    print(f"\n✓ Saved merged file: {merged_path.name}")
    
    # Step 4: Validation
    print("\n✅ Step 4: Validation")
    print("-" * 80)
    
    # Check for duplicates in merged file
    duplicates_in_merged = len(df_merged) - df_merged['video_id'].nunique()
    
    print(f"Merged file stats:")
    print(f"  Total rows: {len(df_merged):,}")
    print(f"  Unique video_ids: {df_merged['video_id'].nunique():,}")
    print(f"  Duplicates: {duplicates_in_merged:,}")
    
    if duplicates_in_merged == 0:
        print("\n🎉 SUCCESS: No duplicates in merged file!")
    else:
        print(f"\n⚠️  WARNING: {duplicates_in_merged} duplicates still present in merged file")
    
    # Compare with expected total
    videos_path = base_path / "videos_bac_only.csv"
    if videos_path.exists():
        df_videos = pd.read_csv(videos_path)
        expected = len(df_videos)
        coverage = (df_merged['video_id'].nunique() / expected) * 100
        print(f"\nCoverage analysis:")
        print(f"  Total videos: {expected:,}")
        print(f"  Transcripts collected: {df_merged['video_id'].nunique():,}")
        print(f"  Coverage: {coverage:.1f}%")
    
    print("\n" + "=" * 80)
    print("✅ DEDUPLICATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Review the cleaned files to verify results")
    print("2. Run your data exploration notebook with clean data")
    print("3. If needed, backups are in data/processed/ with _backup_ suffix")

if __name__ == "__main__":
    main()
