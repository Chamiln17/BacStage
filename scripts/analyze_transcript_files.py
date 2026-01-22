"""
Analyze transcript checkpoint files for size, duplicates, and corruption issues.
"""

import pandas as pd
from pathlib import Path

def analyze_file(filepath: Path, label: str):
    """Analyze a transcript checkpoint file."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {label}")
    print(f"Path: {filepath}")
    print(f"{'='*70}")
    
    try:
        # File size
        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"\n📊 File Size: {size_mb:.2f} MB")
        
        # Load CSV
        df = pd.read_csv(filepath)
        total_rows = len(df)
        print(f"📊 Total Rows: {total_rows}")
        
        # Check for duplicates
        if 'video_id' in df.columns:
            unique_video_ids = df['video_id'].nunique()
            duplicate_count = total_rows - unique_video_ids
            print(f"\n🔍 Duplicate Analysis:")
            print(f"  - Unique Video IDs: {unique_video_ids}")
            print(f"  - Duplicate Rows: {duplicate_count}")
            
            if duplicate_count > 0:
                duplicate_percentage = (duplicate_count / total_rows) * 100
                print(f"  - Duplicate Percentage: {duplicate_percentage:.2f}%")
                
                # Find which video IDs are duplicated
                duplicated_ids = df[df.duplicated(subset=['video_id'], keep=False)]
                if len(duplicated_ids) > 0:
                    print(f"\n  Top 10 Duplicated Video IDs:")
                    dup_counts = duplicated_ids.groupby('video_id').size().sort_values(ascending=False).head(10)
                    for vid, count in dup_counts.items():
                        print(f"    - {vid}: {count} occurrences")
        
        # Column info
        print(f"\n📋 Columns ({len(df.columns)}):")
        for col in df.columns:
            null_count = df[col].isnull().sum()
            null_pct = (null_count / total_rows) * 100
            print(f"  - {col}: {null_count} nulls ({null_pct:.1f}%)")
        
        # Transcript availability
        if 'transcript_available' in df.columns:
            available_count = df['transcript_available'].sum() if df['transcript_available'].dtype == bool else (df['transcript_available'] == True).sum()
            print(f"\n✓ Transcripts Available: {available_count} / {total_rows} ({available_count/total_rows*100:.1f}%)")
        elif 'transcript_text' in df.columns:
            available_count = df['transcript_text'].notna().sum()
            print(f"\n✓ Transcripts Available (from text): {available_count} / {total_rows} ({available_count/total_rows*100:.1f}%)")
        
        # Check for HTML/JS corruption
        if 'transcript_text' in df.columns:
            print(f"\n🔍 Content Validation:")
            html_indicators = ['<!DOCTYPE', '<html', 'window.', 'ytcfg', 'var ', 'function(']
            
            corrupted_count = 0
            for indicator in html_indicators:
                matches = df['transcript_text'].fillna('').str.contains(indicator, case=False, regex=False).sum()
                if matches > 0:
                    print(f"  - Contains '{indicator}': {matches} rows")
                    corrupted_count = max(corrupted_count, matches)
            
            if corrupted_count == 0:
                print(f"  ✓ No HTML/JS corruption detected")
            else:
                print(f"  ⚠️  Potential corruption in ~{corrupted_count} rows")
        
        # Failure reasons
        if 'failure_reason' in df.columns:
            failures = df[df['failure_reason'].notna()]
            if len(failures) > 0:
                print(f"\n❌ Failure Reasons Distribution:")
                failure_counts = failures['failure_reason'].value_counts().head(10)
                for reason, count in failure_counts.items():
                    print(f"  - {reason}: {count}")
        
        # Average transcript length
        if 'transcript_text' in df.columns:
            valid_transcripts = df[df['transcript_text'].notna()]['transcript_text']
            if len(valid_transcripts) > 0:
                avg_length = valid_transcripts.str.len().mean()
                median_length = valid_transcripts.str.len().median()
                max_length = valid_transcripts.str.len().max()
                print(f"\n📝 Transcript Length Stats:")
                print(f"  - Average: {avg_length:.0f} chars")
                print(f"  - Median: {median_length:.0f} chars")
                print(f"  - Max: {max_length:.0f} chars")
        
        return df
        
    except Exception as e:
        print(f"❌ Error analyzing file: {e}")
        return None

def compare_files(df1, df2, label1, label2):
    """Compare two dataframes."""
    print(f"\n{'='*70}")
    print(f"Comparing: {label1} vs {label2}")
    print(f"{'='*70}")
    
    if df1 is None or df2 is None:
        print("Cannot compare - one or both files failed to load")
        return
    
    # Common video IDs
    ids1 = set(df1['video_id'].unique())
    ids2 = set(df2['video_id'].unique())
    
    common = ids1 & ids2
    only_in_1 = ids1 - ids2
    only_in_2 = ids2 - ids1
    
    print(f"\n🔄 Video ID Overlap:")
    print(f"  - Common IDs: {len(common)}")
    print(f"  - Only in {label1}: {len(only_in_1)}")
    print(f"  - Only in {label2}: {len(only_in_2)}")
    
    if len(common) > 0:
        print(f"  - Overlap percentage: {len(common)/min(len(ids1), len(ids2))*100:.1f}%")

def main():
    """Main analysis function."""
    base_path = Path("e:/programming/SIC/data/processed")
    
    part1_path = base_path / "transcripts_part1_checkpoint.csv"
    part2_path = base_path / "transcripts_part2_checkpoint.csv"
    
    # Analyze both files
    df_part1 = analyze_file(part1_path, "Part 1 Checkpoint")
    df_part2 = analyze_file(part2_path, "Part 2 Checkpoint")
    
    # Compare
    if df_part1 is not None and df_part2 is not None:
        compare_files(df_part1, df_part2, "Part 1", "Part 2")
    
    # Summary recommendation
    print(f"\n{'='*70}")
    print("🎯 RECOMMENDATIONS")
    print(f"{'='*70}")
    
    if df_part2 is not None:
        unique_ids = df_part2['video_id'].nunique()
        total_rows = len(df_part2)
        duplicate_count = total_rows - unique_ids
        
        if duplicate_count > 0:
            print(f"\n⚠️  Part 2 contains {duplicate_count} duplicate rows!")
            print(f"   Current size: {part2_path.stat().st_size / (1024*1024):.2f} MB with {total_rows} rows")
            print(f"   After deduplication: Would have {unique_ids} rows")
            estimated_size = (part2_path.stat().st_size / total_rows) * unique_ids / (1024*1024)
            print(f"   Estimated size after dedup: ~{estimated_size:.2f} MB")
            print(f"\n   💡 Run deduplication to save ~{(part2_path.stat().st_size / (1024*1024)) - estimated_size:.2f} MB")

if __name__ == "__main__":
    main()
