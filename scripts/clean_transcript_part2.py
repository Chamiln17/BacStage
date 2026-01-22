"""
Clean transcripts_part2_checkpoint.csv by removing HTML/JS corrupted entries.
Inspired by how transcripts_part1_checkpoint.csv was previously cleaned.
"""

import pandas as pd
from pathlib import Path
import shutil
from datetime import datetime

def is_corrupted_transcript(text):
    """
    Check if transcript text contains HTML/JS corruption indicators.
    Based on the validation logic in transcript_collector.py
    """
    if pd.isna(text):
        return False
    
    text_str = str(text)
    
    # Check for HTML/JS indicators (same as in transcript_collector.py)
    corruption_indicators = [
        '<!DOCTYPE',
        '<html',
        'window.',
        'ytcfg',
        'var ',
        'function(',
        'WIZ_global_data'
    ]
    
    for indicator in corruption_indicators:
        if indicator in text_str:
            return True
    
    return False

def clean_transcript_file(input_path: Path, output_path: Path, backup: bool = True):
    """
    Clean transcript checkpoint file by removing corrupted entries.
    
    Args:
        input_path: Path to corrupted CSV file
        output_path: Path for cleaned output
        backup: Whether to create backup of original file
    """
    print(f"{'='*70}")
    print(f"Cleaning Transcript Checkpoint File")
    print(f"{'='*70}")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    # Create backup
    if backup:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = input_path.parent / f"{input_path.stem}_backup_{timestamp}.csv"
        shutil.copy2(input_path, backup_path)
        print(f"✓ Backup created: {backup_path}")
    
    # Load data
    df = pd.read_csv(input_path)
    original_count = len(df)
    original_size_mb = input_path.stat().st_size / (1024 * 1024)
    
    print(f"\n📊 Original Stats:")
    print(f"  - Rows: {original_count}")
    print(f"  - Size: {original_size_mb:.2f} MB")
    
    # Identify corrupted rows
    if 'transcript_text' in df.columns:
        corrupted_mask = df['transcript_text'].apply(is_corrupted_transcript)
        corrupted_count = corrupted_mask.sum()
        
        print(f"\n🔍 Corruption Analysis:")
        print(f"  - Corrupted rows: {corrupted_count}")
        print(f"  - Clean rows: {original_count - corrupted_count}")
        
        if corrupted_count > 0:
            # Show sample of corrupted video IDs
            print(f"\n  Sample corrupted video IDs:")
            corrupted_ids = df[corrupted_mask]['video_id'].head(10)
            for vid in corrupted_ids:
                print(f"    - {vid}")
            
            # Remove corrupted rows
            df_clean = df[~corrupted_mask].copy()
            
            # For corrupted rows, keep them but mark as failed and clear transcript
            df_corrupted_fixed = df[corrupted_mask].copy()
            df_corrupted_fixed['transcript_text'] = None
            df_corrupted_fixed['transcript_available'] = False
            df_corrupted_fixed['failure_reason'] = 'Youtube returned HTML/JS instead of transcript'
            df_corrupted_fixed['is_generated'] = None
            df_corrupted_fixed['is_translatable'] = None
            df_corrupted_fixed['transcript_language'] = None
            df_corrupted_fixed['transcript_language_code'] = None
            df_corrupted_fixed['segment_count'] = 0
            
            # Combine clean rows with fixed corrupted rows
            df_final = pd.concat([df_clean, df_corrupted_fixed], ignore_index=True)
            
            # Sort by video_id to maintain consistency
            df_final = df_final.sort_values('video_id').reset_index(drop=True)
            
        else:
            print("  ✓ No corruption detected")
            df_final = df.copy()
    else:
        print("⚠️  No transcript_text column found")
        df_final = df.copy()
    
    # Remove duplicates (just in case)
    if 'video_id' in df_final.columns:
        duplicates_before = len(df_final)
        df_final = df_final.drop_duplicates(subset=['video_id'], keep='first')
        duplicates_removed = duplicates_before - len(df_final)
        if duplicates_removed > 0:
            print(f"\n🔄 Removed {duplicates_removed} duplicate rows")
    
    # Save cleaned file
    df_final.to_csv(output_path, index=False)
    final_count = len(df_final)
    final_size_mb = output_path.stat().st_size / (1024 * 1024)
    
    print(f"\n✅ Cleaned Stats:")
    print(f"  - Rows: {final_count}")
    print(f"  - Size: {final_size_mb:.2f} MB")
    print(f"  - Rows removed: {original_count - final_count}")
    print(f"  - Size reduced: {original_size_mb - final_size_mb:.2f} MB ({(1 - final_size_mb/original_size_mb)*100:.1f}%)")
    
    # Verify cleaning
    if 'transcript_text' in df_final.columns:
        still_corrupted = df_final['transcript_text'].apply(is_corrupted_transcript).sum()
        if still_corrupted > 0:
            print(f"\n⚠️  Warning: {still_corrupted} rows still appear corrupted")
        else:
            print(f"\n✓ All HTML/JS corruption removed from transcript_text")
    
    print(f"\n✓ Cleaned file saved to: {output_path}")
    
    return df_final

def main():
    """Main entry point."""
    base_path = Path("e:/programming/SIC/data/processed")
    
    input_file = base_path / "transcripts_part2_checkpoint.csv"
    output_file = base_path / "transcripts_part2_checkpoint_cleaned.csv"
    
    if not input_file.exists():
        print(f"❌ Input file not found: {input_file}")
        return 1
    
    df_clean = clean_transcript_file(input_file, output_file, backup=True)
    
    print(f"\n{'='*70}")
    print("🎯 NEXT STEPS")
    print(f"{'='*70}")
    print("1. Review the cleaned file to verify results")
    print("2. If satisfied, replace the original:")
    print(f"   Move: {output_file}")
    print(f"   To:   {input_file}")
    print("3. Resume transcript collection with clean checkpoint")
    
    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
