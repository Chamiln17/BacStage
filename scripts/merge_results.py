"""
Merge multiple transcript collection results into a single dataset.
Handles errors gracefully and provides detailed statistics.
"""
import pandas as pd
from pathlib import Path
import sys


def load_transcript_file(filepath: str, part_name: str) -> pd.DataFrame:
    """Load a transcript CSV file with error handling."""
    try:
        path = Path(filepath)
        if not path.exists():
            print(f'⚠️  {part_name} not found: {filepath}')
            return None
        
        df = pd.read_csv(filepath)
        print(f'   ✓ {part_name}: {len(df):,} transcripts')
        return df
    
    except pd.errors.EmptyDataError:
        print(f'❌ {part_name} is empty: {filepath}')
        return None
    except pd.errors.ParserError as e:
        print(f'❌ Error parsing {part_name}: {e}')
        return None
    except Exception as e:
        print(f'❌ Unexpected error loading {part_name}: {e}')
        return None


def merge_transcripts():
    """Main function to merge transcript files."""
    print('=' * 60)
    print('📊 TRANSCRIPT MERGE UTILITY')
    print('=' * 60)
    
    # Define input files
    files_to_merge = [
        ('data/processed/transcripts_checkpoint.csv', 'Part 1 (Checkpoint)'),
        ('data/processed/transcripts_part1.csv', 'Part 2 (Part 1)'),
        ('data/processed/transcripts_part2.csv', 'Part 3 (Part 2)'),
    ]
    
    # Load all files
    print('\n� Loading transcript files...')
    dataframes = []
    for filepath, part_name in files_to_merge:
        df = load_transcript_file(filepath, part_name)
        if df is not None:
            dataframes.append(df)
    
    # Check if we have any data
    if not dataframes:
        print('\n❌ No transcript files found to merge!')
        print('   Please check that at least one input file exists.')
        return 1
    
    # Merge dataframes
    print(f'\n🔗 Merging {len(dataframes)} file(s)...')
    try:
        combined = pd.concat(dataframes, ignore_index=True)
        print(f'   ✓ Combined: {len(combined):,} total rows')
    except Exception as e:
        print(f'❌ Error during merge: {e}')
        return 1
    
    # Remove duplicates
    print('\n🧹 Removing duplicates...')
    try:
        if 'video_id' not in combined.columns:
            print('⚠️  Column "video_id" not found. Skipping deduplication.')
            deduplicated_count = 0
        else:
            initial_count = len(combined)
            combined = combined.drop_duplicates(subset=['video_id'], keep='first')
            deduplicated_count = initial_count - len(combined)
            print(f'   ✓ Removed {deduplicated_count:,} duplicate(s)')
            print(f'   ✓ Final count: {len(combined):,} unique transcripts')
    except Exception as e:
        print(f'⚠️  Error during deduplication: {e}')
        print('   Continuing without deduplication...')
    
    # Save merged file
    output_path = 'data/processed/transcripts_merged.csv'
    print(f'\n💾 Saving merged file...')
    try:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(output_path, index=False)
        print(f'   ✓ Saved to: {output_path}')
    except PermissionError:
        print(f'❌ Permission denied writing to: {output_path}')
        print('   Please close the file if it\'s open in Excel or another program.')
        return 1
    except Exception as e:
        print(f'❌ Error saving file: {e}')
        return 1
    
    # Display statistics
    print('\n' + '=' * 60)
    print('📊 STATISTICS')
    print('=' * 60)
    
    try:
        if 'transcript_available' in combined.columns:
            success = (combined['transcript_available'] == True).sum()
            failed = (combined['transcript_available'] == False).sum()
            total = len(combined)
            
            print(f'\nTotal transcripts: {total:,}')
            print(f'   ✅ Success: {success:,} ({success/total*100:.1f}%)')
            print(f'   ❌ Failed:  {failed:,} ({failed/total*100:.1f}%)')
            
            # Additional stats if language column exists
            if 'transcript_language' in combined.columns:
                print(f'\nLanguage distribution:')
                lang_counts = combined[combined['transcript_available'] == True]['transcript_language'].value_counts()
                for lang, count in lang_counts.head(5).items():
                    print(f'   {lang}: {count:,} ({count/success*100:.1f}%)')
            
            # Generated vs manual
            if 'is_generated' in combined.columns:
                generated = (combined['is_generated'] == True).sum()
                manual = (combined['is_generated'] == False).sum()
                print(f'\nCaption type:')
                print(f'   🤖 Auto-generated: {generated:,} ({generated/success*100:.1f}%)')
                print(f'   ✍️  Manual: {manual:,} ({manual/success*100:.1f}%)')
        else:
            print('⚠️  Column "transcript_available" not found.')
            print(f'   Total rows: {len(combined):,}')
    
    except Exception as e:
        print(f'⚠️  Error calculating statistics: {e}')
        print(f'   Total rows: {len(combined):,}')
    
    print('\n✅ Merge complete!')
    return 0


if __name__ == '__main__':
    sys.exit(merge_transcripts())
