"""
Integrate Transcript Features into Engineered Dataset

This script:
1. Extracts transcript features from merged transcripts
2. Merges them with the existing videos_engineered.csv
3. Saves the updated dataset

Usage:
    uv run python scripts/integrate_transcript_features.py
"""
import pandas as pd
import sys
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    print('=' * 60)
    print('📊 TRANSCRIPT FEATURE INTEGRATION')
    print('=' * 60)
    
    # Define paths
    transcripts_path = Path('data/processed/transcripts_merged.csv')
    videos_path = Path('data/processed/videos_bac_only.csv')
    engineered_path = Path('data/processed/videos_engineered.csv')
    transcript_features_output = Path('data/processed/transcript_features.csv')
    final_output = Path('data/processed/videos_engineered_with_transcripts.csv')
    
    # Step 1: Extract transcript features
    print('\n📝 Step 1: Extracting transcript features...')
    try:
        from src.features.transcript_features import extract_features_cli
        
        if not transcripts_path.exists():
            print(f'❌ Transcripts file not found: {transcripts_path}')
            return 1
        
        # Extract features (this will create transcript_features.csv)
        result = extract_features_cli(
            input_path=transcripts_path,
            output_path=transcript_features_output,
            videos_path=videos_path if videos_path.exists() else None
        )
        
        if result != 0:
            print('❌ Feature extraction failed')
            return 1
        
        print(f'   ✓ Transcript features saved to: {transcript_features_output}')
    
    except Exception as e:
        print(f'❌ Error during feature extraction: {e}')
        return 1
    
    # Step 2: Load all datasets
    print('\n📂 Step 2: Loading datasets...')
    try:
        # Load existing engineered features
        if not engineered_path.exists():
            print(f'❌ Engineered features file not found: {engineered_path}')
            return 1
        
        engineered_df = pd.read_csv(engineered_path)
        print(f'   ✓ Loaded engineered features: {len(engineered_df):,} rows, {len(engineered_df.columns)} columns')
        
        # Load transcript features
        transcript_feat_df = pd.read_csv(transcript_features_output)
        print(f'   ✓ Loaded transcript features: {len(transcript_feat_df):,} rows, {len(transcript_feat_df.columns)} columns')
    
    except Exception as e:
        print(f'❌ Error loading datasets: {e}')
        return 1
    
    # Step 3: Merge datasets
    print('\n🔗 Step 3: Merging transcript features with engineered features...')
    try:
        # Check video_id column exists
        if 'video_id' not in engineered_df.columns:
            print('❌ Column "video_id" not found in engineered features')
            return 1
        if 'video_id' not in transcript_feat_df.columns:
            print('❌ Column "video_id" not found in transcript features')
            return 1
        
        # Ensure video_id is string type for consistent merging
        engineered_df['video_id'] = engineered_df['video_id'].astype(str)
        transcript_feat_df['video_id'] = transcript_feat_df['video_id'].astype(str)
        
        # Perform left join (keep all videos from engineered dataset)
        merged_df = engineered_df.merge(
            transcript_feat_df,
            on='video_id',
            how='left',
            suffixes=('', '_transcript')
        )
        
        # Calculate statistics
        videos_with_transcripts = merged_df['transcript_word_count'].notna().sum()
        videos_without_transcripts = merged_df['transcript_word_count'].isna().sum()
        
        print(f'   ✓ Merge complete:')
        print(f'      Total videos: {len(merged_df):,}')
        print(f'      With transcripts: {videos_with_transcripts:,} ({videos_with_transcripts/len(merged_df)*100:.1f}%)')
        print(f'      Without transcripts: {videos_without_transcripts:,} ({videos_without_transcripts/len(merged_df)*100:.1f}%)')
        print(f'      Total columns: {len(merged_df.columns)}')
        
        # List new transcript features
        new_cols = [col for col in transcript_feat_df.columns if col != 'video_id']
        print(f'\n   📊 Added {len(new_cols)} transcript features:')
        for col in sorted(new_cols):
            non_null = merged_df[col].notna().sum()
            print(f'      - {col}: {non_null:,} non-null values')
    
    except Exception as e:
        print(f'❌ Error during merge: {e}')
        return 1
    
    # Step 4: Save merged dataset
    print(f'\n💾 Step 4: Saving merged dataset...')
    try:
        final_output.parent.mkdir(parents=True, exist_ok=True)
        merged_df.to_csv(final_output, index=False)
        print(f'   ✓ Saved to: {final_output}')
        
        # Also update the original videos_engineered.csv (create backup first)
        backup_path = engineered_path.parent / f'{engineered_path.stem}_backup.csv'
        print(f'\n   Creating backup of original file...')
        engineered_df.to_csv(backup_path, index=False)
        print(f'   ✓ Backup saved to: {backup_path}')
        
        # Overwrite original
        merged_df.to_csv(engineered_path, index=False)
        print(f'   ✓ Updated original file: {engineered_path}')
    
    except Exception as e:
        print(f'❌ Error saving files: {e}')
        return 1
    
    # Step 5: Summary
    print('\n' + '=' * 60)
    print('📊 INTEGRATION SUMMARY')
    print('=' * 60)
    print(f'\nOriginal engineered features: {len(engineered_df.columns)} columns')
    print(f'Added transcript features: {len(new_cols)} columns')
    print(f'Final dataset: {len(merged_df.columns)} columns')
    print(f'\nCoverage: {videos_with_transcripts}/{len(merged_df)} videos have transcripts ({videos_with_transcripts/len(merged_df)*100:.1f}%)')
    
    print('\n✅ Integration complete!')
    print(f'\n📁 Output files:')
    print(f'   - {final_output}')
    print(f'   - {engineered_path} (updated)')
    print(f'   - {backup_path} (backup)')
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
