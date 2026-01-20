import pandas as pd

# Load input and checkpoint
videos_df = pd.read_csv('data/processed/videos_bac_only.csv')
checkpoint_df = pd.read_csv('data/processed/transcripts_checkpoint.csv')

# Get processed video IDs
processed_ids = set(checkpoint_df['video_id'].unique())

# Get all video IDs
all_ids = set(videos_df['video_id'].unique())

# Find missing
missing_ids = all_ids - processed_ids

print(f'📊 GAP ANALYSIS:\n')
print(f'Total videos in input: {len(all_ids)}')
print(f'Processed (in checkpoint): {len(processed_ids)}')
print(f'Missing (gaps): {len(missing_ids)}')
print(f'\n✅ Success: {(checkpoint_df["transcript_available"] == True).sum()}')
print(f'❌ Failed: {(checkpoint_df["transcript_available"] == False).sum()}')
print(f'🔴 Never tried: {len(missing_ids)}')

# Save missing IDs to file
if missing_ids:
    missing_df = videos_df[videos_df['video_id'].isin(missing_ids)]
    missing_df.to_csv('data/processed/videos_missing.csv', index=False)
    print(f'\n💾 Missing videos saved to: data/processed/videos_missing.csv')
    print(f'   You can process these separately!')
