import pandas as pd

# Load both parts
part1 = pd.read_csv('data/processed/transcripts_part1.csv')
part2 = pd.read_csv('data/processed/transcripts_part2.csv')

print(f'📊 Merging results:')
print(f'   Part 1: {len(part1)} transcripts')
print(f'   Part 2: {len(part2)} transcripts')

# Merge
combined = pd.concat([part1, part2], ignore_index=True)

# Remove duplicates (just in case)
combined = combined.drop_duplicates(subset=['video_id'], keep='first')

# Save
combined.to_csv('data/processed/transcripts_merged.csv', index=False)

print(f'\n✅ Merged: {len(combined)} total transcripts')
print(f'📁 Saved to: data/processed/transcripts_merged.csv')

# Stats
success = (combined['transcript_available'] == True).sum()
failed = (combined['transcript_available'] == False).sum()
print(f'\n📊 Results:')
print(f'   Success: {success} ({success/len(combined)*100:.1f}%)')
print(f'   Failed: {failed} ({failed/len(combined)*100:.1f}%)')
