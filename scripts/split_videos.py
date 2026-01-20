import pandas as pd

# Load missing videos
missing_df = pd.read_csv('data/processed/videos_missing.csv')

print(f'📊 Total missing videos: {len(missing_df)}')

# Split into two equal parts
half = len(missing_df) // 2

part1 = missing_df.iloc[:half]
part2 = missing_df.iloc[half:]

# Save both parts
part1.to_csv('data/processed/videos_part1_yours.csv', index=False)
part2.to_csv('data/processed/videos_part2_friend.csv', index=False)

print(f'\n✅ Split complete!')
print(f'📁 Part 1 (YOURS): {len(part1)} videos → data/processed/videos_part1_yours.csv')
print(f'📁 Part 2 (FRIEND): {len(part2)} videos → data/processed/videos_part2_friend.csv')
print(f'\n🚀 Commands to run:')
print(f'\nYOUR PC:')
print(f'uv run python src/data/transcript_collector.py --input data/processed/videos_part1_yours.csv --output data/processed/transcripts_part1.csv --proxy "socks5h://127.0.0.1:9150" --workers 3')
print(f'\nFRIEND\'S PC:')
print(f'uv run python src/data/transcript_collector.py --input data/processed/videos_part2_friend.csv --output data/processed/transcripts_part2.csv --proxy "socks5h://127.0.0.1:9150" --workers 3')
print(f'\n💡 After both finish, merge with:')
print(f'python merge_results.py')
