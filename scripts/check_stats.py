import pandas as pd

df = pd.read_csv('data/processed/transcripts_checkpoint.csv')
failed = df[df['transcript_available'] == False]

print('\n📊 SKIP STATISTICS:\n')
print(f'Total videos: {len(df)}')
print(f'Success: {(df["transcript_available"] == True).sum()}')
print(f'Failed: {len(failed)}')
print(f'\nFailed breakdown:')
print(failed['failure_reason'].value_counts())

# Show retry queue potential
retry_eligible = failed[~failed['failure_reason'].str.contains('bot', case=False, na=False)]
print(f'\n🔄 Retry eligible (non-bot errors): {len(retry_eligible)}')
