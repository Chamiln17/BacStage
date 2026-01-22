import pandas as pd
from pathlib import Path

files = [
    'data/processed/transcripts_checkpoint.csv',
    'data/processed/transcripts_part1_checkpoint.csv',
    'data/processed/transcripts_part2_checkpoint.csv'
]

total_unique_ids = set()
total_available_ids = set()

for f in files:
    p = Path(f)
    if not p.exists():
        print(f"File not found: {f}")
        continue
        
    try:
        df = pd.read_csv(p)
        total_rows = len(df)
        
        # Ensure only string IDs
        df['video_id'] = df['video_id'].astype(str)
        
        unique_ids = df['video_id'].nunique()
        available_df = df[df['transcript_available'] == True]
        available_count = len(available_df)
        available_unique = available_df['video_id'].nunique()
        
        print(f"--- {p.name} ---")
        print(f"Total Rows: {total_rows}")
        print(f"Unique Video IDs: {unique_ids}")
        print(f"Transcripts Available: {available_count} (Unique: {available_unique})")
        
        total_unique_ids.update(df['video_id'].tolist())
        total_available_ids.update(available_df['video_id'].tolist())
        
    except Exception as e:
        print(f"Error checking {f}: {e}")

print(f"\n=== COMBINED TOTALS ===", flush=True)
print(f"Total Processed Videos (Unique): {len(total_unique_ids)}", flush=True)
print(f"Total Successfully Collected Transcripts (Unique): {len(total_available_ids)}", flush=True)
