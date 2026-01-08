"""Analyze collected video data."""
import pandas as pd
from pathlib import Path

data_path = Path("data/raw/videos_metadata.csv")
df = pd.read_csv(data_path)

print("="*60)
print("COLLECTION ANALYSIS")
print("="*60)
print(f"\nTotal videos collected: {len(df)}")

print(f"\nVideos per channel:")
print(df['channel_title'].value_counts())

print(f"\nDate range:")
df['publish_date'] = pd.to_datetime(df['publish_date'])
print(f"Oldest: {df['publish_date'].min()}")
print(f"Newest: {df['publish_date'].max()}")

print(f"\nSample of data:")
print(df[['video_id', 'title', 'channel_title', 'view_count']].head())
