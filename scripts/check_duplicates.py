import pandas as pd
import sys
from pathlib import Path

def check_file(path_str):
    p = Path(path_str)
    if not p.exists():
        print(f"File not found: {p}")
        return
    
    print(f"--- Checking {p.name} ---")
    try:
        df = pd.read_csv(p)
        total = len(df)
        unique = df['video_id'].nunique()
        dupes = total - unique
        print(f"Total rows: {total}")
        print(f"Unique IDs: {unique}")
        print(f"Duplicates: {dupes}")
        if dupes > 0:
            print("Top duplicates:")
            print(df['video_id'].value_counts().head())
    except Exception as e:
        print(f"Error reading {p}: {e}")

if __name__ == "__main__":
    check_file(r'e:\programming\SIC\data\processed\transcripts_part1_checkpoint.csv')
    # Check input file if we can guess it or passed as arg
    if len(sys.argv) > 1:
        check_file(sys.argv[1])
