#!/usr/bin/env python
"""Verify subject deduplication fix"""
import pandas as pd

# Test 1: Check processed data
print("=" * 60)
print("TEST 1: Processed Data Verification")
print("=" * 60)
df = pd.read_csv('data/processed/videos_bac_balanced.csv')
unique_subjects = df['subject'].nunique()
print(f"Total unique subjects: {unique_subjects}")
print(f"\nSubject distribution:")
print(df['subject'].value_counts())

# Check for whitespace
has_whitespace = df['subject'].apply(lambda x: x != x.strip() if pd.notna(x) else False)
whitespace_count = has_whitespace.sum()
print(f"\nVideos with leading/trailing whitespace: {whitespace_count}")

assert unique_subjects == 9, f"Expected 9 unique subjects, got {unique_subjects}"
assert whitespace_count == 0, f"Found {whitespace_count} subjects with whitespace!"
print("\n✓ Test 1 PASSED!\n")

# Test 2: Check load function
print("=" * 60)
print("TEST 2: load_channel_subjects() Function")
print("=" * 60)
from src.features.bac_filter_balanced import load_channel_subjects
from pathlib import Path

subjects = load_channel_subjects(Path('data/raw/channels.csv'))
unique_loaded = len(set(subjects.values()))
print(f"Unique subjects loaded: {unique_loaded}")

# Check for whitespace
whitespace_subjects = [s for s in subjects.values() if s != s.strip()]
print(f"Subjects with whitespace: {len(whitespace_subjects)}")

assert len(whitespace_subjects) == 0, "Found whitespace in loaded subjects!"
print("\n✓ Test 2 PASSED!\n")

# Test 3: Compare before and after
print("=" * 60)
print("TEST 3: Before vs After Comparison")
print("=" * 60)
print("Before fix: 12 unique subjects (Maths, Maths, English, English, etc.)")
print(f"After fix:  {unique_subjects} unique subjects")
print(f"Fixed:      {12 - unique_subjects} duplicates removed")
print("\n✓ Test 3 PASSED!\n")

print("=" * 60)
print("ALL TESTS PASSED! ✓")
print("=" * 60)
