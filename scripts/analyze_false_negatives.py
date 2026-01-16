#!/usr/bin/env python
"""
Analyze false negatives from manual review to discover missing Bac markers.

This script identifies common phrases in videos that were:
- Marked as non-Bac by the filter (is_bac_3as=False)
- But are actually Bac content (manual_is_bac=True)

Usage:
    uv run python scripts/analyze_false_negatives.py
"""

import sys
from pathlib import Path
from collections import Counter
import re

import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Fix console encoding for Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")


def extract_phrases(text: str, min_length: int = 2) -> list:
    """Extract potential meaningful phrases from text."""
    if pd.isna(text):
        return []
    
    text = str(text).lower()
    
    # Split by common separators
    phrases = []
    
    # Extract quoted phrases
    quoted = re.findall(r'["\']([^"\']+)["\']', text)
    phrases.extend(quoted)
    
    # Extract phrases with common Bac-related words (even without "bac" marker)
    patterns = [
        r'((?:شرح|درس|حل|تمرين|مراجعة|تحضير)\s+[^\s،,]+(?:\s+[^\s،,]+)?)',
        r'((?:série|exercice|cours|leçon|révision)\s+\w+(?:\s+\w+)?)',
        r'((?:دروس|تمارين|حلول)\s+[^\s،,]+)',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text)
        phrases.extend(matches)
    
    # Clean and filter
    phrases = [p.strip() for p in phrases if len(p.strip()) >= min_length]
    
    return phrases


def main():
    print("=" * 80)
    print("FALSE NEGATIVE ANALYSIS")
    print("=" * 80)
    
    # Load validation data
    val_path = PROJECT_ROOT / "data" / "validation" / "sample_for_manual_review.csv"
    
    if not val_path.exists():
        print(f"Error: {val_path} not found")
        return 1
    
    df = pd.read_csv(val_path)
    print(f"\nLoaded {len(df)} validation samples")
    
    # Check for manual reviews
    reviewed = df.dropna(subset=['manual_is_bac'])
    if len(reviewed) == 0:
        print("Error: No manual reviews found in 'manual_is_bac' column")
        return 1
    
    print(f"Found {len(reviewed)} manually reviewed samples")
    
    # Identify false negatives
    # False negatives: filter said NOT Bac, but manual review says IS Bac
    false_negatives = reviewed[(reviewed['is_bac_3as'] == False) & (reviewed['manual_is_bac'] == True)]
    
    print(f"\n{'='*80}")
    print(f"FALSE NEGATIVES: {len(false_negatives)} videos")
    print(f"{'='*80}")
    print("(Videos incorrectly rejected as non-Bac)")
    
    if len(false_negatives) == 0:
        print("\n✓ No false negatives found! Filter is working perfectly.")
        return 0
    
    # Analyze by category
    print(f"\nFalse negatives by filter category:")
    print(false_negatives['filter_category'].value_counts().to_string())
    
    # Analyze titles
    print(f"\n{'='*80}")
    print("TITLE ANALYSIS")
    print(f"{'='*80}")
    
    print("\n--- Sample False Negative Titles ---\n")
    for i, (idx, row) in enumerate(false_negatives.head(20).iterrows(), 1):
        title = row['title'][:100]
        category = row['filter_category']
        confidence = row['filter_confidence']
        print(f"{i:2d}. [{category}] {title}")
        print(f"    Confidence: {confidence:.2f} | Reason: {row['filter_reason'][:50]}")
        print()
    
    # Check for existing markers
    print(f"\n{'='*80}")
    print("MARKER ANALYSIS")
    print(f"{'='*80}")
    
    # Existing markers to check
    existing_bac_markers = ['bac', '3as', 'بكالوريا', 'باك', 'ثالثة ثانوي', 'terminale']
    
    fn_text = (
        false_negatives['title'].fillna('') + ' ' + 
        false_negatives['description'].fillna('')
    ).str.lower()
    
    for marker in existing_bac_markers:
        has_marker = fn_text.str.contains(marker, case=False, na=False).sum()
        print(f"Videos with '{marker}': {has_marker} / {len(false_negatives)}")
    
    # Extract common words and phrases
    print(f"\n{'='*80}")
    print("COMMON TERMS IN FALSE NEGATIVES")
    print(f"{'='*80}")
    
    # Arabic terms
    arabic_pattern = r'[\u0600-\u06FF]+'
    arabic_words = []
    for title in false_negatives['title'].fillna(''):
        words = re.findall(arabic_pattern, str(title))
        arabic_words.extend(words)
    
    arabic_counter = Counter(arabic_words)
    print("\nTop 30 Arabic terms:")
    for term, count in arabic_counter.most_common(30):
        pct = (count / len(false_negatives)) * 100
        print(f"  {term:30s} : {count:3d} ({pct:.1f}%)")
    
    # Latin/French terms
    latin_pattern = r'\b[a-zA-Z]{3,}\b'
    latin_words = []
    for title in false_negatives['title'].fillna(''):
        words = re.findall(latin_pattern, str(title).lower())
        latin_words.extend(words)
    
    latin_counter = Counter(latin_words)
    print("\nTop 20 Latin/French terms:")
    for term, count in latin_counter.most_common(20):
        pct = (count / len(false_negatives)) * 100
        print(f"  {term:30s} : {count:3d} ({pct:.1f}%)")
    
    # Bigrams
    print(f"\n{'='*80}")
    print("COMMON BIGRAMS (2-word phrases)")
    print(f"{'='*80}")
    
    bigrams = []
    for title in false_negatives['title'].fillna(''):
        words = str(title).split()
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if len(bigram.strip()) > 5:  # Filter very short
                bigrams.append(bigram.lower())
    
    bigram_counter = Counter(bigrams)
    print("\nTop 30 bigrams:")
    for bigram, count in bigram_counter.most_common(30):
        if count >= 2:  # Only show bigrams appearing 2+ times
            pct = (count / len(false_negatives)) * 100
            print(f"  {bigram:50s} : {count:3d} ({pct:.1f}%)")
    
    # Suggested markers
    print(f"\n{'='*80}")
    print("SUGGESTED MARKERS TO ADD")
    print(f"{'='*80}")
    print("\nBased on frequency and Bac-relevance, consider adding:\n")
    
    # High-value suggestions (manual curation based on common patterns)
    suggestions = []
    
    # Check for common educational terms
    edu_terms = {
        'درس': 'lesson/class',
        'شرح': 'explanation',
        'تمرين': 'exercise',
        'حل': 'solution',
        'مراجعة': 'revision',
        'ملخص': 'summary',
        'série': 'series (exercises)',
        'cours': 'course',
        'exercice': 'exercise',
        'correction': 'correction',
    }
    
    for term, meaning in edu_terms.items():
        count = fn_text.str.contains(term, case=False, na=False).sum()
        if count >= 3:
            pct = (count / len(false_negatives)) * 100
            suggestions.append({
                'term': term,
                'meaning': meaning,
                'count': count,
                'pct': pct,
            })
    
    suggestions.sort(key=lambda x: x['count'], reverse=True)
    
    for i, sugg in enumerate(suggestions[:10], 1):
        print(f"{i:2d}. '{sugg['term']:20s}' - {sugg['meaning']:20s} ({sugg['count']:3d} videos, {sugg['pct']:.1f}%)")
    
    print(f"\n{'='*80}")
    print("RECOMMENDATIONS")
    print(f"{'='*80}")
    print("""
1. Review the sample titles above and identify patterns
2. Check the common terms for Bac-specific context
3. Add 3-5 high-signal phrases to config/filter_config.yaml under markers.bac
4. Re-run filter: uv run python run_pipeline.py filter_data --force-discovery
5. Re-validate to check improvement
    """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
