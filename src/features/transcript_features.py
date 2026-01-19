"""
Transcript Feature Extractor

Extracts linguistic and pedagogical features from video transcripts.
Features include readability scores, speech rate, vocabulary diversity, and educational markers.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from textstat import (
    flesch_reading_ease,
    flesch_kincaid_grade,
    gunning_fog,
    automated_readability_index,
)

logger = logging.getLogger(__name__)


def count_syllables(word: str) -> int:
    """Approximate syllable count for a word."""
    word = word.lower()
    count = 0
    vowels = 'aeiouéèêëàâùûôîïœæ'  # French/Arabic vowels included
    previous_was_vowel = False
    for char in word:
        is_vowel = char in vowels
        if is_vowel and not previous_was_vowel:
            count += 1
        previous_was_vowel = is_vowel
    return max(1, count)


def extract_transcript_features(
    transcript_text: str,
    duration_sec: Optional[float] = None,
    subject: Optional[str] = None,
) -> dict:
    """
    Extract features from a transcript.
    
    Args:
        transcript_text: Full transcript text
        duration_sec: Video duration in seconds (for pacing features)
        subject: Video subject for keyword matching
        
    Returns:
        Dictionary of extracted features
    """
    if not transcript_text or pd.isna(transcript_text):
        return _empty_features()
    
    features = {}
    
    # Tokenize
    words = transcript_text.split()
    word_count = len(words)
    
    if word_count == 0:
        return _empty_features()
    
    # Sentences (handle multiple delimiters)
    sentences = re.split(r'[.!?؟。]+', transcript_text)
    sentences = [s.strip() for s in sentences if s.strip()]
    sentence_count = max(len(sentences), 1)
    
    # ============================================
    # 1. BASIC METRICS
    # ============================================
    features['transcript_word_count'] = word_count
    features['transcript_char_count'] = len(transcript_text)
    features['transcript_sentence_count'] = sentence_count
    features['avg_words_per_sentence'] = word_count / sentence_count
    
    # ============================================
    # 2. VOCABULARY DIVERSITY
    # ============================================
    unique_words = set(w.lower() for w in words if len(w) > 2)
    features['lexical_diversity'] = len(unique_words) / word_count
    features['unique_word_count'] = len(unique_words)
    
    # ============================================
    # 3. READABILITY SCORES (English/French optimized)
    # ============================================
    try:
        features['flesch_reading_ease'] = flesch_reading_ease(transcript_text)
        features['flesch_kincaid_grade'] = flesch_kincaid_grade(transcript_text)
        features['gunning_fog_index'] = gunning_fog(transcript_text)
        features['automated_readability_index'] = automated_readability_index(transcript_text)
    except Exception:
        features['flesch_reading_ease'] = 50.0  # Default neutral
        features['flesch_kincaid_grade'] = 10.0
        features['gunning_fog_index'] = 10.0
        features['automated_readability_index'] = 10.0
    
    # ============================================
    # 4. PACING FEATURES (if duration provided)
    # ============================================
    if duration_sec and duration_sec > 0:
        features['speech_rate_wpm'] = (word_count / duration_sec) * 60
        features['speech_rate_optimal'] = 1 if 120 <= features['speech_rate_wpm'] <= 180 else 0
    else:
        features['speech_rate_wpm'] = None
        features['speech_rate_optimal'] = None
    
    # ============================================
    # 5. EDUCATIONAL MARKERS
    # ============================================
    text_lower = transcript_text.lower()
    
    # Question frequency (indicates interactive teaching)
    question_count = text_lower.count('?') + text_lower.count('؟')
    features['question_count'] = question_count
    features['question_density'] = question_count / sentence_count
    
    # Example/illustration markers
    example_markers = [
        'example', 'for instance', 'such as', 'like', 'suppose', 'imagine',
        'مثال', 'مثلا', 'par exemple', 'comme', 'prenons'
    ]
    example_count = sum(text_lower.count(m) for m in example_markers)
    features['example_count'] = example_count
    features['example_density'] = example_count / sentence_count
    
    # Explanation markers (shows pedagogical structure)
    explanation_markers = [
        'because', 'therefore', 'thus', 'since', 'as a result', 'so',
        'لأن', 'إذن', 'لذلك', 'parce que', 'donc', 'ainsi', 'car'
    ]
    explanation_count = sum(text_lower.count(m) for m in explanation_markers)
    features['explanation_count'] = explanation_count
    features['explanation_density'] = explanation_count / word_count
    
    # Contrast markers (shows nuanced teaching)
    contrast_markers = [
        'however', 'but', 'although', 'instead', 'on the other hand',
        'لكن', 'ولكن', 'بينما', 'mais', 'cependant', 'toutefois'
    ]
    contrast_count = sum(text_lower.count(m) for m in contrast_markers)
    features['contrast_count'] = contrast_count
    features['contrast_density'] = contrast_count / sentence_count
    
    # ============================================
    # 6. TECHNICAL TERM DENSITY
    # ============================================
    technical_terms = [
        'theorem', 'proof', 'formula', 'equation', 'function', 'derivative',
        'integral', 'théorème', 'formule', 'équation', 'fonction',
        'نظرية', 'برهان', 'معادلة', 'دالة'
    ]
    technical_count = sum(text_lower.count(t) for t in technical_terms)
    features['technical_term_count'] = technical_count
    features['technical_term_density'] = technical_count / word_count
    
    # ============================================
    # 7. SUBJECT-SPECIFIC KEYWORDS
    # ============================================
    subject_keywords = {
        'Mathematics': ['derivative', 'integral', 'function', 'limit', 'equation',
                       'مشتقة', 'تكامل', 'دالة', 'dérivée', 'intégrale'],
        'Physics': ['force', 'energy', 'momentum', 'velocity', 'wave',
                   'قوة', 'طاقة', 'سرعة', 'force', 'énergie'],
        'Science': ['molecule', 'cell', 'organism', 'reaction', 'evolution',
                   'خلية', 'تفاعل', 'cellule', 'réaction'],
    }
    
    if subject and subject in subject_keywords:
        keywords = subject_keywords[subject]
        match_count = sum(text_lower.count(kw.lower()) for kw in keywords)
        features['subject_keyword_count'] = match_count
        features['subject_keyword_density'] = match_count / word_count
    else:
        features['subject_keyword_count'] = 0
        features['subject_keyword_density'] = 0.0
    
    return features


def _empty_features() -> dict:
    """Return empty feature dict for missing transcripts."""
    return {
        'transcript_word_count': 0,
        'transcript_char_count': 0,
        'transcript_sentence_count': 0,
        'avg_words_per_sentence': 0.0,
        'lexical_diversity': 0.0,
        'unique_word_count': 0,
        'flesch_reading_ease': None,
        'flesch_kincaid_grade': None,
        'gunning_fog_index': None,
        'automated_readability_index': None,
        'speech_rate_wpm': None,
        'speech_rate_optimal': None,
        'question_count': 0,
        'question_density': 0.0,
        'example_count': 0,
        'example_density': 0.0,
        'explanation_count': 0,
        'explanation_density': 0.0,
        'contrast_count': 0,
        'contrast_density': 0.0,
        'technical_term_count': 0,
        'technical_term_density': 0.0,
        'subject_keyword_count': 0,
        'subject_keyword_density': 0.0,
    }


class TranscriptFeatureExtractor:
    """Batch extractor for transcript features."""
    
    def __init__(self):
        self.feature_count = len(_empty_features())
        logger.info(f"Initialized TranscriptFeatureExtractor with {self.feature_count} features")
    
    def extract_all(
        self,
        transcripts_df: pd.DataFrame,
        duration_df: Optional[pd.DataFrame] = None,
        subject_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        Extract features from all transcripts.
        
        Args:
            transcripts_df: DataFrame with video_id and transcript_text columns
            duration_df: Optional DataFrame with video_id and duration_sec columns
            subject_df: Optional DataFrame with video_id and subject columns
            
        Returns:
            DataFrame with video_id and all extracted features
        """
        results = []
        total = len(transcripts_df)
        
        logger.info(f"Extracting features from {total} transcripts")
        
        for idx, row in transcripts_df.iterrows():
            video_id = row['video_id']
            transcript_text = row.get('transcript_text', '')
            
            # Get duration if available
            duration_sec = None
            if duration_df is not None:
                dur_row = duration_df[duration_df['video_id'] == video_id]
                if len(dur_row) > 0:
                    duration_sec = dur_row.iloc[0].get('duration_sec')
            
            # Get subject if available
            subject = None
            if subject_df is not None:
                subj_row = subject_df[subject_df['video_id'] == video_id]
                if len(subj_row) > 0:
                    subject = subj_row.iloc[0].get('subject')
            
            # Extract features
            features = extract_transcript_features(transcript_text, duration_sec, subject)
            features['video_id'] = video_id
            results.append(features)
            
            if (idx + 1) % 500 == 0:
                logger.info(f"Progress: {idx + 1}/{total}")
        
        logger.info(f"Feature extraction complete: {len(results)} videos processed")
        return pd.DataFrame(results)


def extract_features_cli(input_path: Path, output_path: Path, videos_path: Optional[Path] = None) -> int:
    """
    CLI entry point for feature extraction.
    
    Args:
        input_path: Path to transcripts CSV
        output_path: Path to save features
        videos_path: Optional path to videos CSV (for duration/subject)
        
    Returns:
        Exit code (0 for success)
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    logger.info("=" * 60)
    logger.info("Transcript Feature Extraction Pipeline")
    logger.info("=" * 60)
    
    # Load transcripts
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        return 1
    
    transcripts_df = pd.read_csv(input_path)
    logger.info(f"Loaded {len(transcripts_df)} transcripts from {input_path}")
    
    # Load videos for duration/subject if available
    duration_df = None
    subject_df = None
    if videos_path and videos_path.exists():
        videos_df = pd.read_csv(videos_path)
        if 'duration_sec' in videos_df.columns:
            duration_df = videos_df[['video_id', 'duration_sec']]
            logger.info("Using duration data for speech rate calculation")
        if 'subject' in videos_df.columns:
            subject_df = videos_df[['video_id', 'subject']]
            logger.info("Using subject data for keyword matching")
    
    # Extract features
    extractor = TranscriptFeatureExtractor()
    features_df = extractor.extract_all(transcripts_df, duration_df, subject_df)
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(output_path, index=False)
    logger.info(f"Saved {len(features_df.columns)} features to: {output_path}")
    
    return 0


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract transcript features")
    parser.add_argument("--input", type=Path, required=True, help="Transcripts CSV")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV path")
    parser.add_argument("--videos", type=Path, help="Videos CSV for duration/subject")
    
    args = parser.parse_args()
    exit(extract_features_cli(args.input, args.output, args.videos))
