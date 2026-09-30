"""
Transcript module: decides whether a transcript is valid and turns it into features.

Every caller that needs to know "is this real spoken text?" (cleaning, collection,
feature engineering) goes through ``is_valid_transcript``. Features include
readability scores, speech rate, vocabulary diversity, and educational markers.
"""

import logging
import re
from typing import Any, Optional

from textstat import (
    automated_readability_index,
    flesch_kincaid_grade,
    flesch_reading_ease,
    gunning_fog,
)

from src.features.bac_keywords import (
    count_bac_markers,
    count_domain_keywords,
    count_pedagogical_markers,
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


# Markers of YouTube page code (JS/HTML) captured instead of captions.
CORRUPTION_MARKERS = (
    "window.",
    "ytcfg",
    "u003d",  # URL-encoded "="
    "u0026",  # URL-encoded "&"
    "javascript",
    "var ",
    "function(",
    "<html",
    "innertubeapi",
    "commandu0026",
)


def is_valid_transcript(text: Any) -> bool:
    """Return True if ``text`` is real spoken text.

    A valid transcript is a non-empty string containing fewer than two
    corruption markers. Two or more markers means YouTube returned page code.
    """
    if not isinstance(text, str) or not text.strip():
        return False
    lower = text.lower()
    return sum(marker in lower for marker in CORRUPTION_MARKERS) < 2


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
    if not is_valid_transcript(transcript_text):
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
    unique_words = {w.lower() for w in words if len(w) > 2}
    features['lexical_diversity'] = len(unique_words) / word_count
    features['unique_word_count'] = len(unique_words)

    # ============================================
    # 3. READABILITY SCORES (English/French optimized)
    # ============================================
    try:
        features['flesch_reading_ease'] = flesch_reading_ease(transcript_text)
    except Exception as e:
        logger.debug(f"Error calculating flesch_reading_ease: {str(e)[:100]}")
        features['flesch_reading_ease'] = 50.0

    try:
        features['flesch_kincaid_grade'] = flesch_kincaid_grade(transcript_text)
    except Exception as e:
        logger.debug(f"Error calculating flesch_kincaid_grade: {str(e)[:100]}")
        features['flesch_kincaid_grade'] = 10.0

    try:
        features['gunning_fog_index'] = gunning_fog(transcript_text)
    except Exception as e:
        logger.debug(f"Error calculating gunning_fog: {str(e)[:100]}")
        features['gunning_fog_index'] = 10.0

    try:
        features['automated_readability_index'] = automated_readability_index(transcript_text)
    except Exception as e:
        logger.debug(f"Error calculating ARI: {str(e)[:100]}")
        features['automated_readability_index'] = 10.0

    # ============================================
    # 4. PACING FEATURES (if duration provided)
    # ============================================
    if duration_sec and duration_sec > 0:
        features['speech_rate_wpm'] = (word_count / duration_sec) * 60
        features['speech_rate_optimal'] = 1 if 120 <= features['speech_rate_wpm'] <= 180 else 0
        features['speech_rate_above_optimal'] = 1 if features['speech_rate_wpm'] > 180 else 0
        features['speech_rate_below_optimal'] = 1 if features['speech_rate_wpm'] < 100 else 0
    else:
        features['speech_rate_wpm'] = None
        features['speech_rate_optimal'] = None
        features['speech_rate_above_optimal'] = None
        features['speech_rate_below_optimal'] = None

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
    # 7. DOMAIN-SPECIFIC KEYWORDS (from bac_keywords module)
    # ============================================
    domain_keyword_count = count_domain_keywords(transcript_text, subject)
    features['domain_keyword_count'] = domain_keyword_count
    features['domain_keyword_density'] = domain_keyword_count / word_count if word_count > 0 else 0.0

    # Legacy compatibility (subject_keyword_count mapped to domain)
    features['subject_keyword_count'] = domain_keyword_count
    features['subject_keyword_density'] = features['domain_keyword_density']

    # ============================================
    # 8. BAC EXAM MARKERS
    # ============================================
    bac_marker_count = count_bac_markers(transcript_text)
    features['bac_marker_count'] = bac_marker_count
    features['bac_marker_density'] = bac_marker_count / word_count if word_count > 0 else 0.0

    # ============================================
    # 9. PEDAGOGICAL MARKERS
    # ============================================
    pedagogical_count = count_pedagogical_markers(transcript_text)
    features['pedagogical_marker_count'] = pedagogical_count
    features['pedagogical_marker_density'] = pedagogical_count / word_count if word_count > 0 else 0.0

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
        'speech_rate_above_optimal': None,
        'speech_rate_below_optimal': None,
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
        'domain_keyword_count': 0,
        'domain_keyword_density': 0.0,
        'subject_keyword_count': 0,
        'subject_keyword_density': 0.0,
        'bac_marker_count': 0,
        'bac_marker_density': 0.0,
        'pedagogical_marker_count': 0,
        'pedagogical_marker_density': 0.0,
    }
