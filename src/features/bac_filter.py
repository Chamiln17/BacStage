"""
Bac 3AS Video Filter.

This module provides the main filtering logic to classify YouTube videos
as Bac (3AS) content or non-Bac content using keyword-based matching.

Usage:
    from src.features.bac_filter import Bac3ASFilter, filter_videos_dataframe

    # Single video
    filter_engine = Bac3ASFilter()
    result = filter_engine.filter_video(title, description, tags)

    # DataFrame
    df_filtered = filter_videos_dataframe(df)
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from tqdm import tqdm

from src.features.bac_keywords import BAC_KEYWORDS, GRADE_INDICATORS

logger = logging.getLogger(__name__)


class Bac3ASFilter:
    """
    Filter engine for classifying videos as Bac 3AS content.

    Uses keyword-based matching with subject-specific dictionaries
    and confidence scoring.
    """

    def __init__(self) -> None:
        """Initialize the filter with keyword dictionaries."""
        self.keywords = BAC_KEYWORDS
        self.grade_indicators = GRADE_INDICATORS
        # Compile regex for Arabic diacritics removal
        self._diacritics_pattern = re.compile(r"[\u064B-\u0652]")

    def clean_text(self, text: Optional[str]) -> str:
        """
        Normalize text for matching.

        Args:
            text: Input text (can be None)

        Returns:
            Lowercase stripped text, or empty string if None
        """
        if not text or not isinstance(text, str):
            return ""
        return text.lower().strip()

    def remove_arabic_diacritics(self, text: str) -> str:
        """
        Remove Arabic diacritics (tashkeel) for flexible matching.

        Args:
            text: Input text

        Returns:
            Text with diacritics removed
        """
        return self._diacritics_pattern.sub("", text)

    def count_keyword_matches(self, text: str, keywords_list: List[str]) -> int:
        """
        Count how many keywords from the list appear in the text.

        Uses word boundary matching for accuracy.

        Args:
            text: Text to search in
            keywords_list: List of keywords to search for

        Returns:
            Number of unique keywords found
        """
        text_clean = self.clean_text(text)
        text_clean_nodiac = self.remove_arabic_diacritics(text_clean)

        count = 0
        for keyword in keywords_list:
            keyword_clean = self.clean_text(keyword)
            if not keyword_clean:
                continue

            # Escape special regex characters
            keyword_escaped = re.escape(keyword_clean)

            # Try matching with word boundaries (works for Latin scripts)
            # For Arabic, also try without word boundaries due to script differences
            try:
                if (
                    re.search(r"\b" + keyword_escaped + r"\b", text_clean)
                    or re.search(r"\b" + keyword_escaped + r"\b", text_clean_nodiac)
                    or keyword_clean in text_clean  # Fallback for Arabic
                    or keyword_clean in text_clean_nodiac
                ):
                    count += 1
            except re.error:
                # If regex fails, fall back to simple substring match
                if keyword_clean in text_clean or keyword_clean in text_clean_nodiac:
                    count += 1

        return count

    def identify_subject(self, text: str) -> Tuple[Optional[str], int]:
        """
        Identify the most likely subject from the text.

        Args:
            text: Combined text (title + description + tags)

        Returns:
            Tuple of (subject_name, keyword_score)
        """
        text_clean = self.clean_text(text)
        best_subject: Optional[str] = None
        best_score = 0

        for subject_name, subject_dict in self.keywords.items():
            # Combine all language keywords for the subject
            all_keywords: List[str] = []
            for lang_keywords in subject_dict["bac_terms"].values():
                all_keywords.extend(lang_keywords)

            score = self.count_keyword_matches(text_clean, all_keywords)
            if score > best_score:
                best_score = score
                best_subject = subject_name

        return best_subject, best_score

    def check_bac_level(self, text: str) -> Optional[bool]:
        """
        Check if content explicitly targets Bac level.

        Args:
            text: Combined text to check

        Returns:
            True if Bac markers found, False if exclude markers found, None if ambiguous
        """
        text_clean = self.clean_text(text)
        text_clean_nodiac = self.remove_arabic_diacritics(text_clean)

        # Check exclude markers first (priority)
        for marker in self.grade_indicators["exclude_markers"]:
            marker_clean = self.clean_text(marker)
            if marker_clean in text_clean or marker_clean in text_clean_nodiac:
                return False

        # Check Bac markers
        for marker in self.grade_indicators["bac_markers"]:
            marker_clean = self.clean_text(marker)
            if marker_clean in text_clean or marker_clean in text_clean_nodiac:
                return True

        return None  # Ambiguous - no explicit markers

    def filter_video(
        self,
        title: Optional[str],
        description: Optional[str],
        tags: Optional[str] = "",
    ) -> Dict[str, Any]:
        """
        Main filtering function for a single video.

        Args:
            title: Video title
            description: Video description
            tags: Video tags (space or comma separated)

        Returns:
            Dictionary with filtering results:
                - is_bac_3as: bool - Whether video is classified as Bac 3AS
                - confidence: float - Confidence score (0-1)
                - subject: str - Detected subject name
                - subject_score: int - Number of subject keywords matched
                - bac_keyword_count: int - Total Bac keywords matched
                - exclude_keyword_count: int - Exclusion keywords matched
                - reason: str - Explanation for classification
        """
        # Combine all text
        title_str = str(title) if title else ""
        desc_str = str(description) if description else ""
        tags_str = str(tags) if tags else ""
        full_text = f"{title_str} {desc_str} {tags_str}"

        # Check explicit grade level first
        explicit_level = self.check_bac_level(full_text)
        if explicit_level is False:
            return {
                "is_bac_3as": False,
                "confidence": 0.0,
                "subject": None,
                "subject_score": 0,
                "bac_keyword_count": 0,
                "exclude_keyword_count": 1,
                "reason": "Explicit non-Bac grade marker detected",
            }

        # Identify subject
        subject, subject_score = self.identify_subject(full_text)

        if not subject:
            return {
                "is_bac_3as": False,
                "confidence": 0.0,
                "subject": None,
                "subject_score": 0,
                "bac_keyword_count": 0,
                "exclude_keyword_count": 0,
                "reason": "No subject detected",
            }

        # Get subject-specific data
        subject_dict = self.keywords[subject]

        # Count Bac keywords
        all_bac_keywords: List[str] = []
        for lang_keywords in subject_dict["bac_terms"].values():
            all_bac_keywords.extend(lang_keywords)
        bac_keyword_count = self.count_keyword_matches(full_text, all_bac_keywords)

        # Count exclude keywords
        all_exclude_keywords: List[str] = []
        exclude_terms = subject_dict.get("exclude_terms", {})
        for lang_keywords in exclude_terms.values():
            all_exclude_keywords.extend(lang_keywords)
        exclude_keyword_count = self.count_keyword_matches(full_text, all_exclude_keywords)

        # Calculate confidence score
        # Formula: bac_keywords / (bac_keywords + exclude_keywords + base_constant)
        # The +3 prevents overconfidence with very few matches
        denominator = max(bac_keyword_count + exclude_keyword_count + 3, 3)
        confidence = bac_keyword_count / denominator

        # Boost confidence if explicit Bac markers are present
        if explicit_level is True:
            confidence = min(confidence + 0.2, 1.0)

        # Apply threshold
        threshold = subject_dict.get("threshold", 0.7)
        is_bac = confidence >= threshold and exclude_keyword_count == 0

        # Override: if explicit Bac marker found, be more lenient
        if explicit_level is True and bac_keyword_count > 0:
            is_bac = True

        # Determine reason
        if is_bac:
            reason = f"Matched {bac_keyword_count} keywords for {subject}"
            if explicit_level is True:
                reason += " (explicit Bac marker found)"
        else:
            if exclude_keyword_count > 0:
                reason = f"Excluded: {exclude_keyword_count} exclusion terms found"
            elif confidence < threshold:
                reason = f"Below threshold: {confidence:.2f} < {threshold}"
            else:
                reason = "Insufficient keyword matches"

        return {
            "is_bac_3as": is_bac,
            "confidence": min(confidence, 1.0),
            "subject": subject,
            "subject_score": subject_score,
            "bac_keyword_count": bac_keyword_count,
            "exclude_keyword_count": exclude_keyword_count,
            "reason": reason,
        }


def filter_videos_dataframe(
    df: pd.DataFrame,
    title_col: str = "title",
    description_col: str = "description",
    tags_col: str = "tags",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Apply Bac filter to all videos in a DataFrame.

    Args:
        df: DataFrame with video data
        title_col: Column name for titles
        description_col: Column name for descriptions
        tags_col: Column name for tags
        show_progress: Whether to show progress bar

    Returns:
        Original DataFrame with new columns added:
            - is_bac_3as (bool)
            - filter_confidence (float)
            - detected_subject (str)
            - subject_score (int)
            - bac_keyword_count (int)
            - exclude_keyword_count (int)
            - filter_reason (str)
    """
    filter_engine = Bac3ASFilter()
    results: List[Dict[str, Any]] = []

    iterator = df.iterrows()
    if show_progress:
        iterator = tqdm(iterator, total=len(df), desc="Filtering videos")

    for _, row in iterator:
        result = filter_engine.filter_video(
            title=row.get(title_col, ""),
            description=row.get(description_col, ""),
            tags=row.get(tags_col, ""),
        )
        results.append(result)

    # Create results DataFrame
    results_df = pd.DataFrame(results)

    # Rename columns for clarity
    results_df = results_df.rename(
        columns={
            "confidence": "filter_confidence",
            "subject": "detected_subject",
            "reason": "filter_reason",
        }
    )

    # Combine with original DataFrame
    df_result = pd.concat([df.reset_index(drop=True), results_df], axis=1)

    return df_result


def get_filter_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate statistics from a filtered DataFrame.

    Args:
        df: DataFrame that has been processed by filter_videos_dataframe

    Returns:
        Dictionary with filtering statistics
    """
    total = len(df)
    bac_count = df["is_bac_3as"].sum()
    non_bac_count = total - bac_count

    stats = {
        "total_videos": total,
        "bac_3as_count": int(bac_count),
        "non_bac_count": int(non_bac_count),
        "bac_percentage": round(100 * bac_count / total, 2) if total > 0 else 0,
    }

    # Subject distribution (for Bac videos only)
    if bac_count > 0:
        bac_df = df[df["is_bac_3as"]]
        subject_dist = bac_df["detected_subject"].value_counts().to_dict()
        stats["subject_distribution"] = subject_dist

        # Confidence statistics
        conf_stats = bac_df["filter_confidence"].describe().to_dict()
        stats["confidence_stats"] = {
            "mean": round(conf_stats["mean"], 3),
            "std": round(conf_stats["std"], 3),
            "min": round(conf_stats["min"], 3),
            "max": round(conf_stats["max"], 3),
        }

    return stats
