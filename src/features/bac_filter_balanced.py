"""
Balanced Bac 3AS Filter (Data-Driven Approach)

This module implements a balanced filtering strategy that:
1. Uses channel priors (data-driven) to identify Bac-heavy channels
2. Uses TF-IDF discovered terms for soft positives
3. Applies minimal grade markers for hard include/exclude decisions

The filter maximizes recall while maintaining quality, with minimal manual
keyword maintenance.
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd


class BalancedBacFilter:
    """
    Data-driven Bac 3AS filter using channel priors and TF-IDF terms.
    
    The filtering logic follows a decision tree:
    1. Hard exclude: non-Bac markers present (unless strong Bac intent)
    2. Hard include: Bac markers present
    3. Ambiguous: Use channel prior + soft positives
    """
    
    def __init__(
        self,
        bac_markers: Optional[List[str]] = None,
        non_bac_markers: Optional[List[str]] = None,
        strong_bac_intent: Optional[List[str]] = None,
        bac_heavy_channels: Optional[Set[str]] = None,
        tfidf_terms: Optional[List[str]] = None,
        channel_subjects: Optional[Dict[str, str]] = None,
        duration_min: int = 300,
        require_tfidf: bool = False,
        require_duration: bool = False,
        allow_channel_subject: bool = True,
    ):
        """
        Initialize the balanced Bac filter.
        
        Args:
            bac_markers: List of positive Bac markers
            non_bac_markers: List of negative (non-Bac) markers
            strong_bac_intent: Phrases that override non-Bac markers in conflicts
            bac_heavy_channels: Set of channel IDs classified as Bac-heavy
            tfidf_terms: List of TF-IDF discovered Bac-associated terms
            channel_subjects: Dict mapping channel_id -> subject
            duration_min: Minimum duration (seconds) for soft positive
            require_tfidf: If True, TF-IDF hit is mandatory for ambiguous
            require_duration: If True, duration gate is mandatory for ambiguous
            allow_channel_subject: If True, channel subject counts as soft positive
        """
        # Default markers
        self.bac_markers = bac_markers or [
            "bac", "3as", "بكالوريا", "باك", "ثالثة ثانوي",
            "السنة الثالثة ثانوي", "terminale"
        ]
        
        self.non_bac_markers = non_bac_markers or [
            "1as", "2as", "سنة أولى ثانوي", "سنة ثانية ثانوي",
            "أولى ثانوي", "ثانية ثانوي", "متوسط", "bem",
            "1am", "2am", "3am", "4am", "ابتدائي"
        ]
        
        self.strong_bac_intent = strong_bac_intent or [
            "مراجعة بكالوريا", "تحضير بكالوريا", "تصحيح بكالوريا",
            "موضوع بكالوريا", "حل موضوع بكالوريا",
            "bac blanc", "révision bac", "corrigé bac", "sujet bac"
        ]
        
        # Data-driven components
        self.bac_heavy_channels = bac_heavy_channels or set()
        self.tfidf_terms = tfidf_terms or []
        self.channel_subjects = channel_subjects or {}
        
        # Soft positive config
        self.duration_min = duration_min
        self.require_tfidf = require_tfidf
        self.require_duration = require_duration
        self.allow_channel_subject = allow_channel_subject
        
        # Compile regex patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for marker detection."""
        def make_pattern(markers: List[str]) -> re.Pattern:
            pattern = "|".join([re.escape(m) for m in markers])
            return re.compile(pattern, re.IGNORECASE)
        
        self._bac_pattern = make_pattern(self.bac_markers)
        self._non_bac_pattern = make_pattern(self.non_bac_markers)
        self._strong_intent_pattern = make_pattern(self.strong_bac_intent)
        
        if self.tfidf_terms:
            self._tfidf_pattern = make_pattern(self.tfidf_terms)
        else:
            self._tfidf_pattern = None
    
    def _has_bac_markers(self, text: str) -> bool:
        """Check if text contains Bac markers."""
        return bool(self._bac_pattern.search(text))
    
    def _has_non_bac_markers(self, text: str) -> bool:
        """Check if text contains non-Bac markers."""
        return bool(self._non_bac_pattern.search(text))
    
    def _has_strong_bac_intent(self, text: str) -> bool:
        """Check if text contains strong Bac intent phrases."""
        return bool(self._strong_intent_pattern.search(text))
    
    def _has_tfidf_hit(self, text: str) -> bool:
        """Check if text contains TF-IDF discovered terms."""
        if self._tfidf_pattern is None:
            return False
        return bool(self._tfidf_pattern.search(text))
    
    def _check_soft_positives(
        self,
        text: str,
        channel_id: str,
        duration_sec: float,
    ) -> Tuple[bool, str]:
        """
        Check if video has soft positives (for ambiguous cases).
        
        Returns:
            Tuple of (has_soft_positive, reason)
        """
        positives = []
        
        # Duration gate
        has_duration = duration_sec >= self.duration_min
        if has_duration:
            positives.append(f"duration >= {self.duration_min}s")
        
        # TF-IDF hit
        has_tfidf = self._has_tfidf_hit(text)
        if has_tfidf:
            positives.append("TF-IDF term hit")
        
        # Channel subject
        has_subject = self.allow_channel_subject and channel_id in self.channel_subjects
        if has_subject:
            positives.append(f"channel subject: {self.channel_subjects.get(channel_id, 'Unknown')}")
        
        # Apply requirements
        if self.require_tfidf and self.require_duration:
            # Strict: require both TF-IDF AND duration
            passed = has_tfidf and has_duration
        elif self.require_tfidf:
            # TF-IDF mandatory
            passed = has_tfidf
        elif self.require_duration:
            # Duration mandatory
            passed = has_duration
        else:
            # Lenient: any soft positive is enough
            passed = has_duration or has_tfidf or has_subject
        
        reason = ", ".join(positives) if positives else "no soft positives"
        return passed, reason
    
    def filter_video(
        self,
        title: str,
        description: str,
        tags: str,
        channel_id: str,
        duration_sec: float,
    ) -> Dict[str, Any]:
        """
        Apply balanced filter to a single video.
        
        Args:
            title: Video title
            description: Video description
            tags: Video tags (comma-separated string)
            channel_id: YouTube channel ID
            duration_sec: Video duration in seconds
        
        Returns:
            Dict with filter results:
            - is_bac_3as: bool
            - filter_category: str (bac_3as | non_bac | unknown | bac_3as_ambiguous)
            - filter_confidence: float (0-1)
            - filter_reason: str
            - subject: str (from channel mapping)
        """
        # Combine text for analysis
        text = f"{title} {description} {tags}".lower()
        
        # Check for markers
        has_bac = self._has_bac_markers(text)
        has_non_bac = self._has_non_bac_markers(text)
        has_strong_intent = self._has_strong_bac_intent(text)
        
        # Get channel info
        is_bac_heavy = channel_id in self.bac_heavy_channels
        subject = self.channel_subjects.get(channel_id, "Unknown")
        
        # Decision tree
        
        # Rule 1: Hard exclude (non-Bac markers, unless strong intent override)
        if has_non_bac and not has_bac:
            if has_strong_intent:
                # Conflict override: strong intent saves it
                return {
                    "is_bac_3as": True,
                    "filter_category": "bac_3as",
                    "filter_confidence": 0.7,
                    "filter_reason": "Strong Bac intent overrides non-Bac markers",
                    "subject": subject,
                }
            else:
                return {
                    "is_bac_3as": False,
                    "filter_category": "non_bac",
                    "filter_confidence": 0.9,
                    "filter_reason": "Non-Bac grade markers detected",
                    "subject": subject,
                }
        
        # Rule 2: Conflict (both markers present)
        if has_bac and has_non_bac:
            if has_strong_intent:
                return {
                    "is_bac_3as": True,
                    "filter_category": "bac_3as",
                    "filter_confidence": 0.75,
                    "filter_reason": "Both markers present, strong Bac intent resolves conflict",
                    "subject": subject,
                }
            else:
                # Default: exclude conflicts without strong intent
                return {
                    "is_bac_3as": False,
                    "filter_category": "non_bac",
                    "filter_confidence": 0.5,
                    "filter_reason": "Both Bac and non-Bac markers present, no strong intent",
                    "subject": subject,
                }
        
        # Rule 3: Hard include (explicit Bac markers)
        if has_bac:
            return {
                "is_bac_3as": True,
                "filter_category": "bac_3as",
                "filter_confidence": 0.95,
                "filter_reason": "Explicit Bac markers detected",
                "subject": subject,
            }
        
        # Rule 4: Ambiguous (no grade markers)
        if not is_bac_heavy:
            return {
                "is_bac_3as": False,
                "filter_category": "unknown",
                "filter_confidence": 0.4,
                "filter_reason": "No markers, channel not Bac-heavy",
                "subject": subject,
            }
        
        # Ambiguous + Bac-heavy channel: check soft positives
        passed, soft_reason = self._check_soft_positives(text, channel_id, duration_sec)
        
        if passed:
            return {
                "is_bac_3as": True,
                "filter_category": "bac_3as_ambiguous",
                "filter_confidence": 0.65,
                "filter_reason": f"Bac-heavy channel + soft positives ({soft_reason})",
                "subject": subject,
            }
        else:
            return {
                "is_bac_3as": False,
                "filter_category": "unknown",
                "filter_confidence": 0.35,
                "filter_reason": f"Bac-heavy channel but no soft positives ({soft_reason})",
                "subject": subject,
            }


def load_channel_priors(priors_path: Path) -> Set[str]:
    """Load Bac-heavy channel IDs from channel_priors.csv."""
    if not priors_path.exists():
        return set()
    
    df = pd.read_csv(priors_path)
    bac_heavy = df[df["is_bac_heavy"] == True]["channel_id"].tolist()
    return set(bac_heavy)


def load_tfidf_terms(terms_path: Path) -> List[str]:
    """Load TF-IDF discovered terms from JSON."""
    if not terms_path.exists():
        return []
    
    with open(terms_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_channel_subjects(channels_path: Path) -> Dict[str, str]:
    """Load channel -> subject mapping from channels.csv."""
    if not channels_path.exists():
        return {}
    
    df = pd.read_csv(channels_path)
    if "channel_id" not in df.columns or "subjects" not in df.columns:
        return {}
    
    return dict(zip(df["channel_id"], df["subjects"]))


def filter_videos_dataframe(
    df: pd.DataFrame,
    bac_filter: BalancedBacFilter,
    title_col: str = "title",
    description_col: str = "description",
    tags_col: str = "tags",
    channel_col: str = "channel_id",
    duration_col: str = "duration_sec",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Apply balanced Bac filter to a DataFrame of videos.
    
    Args:
        df: Input DataFrame
        bac_filter: Configured BalancedBacFilter instance
        title_col: Name of title column
        description_col: Name of description column
        tags_col: Name of tags column
        channel_col: Name of channel ID column
        duration_col: Name of duration column
        show_progress: Show progress bar
    
    Returns:
        DataFrame with filter columns added
    """
    results = []
    
    # Optional progress bar
    iterator = df.iterrows()
    if show_progress:
        try:
            from tqdm import tqdm
            iterator = tqdm(df.iterrows(), total=len(df), desc="Filtering videos")
        except ImportError:
            pass
    
    for _, row in iterator:
        title = str(row.get(title_col, "") or "")
        description = str(row.get(description_col, "") or "")
        tags = str(row.get(tags_col, "") or "")
        channel_id = str(row.get(channel_col, "") or "")
        duration_sec = float(row.get(duration_col, 0) or 0)
        
        result = bac_filter.filter_video(
            title=title,
            description=description,
            tags=tags,
            channel_id=channel_id,
            duration_sec=duration_sec,
        )
        results.append(result)
    
    # Add results to DataFrame
    result_df = pd.DataFrame(results)
    df_out = df.copy()
    
    for col in result_df.columns:
        df_out[col] = result_df[col].values
    
    return df_out


def get_filter_statistics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute filter statistics from filtered DataFrame.
    
    Args:
        df: DataFrame with filter columns
    
    Returns:
        Dict with statistics
    """
    stats = {
        "total_videos": len(df),
        "bac_3as_count": int(df["is_bac_3as"].sum()),
        "non_bac_count": int((~df["is_bac_3as"]).sum()),
    }
    
    stats["bac_percentage"] = (
        stats["bac_3as_count"] / stats["total_videos"] * 100
        if stats["total_videos"] > 0 else 0
    )
    
    # Category distribution
    if "filter_category" in df.columns:
        stats["category_distribution"] = df["filter_category"].value_counts().to_dict()
    
    # Subject distribution (for Bac videos)
    if "subject" in df.columns:
        bac_only = df[df["is_bac_3as"]]
        stats["subject_distribution"] = bac_only["subject"].value_counts().to_dict()
    
    # Confidence stats
    if "filter_confidence" in df.columns:
        bac_only = df[df["is_bac_3as"]]
        if len(bac_only) > 0:
            stats["confidence_stats"] = {
                "mean": float(bac_only["filter_confidence"].mean()),
                "std": float(bac_only["filter_confidence"].std()),
                "min": float(bac_only["filter_confidence"].min()),
                "max": float(bac_only["filter_confidence"].max()),
            }
    
    return stats
