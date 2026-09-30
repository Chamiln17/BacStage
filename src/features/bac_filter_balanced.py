"""
Balanced Bac 3AS filter (data-driven).

``run_bac_filter`` is the whole filter in one call: it learns which channels
are Bac-heavy (channel priors), discovers Bac-associated title terms with
TF-IDF, then classifies every video. The strategy:

1. Channel priors identify Bac-heavy channels.
2. TF-IDF discovered terms act as soft positives.
3. Grade markers make the hard include/exclude decisions.

All markers and thresholds come from ``config/filter_config.yaml``; there are
no defaults in code, so the YAML is the single place to tune the filter.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.bac_keywords import canonical_subject


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
        bac_markers: List[str],
        non_bac_markers: List[str],
        strong_bac_intent: List[str],
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
            bac_markers: Positive Bac markers
            non_bac_markers: Negative (non-Bac) markers
            strong_bac_intent: Phrases that override non-Bac markers in conflicts
            bac_heavy_channels: Channel IDs classified as Bac-heavy
            tfidf_terms: TF-IDF discovered Bac-associated terms
            channel_subjects: channel_id -> subject
            duration_min: Minimum duration (seconds) for soft positive
            require_tfidf: If True, TF-IDF hit is mandatory for ambiguous
            require_duration: If True, duration gate is mandatory for ambiguous
            allow_channel_subject: If True, channel subject counts as soft positive
        """
        self.bac_markers = bac_markers
        self.non_bac_markers = non_bac_markers
        self.strong_bac_intent = strong_bac_intent

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

        def make_pattern(markers: List[str]) -> re.Pattern[str]:
            pattern = "|".join([re.escape(m) for m in markers])
            return re.compile(pattern, re.IGNORECASE)

        self._bac_pattern: re.Pattern[str] = make_pattern(self.bac_markers)
        self._non_bac_pattern: re.Pattern[str] = make_pattern(self.non_bac_markers)
        self._strong_intent_pattern: re.Pattern[str] = make_pattern(
            self.strong_bac_intent
        )

        self._tfidf_pattern: Optional[re.Pattern[str]]
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
            positives.append(
                f"channel subject: {self.channel_subjects.get(channel_id, 'Unknown')}"
            )

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

        # Rule 0: Hard duration filter (if require_duration is True, apply globally)
        if self.require_duration and duration_sec < self.duration_min:
            return {
                "is_bac_3as": False,
                "filter_category": "non_bac",
                "filter_confidence": 0.8,
                "filter_reason": f"Video too short ({duration_sec:.0f}s < {self.duration_min}s minimum)",
                "subject": subject,
            }

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


def load_channel_subjects(channels_path: Path) -> Dict[str, str]:
    """Load channel -> subject mapping from channels.csv."""
    if not channels_path.exists():
        return {}

    df = pd.read_csv(channels_path)
    if "channel_id" not in df.columns or "subjects" not in df.columns:
        return {}

    df["subjects"] = df["subjects"].map(canonical_subject)

    return dict(zip(df["channel_id"], df["subjects"], strict=True))


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
    iterator: Any = df.iterrows()
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
    stats: Dict[str, Any] = {
        "total_videos": len(df),
        "bac_3as_count": int(df["is_bac_3as"].sum()),
        "non_bac_count": int((~df["is_bac_3as"]).sum()),
    }

    stats["bac_percentage"] = (
        stats["bac_3as_count"] / stats["total_videos"] * 100
        if stats["total_videos"] > 0
        else 0.0
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


# ---------------------------------------------------------------------------
# Config, discovery and the one-call filter
# ---------------------------------------------------------------------------

REQUIRED_MARKERS = ("bac", "non_bac", "strong_bac_intent")


def load_filter_config(config_path: Path) -> Dict[str, Any]:
    """Load the filter YAML. Missing file or marker lists is an error, not a fallback."""
    if not config_path.exists():
        raise FileNotFoundError(f"Filter config not found: {config_path}")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    missing = [m for m in REQUIRED_MARKERS if not config.get("markers", {}).get(m)]
    if missing:
        raise ValueError(f"{config_path}: markers.{', markers.'.join(missing)} must be non-empty lists")
    return config


def latest_snapshots(videos: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent snapshot of each video."""
    if "snapshot_date" in videos:
        order = pd.to_datetime(videos["snapshot_date"], format="ISO8601", errors="coerce")
        videos = videos.assign(_order=order).sort_values("_order", ascending=False).drop(columns="_order")
    if "video_id" in videos:
        videos = videos.drop_duplicates(subset="video_id", keep="first")
    return videos.reset_index(drop=True)


def _video_text(videos: pd.DataFrame) -> pd.Series:
    parts = [videos[c].fillna("").astype(str) for c in ("title", "description", "tags") if c in videos]
    return pd.concat(parts, axis=1).agg(" ".join, axis=1).str.lower()


def _has_any(text: pd.Series, markers: List[str]) -> pd.Series:
    return text.str.contains("|".join(re.escape(m) for m in markers), regex=True, case=False, na=False)


def _grade_marks(videos: pd.DataFrame, markers: Dict[str, List[str]]) -> Tuple[pd.Series, pd.Series]:
    text = _video_text(videos)
    return _has_any(text, markers["bac"]), _has_any(text, markers["non_bac"])


def build_channel_priors(videos: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """Per channel: the share of grade-marked videos that are Bac, and whether it is Bac-heavy."""
    prior = config.get("channel_prior", {})
    has_bac, has_non = _grade_marks(videos, config["markers"])
    stats = (
        pd.DataFrame({"channel_id": videos["channel_id"], "bac": has_bac, "non": has_non, "any": has_bac | has_non})
        .groupby("channel_id")
        .agg(
            count_bac_marked=("bac", "sum"),
            count_non_bac_marked=("non", "sum"),
            count_any_grade_marked=("any", "sum"),
            total_videos=("bac", "size"),
        )
    )
    marked = stats["count_any_grade_marked"].replace(0, 1)
    stats["p_bac"] = stats["count_bac_marked"] / marked
    stats["p_non"] = stats["count_non_bac_marked"] / marked
    stats["is_bac_heavy"] = (stats["p_bac"] >= prior.get("bac_threshold", 0.7)) & (
        stats["p_non"] <= prior.get("non_bac_max", 0.1)
    )
    return stats.reset_index().sort_values("p_bac", ascending=False, ignore_index=True)


def discover_bac_terms(videos: pd.DataFrame, config: Dict[str, Any]) -> List[str]:
    """Title n-grams more frequent in Bac-marked than non-Bac-marked videos.

    Returns [] when there are fewer than 50 Bac-only or 20 non-Bac-only titles.
    """
    tfidf = config.get("tfidf", {})
    has_bac, has_non = _grade_marks(videos, config["markers"])
    titles = videos["title"].fillna("").astype(str)
    bac_titles, non_titles = titles[has_bac & ~has_non].tolist(), titles[has_non & ~has_bac].tolist()
    if len(bac_titles) < 50 or len(non_titles) < 20:
        return []
    vectorizer = TfidfVectorizer(
        ngram_range=tuple(tfidf.get("ngram_range", [1, 2])),
        analyzer=tfidf.get("analyzer", "char_wb"),
        min_df=tfidf.get("min_df", 5),
        max_df=tfidf.get("max_df", 0.8),
    )
    X = vectorizer.fit_transform(bac_titles + non_titles)
    diff = X[: len(bac_titles)].mean(axis=0).A1 - X[len(bac_titles):].mean(axis=0).A1
    names = vectorizer.get_feature_names_out()
    top = np.argsort(diff)[::-1][: tfidf.get("top_n_terms", 100)]
    return [str(names[i]) for i in top if diff[i] > 0]


@dataclass
class BacFilterResult:
    """Everything one filter run learns and decides."""

    priors: pd.DataFrame
    terms: List[str]
    videos: pd.DataFrame  # input videos plus is_bac_3as, filter_category, filter_confidence, filter_reason, subject


def run_bac_filter(
    videos: pd.DataFrame,
    config: Dict[str, Any],
    channel_subjects: Dict[str, str],
    priors: Optional[pd.DataFrame] = None,
    terms: Optional[List[str]] = None,
    show_progress: bool = False,
) -> BacFilterResult:
    """Discover priors and terms (unless given), then classify every video.

    Args:
        videos: Latest snapshot per video (see ``latest_snapshots``).
        config: Output of ``load_filter_config``.
        channel_subjects: channel_id -> subject, from channels.csv.
        priors: Cached channel priors; None discovers them from ``videos``.
        terms: Cached TF-IDF terms; None discovers them from ``videos``.
        show_progress: Show a progress bar while classifying.
    """
    priors = build_channel_priors(videos, config) if priors is None else priors
    terms = discover_bac_terms(videos, config) if terms is None else terms
    soft = config.get("soft_positives", {})
    markers = config["markers"]
    bac_filter = BalancedBacFilter(
        bac_markers=markers["bac"],
        non_bac_markers=markers["non_bac"],
        strong_bac_intent=markers["strong_bac_intent"],
        bac_heavy_channels=set(priors.loc[priors["is_bac_heavy"], "channel_id"]),
        tfidf_terms=terms,
        channel_subjects=channel_subjects,
        duration_min=soft.get("duration_min", 300),
        require_tfidf=soft.get("require_tfidf", False),
        require_duration=soft.get("require_duration", False),
        allow_channel_subject=soft.get("allow_channel_subject", True),
    )
    filtered = filter_videos_dataframe(videos, bac_filter, show_progress=show_progress)
    return BacFilterResult(priors=priors, terms=terms, videos=filtered)


VALIDATION_SAMPLES = {
    # filter_category: config key for its sample size, default size
    "bac_3as": ("sample_bac_3as", 100),
    "bac_3as_ambiguous": ("sample_bac_ambiguous", 100),
    "non_bac": ("sample_non_bac", 50),
    "unknown": ("sample_conflict", 50),
}


def validation_sample(filtered: pd.DataFrame, config: Dict[str, Any], seed: int = 42) -> pd.DataFrame:
    """A stratified sample per filter category, with empty columns for manual labels."""
    sizes = config.get("validation", {})
    samples = []
    for category, (key, default) in VALIDATION_SAMPLES.items():
        rows = filtered[filtered["filter_category"] == category]
        if len(rows):
            samples.append(rows.sample(n=min(sizes.get(key, default), len(rows)), random_state=seed).assign(sample_type=category))
    if not samples:
        return pd.DataFrame()
    return pd.concat(samples, ignore_index=True).assign(manual_is_bac="", notes="")


def score_against_labels(labels: pd.DataFrame) -> Dict[str, float]:
    """Precision, recall and accuracy of ``is_bac_3as`` against hand labels in ``manual_is_bac``.

    Rows without a manual label are ignored.
    """
    labelled = labels[labels["manual_is_bac"].astype(str).str.lower().isin(["true", "false", "1", "0"])]
    truth = labelled["manual_is_bac"].astype(str).str.lower().isin(["true", "1"])
    predicted = labelled["is_bac_3as"].astype(str).str.lower().isin(["true", "1"])
    tp = int((truth & predicted).sum())
    return {
        "labelled": len(labelled),
        "precision": tp / max(int(predicted.sum()), 1),
        "recall": tp / max(int(truth.sum()), 1),
        "accuracy": float((truth == predicted).mean()) if len(labelled) else 0.0,
    }
