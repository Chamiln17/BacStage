"""
Feature engineering: turns videos into the engagement model's features.

``VideoFeatureEngineer`` is fit once on training videos, then transforms any
videos into the model's feature matrix, including planned videos that have no
statistics yet. Everything learned from data (the channel table, subject
categories, selected columns, scaler, TF-IDF vocabulary, embedding PCA) is
learned in ``fit`` and reused by ``transform``, so training and prediction
build features the same way. ``transform`` never reads view, like, or comment
counts, and nothing in this module reads from disk.
"""

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, Protocol, Sequence

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

from src.features.bac_keywords import (
    canonical_subject,
    count_bac_markers,
    count_pedagogical_markers,
    get_exam_keyword_intensity,
)
from src.features.transcript_features import (
    extract_transcript_features,
    is_valid_transcript,
)

logger = logging.getLogger(__name__)

TRANSCRIPT_FEATURES = [
    "transcript_word_count",
    "transcript_char_count",
    "transcript_sentence_count",
    "avg_words_per_sentence",
    "lexical_diversity",
    "unique_word_count",
    "flesch_reading_ease",
    "flesch_kincaid_grade",
    "gunning_fog_index",
    "automated_readability_index",
    "speech_rate_wpm",
    "speech_rate_optimal",
    "speech_rate_above_optimal",
    "speech_rate_below_optimal",
    "question_count",
    "question_density",
    "example_count",
    "example_density",
    "explanation_count",
    "explanation_density",
    "contrast_count",
    "contrast_density",
    "technical_term_count",
    "technical_term_density",
    "subject_keyword_count",
    "subject_keyword_density",
    "domain_keyword_count",
    "domain_keyword_density",
    "bac_marker_count",
    "bac_marker_density",
    "pedagogical_marker_count",
    "pedagogical_marker_density",
]

CHANNEL_FEATURES = ["channel_video_count", "channel_avg_views", "channel_age_days"]

# Numeric features the model may use, before VIF selection. Subject one-hot
# columns are added from the categories seen in fit. Recency features
# (days_since_publish and its variants) are deliberately absent: a planned
# video has no age, so they would be out of distribution at prediction time.
NUMERIC_CANDIDATES = [
    "is_weekday",
    "duration_sec",
    "title_length",
    "description_length",
    "title_word_count",
    "tag_count",
    "is_exam_focused",
    "is_title_question",
    "exam_keyword_intensity",
    "title_bac_markers",
    "desc_pedagogical_markers",
    "has_transcript",
    *CHANNEL_FEATURES,
    *TRANSCRIPT_FEATURES,
]

# Columns of the engineered CSV export (`run_pipeline.py engineer`).
EXPORT_COLUMNS = [
    "video_id",
    "title",
    "channel_id",
    "channel_title",
    "description",
    "publish_date",
    "subject",
    "view_count",
    "like_count",
    "comment_count",
    "like_ratio",
    "comment_ratio",
    "engagement_score",
    "engagement_category",
    *NUMERIC_CANDIDATES,
]

UNKNOWN_SUBJECT = "Unknown"


class TextEmbedder(Protocol):
    """Anything that turns texts into fixed-size vectors (AraBERT in production)."""

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray: ...


def engagement_score(videos: pd.DataFrame) -> pd.Series:
    """The model's target: log1p((3 * comments + likes) / sqrt(views)).

    Comments weigh 3x likes because they signal active learning. Views of 0
    are treated as 1.
    """
    views = videos["view_count"].fillna(0).clip(lower=1)
    interactions = 3 * videos["comment_count"].fillna(0) + videos["like_count"].fillna(0)
    return np.log1p(interactions / np.sqrt(views)).rename("engagement_score")


def engagement_category(scores: pd.Series, thresholds: Sequence[float]) -> pd.Series:
    """Label scores Low / Medium / High using thresholds learned from training scores."""
    low, high = thresholds
    return pd.Series(
        np.select([scores >= high, scores >= low], ["High", "Medium"], "Low"),
        index=scores.index,
    )


def select_features_vif(
    X: pd.DataFrame, vif_threshold: float = 10.0, correlation_threshold: float = 0.95
) -> List[str]:
    """Drop near-duplicate columns, then iteratively drop the highest-VIF column.

    Args:
        X: Numeric feature frame.
        vif_threshold: Keep dropping while the max VIF is above this.
        correlation_threshold: Drop a column correlated above this with an earlier one.

    Returns:
        Names of the columns kept, in their original order.
    """
    X = X.fillna(0)
    X = X.loc[:, X.std() > 0]  # constant columns have undefined VIF
    upper = X.corr().abs().where(np.triu(np.ones((X.shape[1], X.shape[1])), k=1).astype(bool))
    X = X.drop(columns=[c for c in upper.columns if (upper[c] > correlation_threshold).any()])

    while X.shape[1] >= 2:
        with np.errstate(divide="ignore"):  # perfectly collinear columns get VIF = inf and drop first
            vifs = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
        worst = int(np.nanargmax(vifs))
        if vifs[worst] <= vif_threshold:
            break
        X = X.drop(columns=X.columns[worst])
    return X.columns.tolist()


# ponytail: process-wide memo; training transforms the same transcripts ~5 times.
# Unbounded, so memory grows with distinct transcripts seen (~10k here). Bound it
# with maxsize if this ever runs as a long-lived service over new videos.
_transcript_features = lru_cache(maxsize=None)(extract_transcript_features)


def _as_utc(value: Any) -> pd.Series:
    return pd.to_datetime(value, format="ISO8601", utc=True)


class VideoFeatureEngineer:
    """Fit on training videos; transform any videos into model features.

    Input columns:
        Required: ``title``, ``duration_sec``, and ``channel_id`` or ``subject``.
        Optional: ``description``, ``tags``, ``publish_date`` (defaults to now),
        ``transcript_text``, ``video_id``, ``channel_title``.
        ``fit`` also needs ``channel_id``, ``publish_date`` and ``view_count``
        to learn the channel table.
    """

    def __init__(
        self,
        embedder: Optional[TextEmbedder] = None,
        reference_date: Optional[datetime] = None,
        max_embedding_dims: int = 100,
        random_seed: int = 42,
    ) -> None:
        """
        Args:
            embedder: Text embedder for title + description. None skips embeddings.
            reference_date: "Now" for age features (channel age, days since publish).
                Defaults to the current UTC time.
            max_embedding_dims: PCA size for the embeddings.
            random_seed: Seed for PCA.
        """
        self.embedder = embedder
        now = reference_date or datetime.now(timezone.utc)
        self.reference_date = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        self.max_embedding_dims = max_embedding_dims
        self.random_seed = random_seed
        self.channel_table: Optional[pd.DataFrame] = None
        self.channel_defaults: Dict[str, float] = {}
        self.subjects: List[str] = []
        self.feature_columns: List[str] = []
        self.scaler: Optional[StandardScaler] = None
        self.tfidf: Optional[TfidfVectorizer] = None
        self.pca: Optional[PCA] = None

    def __getstate__(self) -> Dict[str, Any]:
        # The embedder (a ~500 MB model) is supplied again at load time, never pickled.
        state = self.__dict__.copy()
        state["embedder"] = None
        return state

    @property
    def uses_embeddings(self) -> bool:
        """True if fit learned an embedding PCA, so transform needs an embedder."""
        return self.pca is not None

    @property
    def feature_names(self) -> List[str]:
        """Names of the columns ``transform`` returns, in order."""
        if self.tfidf is None:
            raise RuntimeError("VideoFeatureEngineer is not fitted")
        names = list(self.feature_columns)
        names += [f"tfidf_{t}" for t in self.tfidf.get_feature_names_out()]
        if self.pca is not None:
            names += [f"embedding_pc{i}" for i in range(self.pca.n_components_)]
        return names

    def fit(self, videos: pd.DataFrame) -> "VideoFeatureEngineer":
        """Learn everything ``transform`` needs from training videos."""
        _require(videos, ["title", "duration_sec", "channel_id", "publish_date", "view_count"])
        self.channel_table = self._learn_channel_table(videos)
        self.channel_defaults = self.channel_table[CHANNEL_FEATURES].median().to_dict()

        per_video = self.video_features(videos)
        self.subjects = sorted(per_video["subject"].unique())
        candidates = self._numeric_frame(per_video, self._candidate_columns())
        self.feature_columns = select_features_vif(candidates)
        self.scaler = StandardScaler().fit(candidates[self.feature_columns])

        texts = _texts(videos)
        small = len(texts) < 20
        self.tfidf = TfidfVectorizer(
            max_features=30,
            min_df=1 if small else 10,
            max_df=1.0 if small else 0.5,
            ngram_range=(1, 2),
            sublinear_tf=True,
        ).fit(texts)

        self.pca = None
        if self.embedder is not None:
            embeddings = self.embedder.get_embeddings(texts, batch_size=32)
            dims = min(self.max_embedding_dims, *embeddings.shape)
            self.pca = PCA(n_components=dims, random_state=self.random_seed).fit(embeddings)

        logger.info(
            f"Fitted on {len(videos)} videos: {len(self.feature_columns)}/{len(candidates.columns)} "
            f"numeric features kept, {len(self.subjects)} subjects, "
            f"embeddings={'on' if self.pca is not None else 'off'}"
        )
        return self

    def transform(self, videos: pd.DataFrame) -> np.ndarray:
        """Build the model's feature matrix, one row per video."""
        if self.scaler is None or self.tfidf is None:
            raise RuntimeError("VideoFeatureEngineer is not fitted")
        per_video = self.video_features(videos)
        numeric = self._numeric_frame(per_video, self.feature_columns)
        texts = _texts(videos)
        parts = [self.scaler.transform(numeric), self.tfidf.transform(texts).toarray()]
        if self.pca is not None:
            if self.embedder is None:
                raise RuntimeError("Fitted with embeddings: set `embedder` before transform")
            parts.append(self.pca.transform(self.embedder.get_embeddings(texts, batch_size=32)))
        return np.hstack(parts)

    def known_channel(self, videos: pd.DataFrame) -> pd.Series:
        """True where the video's channel was seen in fit."""
        if self.channel_table is None:
            raise RuntimeError("VideoFeatureEngineer is not fitted")
        ids = videos["channel_id"] if "channel_id" in videos else pd.Series(None, index=videos.index)
        return ids.isin(self.channel_table.index)

    def video_features(self, videos: pd.DataFrame) -> pd.DataFrame:
        """Readable per-video features: temporal, text, transcript and channel.

        Needs a fitted channel table (``fit`` builds it first). Statistics
        columns are passed through untouched and never used.
        """
        if self.channel_table is None:
            raise RuntimeError("VideoFeatureEngineer is not fitted")
        _require(videos, ["title", "duration_sec"])
        if "channel_id" not in videos and "subject" not in videos:
            raise ValueError("Each video needs a channel_id or a subject")

        df = videos.copy().reset_index(drop=True)
        for col in ["description", "tags"]:
            df[col] = df[col].fillna("").astype(str) if col in df else ""
        df["title"] = df["title"].fillna("").astype(str)
        df["duration_sec"] = pd.to_numeric(df["duration_sec"], errors="coerce").fillna(0)
        if "channel_id" not in df:
            df["channel_id"] = None
        if "publish_date" not in df:
            df["publish_date"] = pd.NaT
        # A planned video without a date is scored as if published now.
        df["publish_date"] = _as_utc(df["publish_date"]).fillna(pd.Timestamp.now(tz="UTC"))

        channel = self.channel_table.reindex(df["channel_id"]).reset_index(drop=True)
        given = df["subject"] if "subject" in df else pd.Series(np.nan, index=df.index)
        given = given.map(lambda x: x if pd.isna(x) or x == UNKNOWN_SUBJECT else canonical_subject(x))
        df["subject"] = given.fillna(channel["subject"]).fillna(UNKNOWN_SUBJECT)
        for col in CHANNEL_FEATURES:
            df[col] = channel[col].fillna(self.channel_defaults[col]).values

        df = self._create_temporal_features(df)
        df = self._create_text_features(df)

        texts = df["transcript_text"] if "transcript_text" in df else pd.Series(None, index=df.index)
        df["has_transcript"] = texts.map(is_valid_transcript).astype(int)
        transcript = pd.DataFrame(
            [
                _transcript_features(t, float(d), s)
                for t, d, s in zip(texts, df["duration_sec"], df["subject"], strict=True)
            ],
            index=df.index,
        )
        return pd.concat([df, transcript[TRANSCRIPT_FEATURES]], axis=1)

    def _learn_channel_table(self, videos: pd.DataFrame) -> pd.DataFrame:
        df = videos.assign(publish_date=_as_utc(videos["publish_date"]))
        if "subject" not in df:
            df["subject"] = UNKNOWN_SUBJECT
        table = df.groupby("channel_id").agg(
            subject=("subject", lambda s: s.mode().iat[0] if s.notna().any() else UNKNOWN_SUBJECT),
            channel_video_count=("title", "size"),
            channel_avg_views=("view_count", "mean"),
            first_publish=("publish_date", "min"),
        )
        table["channel_age_days"] = (pd.Timestamp(self.reference_date) - table["first_publish"]).dt.days
        return table.drop(columns="first_publish")

    def _candidate_columns(self) -> List[str]:
        # Same encoding as pd.get_dummies(drop_first=True) on sorted categories.
        return NUMERIC_CANDIDATES + [f"subject_{s}" for s in self.subjects[1:]]

    @staticmethod
    def _numeric_frame(per_video: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        frame = pd.DataFrame(index=per_video.index)
        for col in columns:
            if col.startswith("subject_") and col not in per_video:
                frame[col] = (per_video["subject"] == col[len("subject_"):]).astype(float)
            else:
                frame[col] = pd.to_numeric(per_video[col], errors="coerce").astype(float)
        return frame.fillna(0.0)

    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create temporal features from publish date.

        Features created:
        Basic temporal:
        - days_since_publish: Age of video in days
        - publish_hour: Hour of day (0-23)
        - publish_day_of_week: Day name (Monday-Sunday)
        - publish_month, publish_year: Month and year numbers
        - is_evening_upload: Binary (17-21h)
        - is_weekday: Binary (Sun-Thu)

        Cyclic encoding (Phase 1 improvements):
        - publish_hour_sin, publish_hour_cos: Cyclic encoding of hour (24h cycle)
        - publish_month_sin, publish_month_cos: Cyclic encoding of month (12-month cycle)
        - publish_day_of_month: Day of month (1-31)
        - publish_day_of_month_sin, publish_day_of_month_cos: Cyclic encoding of day of month

        Polynomial features:
        - days_since_publish_squared: Non-linear decay effect
        - log_days_since_publish: Log transform for better scaling

        Interaction features:
        - days_since_publish_x_hour: Recency × upload time interaction
        - is_weekday_x_hour: Weekday × upload hour interaction
        - days_since_publish_x_is_weekday: Recency × weekday interaction

        Args:
            df: DataFrame with publish_date column

        Returns:
            DataFrame with added temporal features
        """
        logger.debug("Creating temporal features")

        # Clipped: a planned video dated after the reference date has age 0.
        df["days_since_publish"] = (self.reference_date - df["publish_date"]).dt.days.clip(lower=0)

        df["publish_hour"] = df["publish_date"].dt.hour
        df["publish_day_of_week"] = df["publish_date"].dt.day_name()
        df["publish_month"] = df["publish_date"].dt.month
        df["publish_year"] = df["publish_date"].dt.year
        df["publish_day_of_month"] = df["publish_date"].dt.day

        # Evening upload (5-9 PM, optimal time per research)
        df["is_evening_upload"] = (
            df["publish_hour"].isin([17, 18, 19, 20, 21])
        ).astype(int)

        # Weekday upload
        df["is_weekday"] = (
            ~df["publish_day_of_week"].isin(["Saturday", "Friday"])
        ).astype(int)

        # PHASE 1: Cyclic encoding for time features
        # Hour: 24-hour cycle (preserves continuity: 23h is close to 0h)
        df["publish_hour_sin"] = np.sin(2 * np.pi * df["publish_hour"] / 24)
        df["publish_hour_cos"] = np.cos(2 * np.pi * df["publish_hour"] / 24)

        # Month: 12-month cycle
        df["publish_month_sin"] = np.sin(2 * np.pi * df["publish_month"] / 12)
        df["publish_month_cos"] = np.cos(2 * np.pi * df["publish_month"] / 12)

        # Day of month: 31-day cycle (approximate month length)
        df["publish_day_of_month_sin"] = np.sin(2 * np.pi * df["publish_day_of_month"] / 31)
        df["publish_day_of_month_cos"] = np.cos(2 * np.pi * df["publish_day_of_month"] / 31)

        # PHASE 1: Polynomial features for non-linear relationships
        # Square of days (captures accelerated decay/growth patterns)
        df["days_since_publish_squared"] = df["days_since_publish"] ** 2

        # Log transform (handles exponential growth/decay, reduces skew)
        df["log_days_since_publish"] = np.log1p(df["days_since_publish"])  # log1p = log(1+x) to handle 0

        # PHASE 1: Time-based interaction features
        # Recency × upload time (fresh content at different hours may perform differently)
        df["days_since_publish_x_hour"] = df["days_since_publish"] * df["publish_hour"]

        # Weekday × upload hour (timing effects differ on weekends vs weekdays)
        df["is_weekday_x_hour"] = df["is_weekday"] * df["publish_hour"]

        # Recency × weekday (weekend content may age differently)
        df["days_since_publish_x_is_weekday"] = df["days_since_publish"] * df["is_weekday"]

        logger.debug(
            f"Temporal features: evening={df['is_evening_upload'].sum()}, "
            f"weekday={df['is_weekday'].sum()}, "
            f"cyclic features added: 6, interaction features added: 3"
        )

        return df

    def _create_text_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features from title and description text.

        Features created:
        - title_length: Character count
        - description_length: Character count
        - title_word_count: Word count
        - tag_count: Number of tags
        - is_exam_focused: Binary indicator for exam-related content

        Args:
            df: DataFrame with title and description columns

        Returns:
            DataFrame with added text features
        """
        logger.debug("Creating text features")

        df["title_length"] = df["title"].str.len()
        df["description_length"] = df["description"].str.len()
        df["title_word_count"] = df["title"].str.split().str.len()

        # Tag count
        if "tags" in df.columns:
            df["tag_count"] = df["tags"].str.split(",").str.len()
            df.loc[df["tags"] == "", "tag_count"] = 0
        else:
            df["tag_count"] = 0

        # Exam-focused content detection
        exam_keywords = [
            "bac",
            "exam",
            "examen",
            "2024",
            "2025",
            "revision",
            "exercise",
            "تمرين",
            "امتحان",
            "باك",
        ]
        pattern = "|".join(exam_keywords)
        df["is_exam_focused"] = (
            df["title"].str.lower().str.contains(pattern, na=False)
        ).astype(int)

        # TIER 2: Additional text features
        # Is title a question? (engagement signal)
        df["is_title_question"] = (
            df["title"].str.strip().str.endswith(("?", "؟"))
        ).astype(int)

        # Exam keyword intensity (normalized by word count)
        df["exam_keyword_intensity"] = [
            get_exam_keyword_intensity(f"{t} {d}", n + len(str(d).split()))
            for t, d, n in zip(df["title"], df["description"], df["title_word_count"], strict=True)
        ]
        df["title_bac_markers"] = df["title"].map(lambda x: count_bac_markers(str(x)))
        df["desc_pedagogical_markers"] = df["description"].map(
            lambda x: count_pedagogical_markers(str(x))
        )

        logger.debug(
            f"Text features: exam_focused={df['is_exam_focused'].sum()}, "
            f"title_questions={df['is_title_question'].sum()}"
        )

        return df



def _require(videos: pd.DataFrame, columns: List[str]) -> None:
    missing = [c for c in columns if c not in videos.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _texts(videos: pd.DataFrame) -> List[str]:
    """Title + description, the text the TF-IDF and embeddings see."""
    title = videos["title"].fillna("").astype(str)
    description = (
        videos["description"].fillna("").astype(str) if "description" in videos else ""
    )
    return (title + " " + description).tolist()


def engineered_export(videos: pd.DataFrame, reference_date: Optional[datetime] = None) -> pd.DataFrame:
    """The engineered CSV for notebooks: per-video features plus engagement columns.

    Channel statistics here come from all ``videos``. For modeling, fit
    ``VideoFeatureEngineer`` on the training split only.
    """
    engineer = VideoFeatureEngineer(reference_date=reference_date)
    engineer.channel_table = engineer._learn_channel_table(videos)
    engineer.channel_defaults = engineer.channel_table[CHANNEL_FEATURES].median().to_dict()
    df = engineer.video_features(videos)
    views = df["view_count"].fillna(0).clip(lower=1)
    df["like_ratio"] = df["like_count"].fillna(0) / views
    df["comment_ratio"] = df["comment_count"].fillna(0) / views
    df["engagement_score"] = engagement_score(df)
    df["engagement_category"] = engagement_category(
        df["engagement_score"], df["engagement_score"].quantile([0.33, 0.67]).tolist()
    )
    return df[[c for c in EXPORT_COLUMNS if c in df.columns]]
