"""
Feature engineering module for video engagement analysis.

This module transforms raw video metadata into engineered features suitable for
machine learning models.
"""

import logging
from datetime import datetime
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class VideoFeatureEngineer:
    """
    Feature engineering for YouTube video engagement analysis.

    Transforms raw video metadata into features that capture:
    - Temporal patterns (upload timing, age)
    - Content characteristics (duration, text features)
    - Engagement metrics (ratios, scores)
    - Channel-level features
    """

    def __init__(self, collection_date: Optional[datetime] = None, channels_path: Optional[str] = None) -> None:
        """
        Initialize feature engineer.

        Args:
            collection_date: Reference date for calculating recency features.
                           Defaults to current datetime if not provided.
            channels_path: Path to channels.csv with subject labels.
                          Defaults to data/raw/channels.csv if not provided.
        """
        if collection_date is None:
            # Make timezone-aware to match API data
            from datetime import timezone

            collection_date = datetime.now(timezone.utc)
        elif collection_date.tzinfo is None:
            # Make timezone-aware if naive datetime provided
            from datetime import timezone

            collection_date = collection_date.replace(tzinfo=timezone.utc)

        self.collection_date = collection_date
        
        # Load channel subjects
        if channels_path is None:
            from pathlib import Path
            channels_path = Path("data/raw/channels.csv")
        
        try:
            self.channels_df = pd.read_csv(channels_path)
            logger.info(f"Loaded {len(self.channels_df)} channels with subjects")
        except FileNotFoundError:
            logger.warning(f"channels.csv not found at {channels_path}, subject column will be missing")
            self.channels_df = None
        
        logger.info(
            f"Feature engineer initialized with collection date: {self.collection_date}"
        )

    def fit_transform(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw video metadata into engineered features.

        Args:
            videos_df: DataFrame with raw video metadata

        Returns:
            DataFrame with engineered features

        Raises:
            ValueError: If required columns are missing
        """
        required_cols = [
            "video_id",
            "title",
            "description",
            "publish_date",
            "duration_sec",
            "view_count",
            "like_count",
            "comment_count",
        ]
        missing_cols = [col for col in required_cols if col not in videos_df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        logger.info(f"Starting feature engineering on {len(videos_df)} videos")

        df = videos_df.copy()

        # Ensure datetime type (handle both ISO8601 and standard formats)
        if not pd.api.types.is_datetime64_any_dtype(df["publish_date"]):
            df["publish_date"] = pd.to_datetime(
                df["publish_date"], format="ISO8601", utc=True
            )

        # Clean basic fields
        df = self._clean_data(df)

        # Create feature groups
        df = self._create_temporal_features(df)
        df = self._create_text_features(df)
        df = self._create_engagement_features(df)
        df = self._create_channel_features(df)

        logger.info(f"Feature engineering complete: {df.shape[1]} features")

        return df

    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and prepare raw data.

        Args:
            df: Raw video DataFrame

        Returns:
            Cleaned DataFrame
        """
        logger.debug("Cleaning data")

        # Remove videos with 0 views (likely private/unlisted)
        initial_count = len(df)
        df = df[df["view_count"] > 0].copy()
        removed = initial_count - len(df)
        if removed > 0:
            logger.info(f"Removed {removed} videos with 0 views")

        # Handle missing statistics
        df["comment_count"] = df["comment_count"].fillna(0).astype(int)
        df["like_count"] = df["like_count"].fillna(0).astype(int)
        df["description"] = df["description"].fillna("")
        df["tags"] = df["tags"].fillna("") if "tags" in df.columns else ""

        return df

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

        df["days_since_publish"] = (self.collection_date - df["publish_date"]).dt.days

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
        - subject: Detected subject category
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

        # Subject from channel (merge from channels.csv)
        if self.channels_df is not None and "channel_id" in df.columns:
            # Drop existing subject column if present
            if 'subject' in df.columns:
                df = df.drop(columns=['subject'])
            
            df = df.merge(
                self.channels_df[['channel_id', 'subjects']],
                on='channel_id',
                how='left'
            )
            df = df.rename(columns={'subjects': 'subject'})
            df['subject'] = df['subject'].fillna('Unknown')
            logger.debug(f"Assigned subjects from channels: {df['subject'].value_counts().to_dict()}")
        else:
            df['subject'] = 'Unknown'
            logger.warning("No channel subjects available, all videos marked as Unknown")

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

        logger.debug(
            f"Text features: exam_focused={df['is_exam_focused'].sum()}"
        )

        return df

    def _create_engagement_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create engagement metrics and ratios.

        Features created:
        - like_ratio: likes / views
        - comment_ratio: comments / views
        - engagement_score: weighted combination
        - engagement_category: Low/Medium/High classification

        Args:
            df: DataFrame with view_count, like_count, comment_count

        Returns:
            DataFrame with added engagement features
        """
        logger.debug("Creating engagement features")

        # Avoid division by zero
        safe_views = df["view_count"].replace(0, 1)

        # Ratios
        df["like_ratio"] = df["like_count"] / safe_views
        df["comment_ratio"] = df["comment_count"] / safe_views

        # Age Factor: min(1.0, days_since_publish / 365)
        # Using 365.25 days for year duration approximation
        df["age_factor"] = (df["days_since_publish"] / 365.25).clip(upper=0.50)

        # Enhanced Weighted Engagement Score
        # Formula: (0.4 * LikeRatio + 0.5 * CommentRatio + 0.1 * AgeFactor) * 100
        df["engagement_score"] = (
            0.40 * df["like_ratio"] +
            0.50 * df["comment_ratio"] +
            0.10 * df["age_factor"]
        ) * 100

        # Engagement category (based on percentiles)
        df["engagement_category"] = self._categorize_engagement(df["engagement_score"])

        logger.debug(
            f"Engagement distribution: "
            f"{df['engagement_category'].value_counts().to_dict()}"
        )

        return df

    @staticmethod
    def _categorize_engagement(scores: pd.Series) -> pd.Series:
        """
        Categorize engagement scores into Low/Medium/High.

        Args:
            scores: Series of engagement scores

        Returns:
            Series of category labels
        """
        percentile_33 = scores.quantile(0.33)
        percentile_67 = scores.quantile(0.67)

        def categorize(score: float) -> str:
            if score >= percentile_67:
                return "High"
            elif score >= percentile_33:
                return "Medium"
            else:
                return "Low"

        return scores.apply(categorize)

    def _create_channel_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create channel-level aggregated features.

        Features created:
        - channel_video_count: Number of videos in dataset from this channel
        - channel_avg_views: Average views for channel
        - channel_avg_engagement: Average engagement ratio for channel
        - channel_age_days: Days since oldest video from channel

        Args:
            df: DataFrame with channel_id

        Returns:
            DataFrame with added channel features
        """
        logger.debug("Creating channel features")

        if "channel_id" not in df.columns:
            logger.warning("channel_id column missing, skipping channel features")
            return df

        # Channel statistics
        channel_stats = (
            df.groupby("channel_id")
            .agg(
                {
                    "video_id": "count",
                    "view_count": "mean",
                    "duration_sec": "mean",
                    "publish_date": "min",
                }
            )
            .rename(
                columns={
                    "video_id": "channel_video_count",
                    "view_count": "channel_avg_views",
                    "duration_sec": "channel_video_avg_duration",
                }
            )
        )

        # Channel age (days since first video)
        channel_stats["channel_age_days"] = (
            self.collection_date - channel_stats["publish_date"]
        ).dt.days
        channel_stats = channel_stats.drop("publish_date", axis=1)

        # Merge back to original dataframe
        df = df.merge(channel_stats, on="channel_id", how="left")

        logger.debug(f"Channel features: {df['channel_id'].nunique()} unique channels")

        return df

    def select_features(
        self, df: pd.DataFrame, include_target: bool = True
    ) -> pd.DataFrame:
        """
        Select final feature set for modeling.

        Args:
            df: DataFrame with all engineered features
            include_target: Whether to include target variable (engagement_category)

        Returns:
            DataFrame with selected features
        """
        feature_columns = [
            # Identifiers
            "video_id",
            "title",
            "channel_id",
            "channel_title",
            "description",
            # Temporal (basic)
            "publish_date",
            # Content
            "duration_sec",
            "title_length",
            "description_length",
            "title_word_count",
            "tag_count",
            "subject",
            "is_exam_focused",
            # Engagement (features)
            "view_count",
            "like_count",
            "comment_count",
            "like_ratio",
            "comment_ratio",
            "engagement_score",
            # Channel
            "channel_video_count",
            "channel_avg_views",
            "channel_video_avg_duration",
            "channel_age_days",
        ]

        if include_target:
            feature_columns.append("engagement_category")

        # Only include columns that exist
        available_columns = [col for col in feature_columns if col in df.columns]

        logger.info(f"Selected {len(available_columns)} features for modeling")

        return df[available_columns].copy()
