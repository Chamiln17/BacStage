"""Tests for feature engineering module."""

from datetime import datetime

import pandas as pd
import pytest

from src.features.engineer import VideoFeatureEngineer


class TestVideoFeatureEngineer:
    """Test suite for VideoFeatureEngineer class."""

    def test_init_default_date(self) -> None:
        """Test initialization with default collection date."""
        engineer = VideoFeatureEngineer()
        assert isinstance(engineer.collection_date, datetime)

    def test_init_custom_date(self) -> None:
        """Test initialization with custom collection date."""
        custom_date = datetime(2024, 6, 1)
        engineer = VideoFeatureEngineer(collection_date=custom_date)
        assert engineer.collection_date == custom_date

    def test_fit_transform_missing_columns(self) -> None:
        """Test fit_transform raises error with missing required columns."""
        engineer = VideoFeatureEngineer()
        invalid_df = pd.DataFrame({"video_id": ["vid1"], "title": ["Test"]})

        with pytest.raises(ValueError, match="Missing required columns"):
            engineer.fit_transform(invalid_df)

    def test_fit_transform_success(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test successful feature engineering transformation."""
        engineer = VideoFeatureEngineer(collection_date=datetime(2024, 6, 1))
        result = engineer.fit_transform(sample_raw_videos)

        # Check that new features were created
        assert "days_since_publish" in result.columns
        assert "publish_hour" in result.columns
        assert "is_evening_upload" in result.columns
        assert "is_weekday" in result.columns
        assert "title_length" in result.columns
        assert "subject" in result.columns
        assert "like_ratio" in result.columns
        assert "engagement_score" in result.columns
        assert "engagement_category" in result.columns

        # Check no data loss
        assert len(result) <= len(sample_raw_videos)  # May remove 0-view videos

    def test_temporal_features(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test temporal feature creation."""
        engineer = VideoFeatureEngineer(collection_date=datetime(2024, 6, 1))
        result = engineer._create_temporal_features(sample_raw_videos.copy())

        # Check evening upload detection
        assert result.loc[0, "is_evening_upload"] == 1  # 18:00
        assert result.loc[1, "is_evening_upload"] == 0  # 14:30
        assert result.loc[2, "is_evening_upload"] == 1  # 20:15

        # Check publish hour extraction
        assert result.loc[0, "publish_hour"] == 18
        assert result.loc[1, "publish_hour"] == 14
        assert result.loc[2, "publish_hour"] == 20

    def test_text_features(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test text feature extraction."""
        engineer = VideoFeatureEngineer()
        result = engineer._create_text_features(sample_raw_videos.copy())

        # Check title length
        assert result.loc[0, "title_length"] > 0
        assert result.loc[0, "title_word_count"] >= 1

        # Check exam-focused detection
        assert result.loc[0, "is_exam_focused"] == 1  # Contains "Bac 2025"
        assert result.loc[1, "is_exam_focused"] == 1  # Contains "exercise"

        # Check tag count
        assert result.loc[0, "tag_count"] == 4  # math,bac,integral,calculus

    def test_subject_extraction(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test subject classification."""
        engineer = VideoFeatureEngineer()
        result = engineer._create_text_features(sample_raw_videos.copy())

        assert result.loc[0, "subject"] == "Math"
        assert result.loc[1, "subject"] == "Physics"
        assert result.loc[2, "subject"] == "Arabic"

    def test_engagement_features(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test engagement metric calculation."""
        engineer = VideoFeatureEngineer()
        result = engineer._create_engagement_features(sample_raw_videos.copy())

        # Check ratio calculations
        assert result.loc[0, "like_ratio"] == 250 / 5000
        assert result.loc[0, "comment_ratio"] == 45 / 5000

        # Check engagement score exists and is calculated
        assert "engagement_score" in result.columns
        assert result["engagement_score"].notna().all()

        # Check engagement category
        assert "engagement_category" in result.columns
        assert result["engagement_category"].isin(["Low", "Medium", "High"]).all()

    def test_channel_features(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test channel-level feature aggregation."""
        engineer = VideoFeatureEngineer(collection_date=datetime(2024, 6, 1))
        result = engineer._create_channel_features(sample_raw_videos.copy())

        # Check channel features exist
        assert "channel_video_count" in result.columns
        assert "channel_avg_views" in result.columns
        assert "channel_age_days" in result.columns

        # Channel 1 has 2 videos
        ch1_rows = result[result["channel_id"] == "ch1"]
        assert all(ch1_rows["channel_video_count"] == 2)

    def test_clean_data_removes_zero_views(self) -> None:
        """Test data cleaning removes videos with zero views."""
        df = pd.DataFrame(
            {
                "video_id": ["vid1", "vid2"],
                "view_count": [1000, 0],
                "like_count": [10, 0],
                "comment_count": [5, 0],
            }
        )

        engineer = VideoFeatureEngineer()
        result = engineer._clean_data(df)

        assert len(result) == 1
        assert result.loc[0, "video_id"] == "vid1"

    def test_select_features(self, sample_raw_videos: pd.DataFrame) -> None:
        """Test feature selection for modeling."""
        engineer = VideoFeatureEngineer()
        transformed = engineer.fit_transform(sample_raw_videos)
        selected = engineer.select_features(transformed, include_target=True)

        # Check that important features are included
        assert "video_id" in selected.columns
        assert "engagement_score" in selected.columns
        assert "engagement_category" in selected.columns
        assert "subject" in selected.columns

        # Check that target can be excluded
        selected_no_target = engineer.select_features(transformed, include_target=False)
        assert "engagement_category" not in selected_no_target.columns
