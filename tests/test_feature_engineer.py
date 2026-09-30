"""Tests for VideoFeatureEngineer through its fit / transform interface."""

import pickle
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src.features.engineer import (
    VideoFeatureEngineer,
    engagement_category,
    engagement_score,
    engineered_export,
)

REFERENCE = datetime(2025, 1, 1, tzinfo=timezone.utc)
STATS = ["view_count", "like_count", "comment_count"]


@pytest.fixture
def fitted(training_videos: pd.DataFrame) -> VideoFeatureEngineer:
    return VideoFeatureEngineer(reference_date=REFERENCE).fit(training_videos)


def test_transform_shape_matches_feature_names(fitted, training_videos) -> None:
    X = fitted.transform(training_videos)
    assert X.shape == (len(training_videos), len(fitted.feature_names))
    assert np.isfinite(X).all()


def test_transform_ignores_statistics(fitted, training_videos) -> None:
    """No target leakage: output is identical with stats present, NaN, or absent."""
    with_stats = fitted.transform(training_videos)
    nan_stats = fitted.transform(training_videos.assign(**dict.fromkeys(STATS, np.nan)))
    no_stats = fitted.transform(training_videos.drop(columns=STATS))
    np.testing.assert_array_equal(with_stats, nan_stats)
    np.testing.assert_array_equal(with_stats, no_stats)


def test_planned_video_minimal_fields(fitted) -> None:
    planned = pd.DataFrame([{"title": "مراجعة بكالوريا التكامل", "duration_sec": 900, "channel_id": "UC_math"}])
    X = fitted.transform(planned)
    assert X.shape == (1, len(fitted.feature_names))
    assert fitted.known_channel(planned).tolist() == [True]


def test_subject_comes_from_channel_table_when_missing(fitted) -> None:
    planned = pd.DataFrame([{"title": "درس", "duration_sec": 600, "channel_id": "UC_bio"}])
    assert fitted.video_features(planned)["subject"].iat[0] == "Natural Sciences"


def test_unknown_channel_uses_training_medians(fitted) -> None:
    planned = pd.DataFrame([{"title": "درس", "duration_sec": 600, "channel_id": "UC_new", "subject": "Maths"}])
    row = fitted.video_features(planned).iloc[0]
    assert fitted.known_channel(planned).tolist() == [False]
    assert row["channel_avg_views"] == fitted.channel_defaults["channel_avg_views"]


def test_channel_table_learned_from_fit_rows_only(training_videos) -> None:
    half = training_videos.iloc[:30]
    engineer = VideoFeatureEngineer(reference_date=REFERENCE).fit(half)
    expected = half.groupby("channel_id")["view_count"].mean()
    pd.testing.assert_series_equal(
        engineer.channel_table["channel_avg_views"], expected, check_names=False
    )


def test_has_transcript_counts_only_valid_transcripts(fitted) -> None:
    videos = pd.DataFrame(
        {
            "title": ["a", "b", "c"],
            "duration_sec": [600, 600, 600],
            "channel_id": ["UC_math"] * 3,
            "transcript_text": ["نشرح الدالة الأسية", 'window.x=1; var ytcfg={}', None],
        }
    )
    features = fitted.video_features(videos)
    assert features["has_transcript"].tolist() == [1, 0, 0]
    assert features["transcript_word_count"].tolist() == [3, 0, 0]


def test_requires_title_duration_and_channel_or_subject(fitted) -> None:
    with pytest.raises(ValueError, match="duration_sec"):
        fitted.transform(pd.DataFrame([{"title": "x", "channel_id": "UC_math"}]))
    with pytest.raises(ValueError, match="channel_id or a subject"):
        fitted.transform(pd.DataFrame([{"title": "x", "duration_sec": 60}]))


def test_transform_before_fit_raises(training_videos) -> None:
    with pytest.raises(RuntimeError, match="not fitted"):
        VideoFeatureEngineer().transform(training_videos)


def test_embeddings_need_an_embedder_and_are_not_pickled(training_videos, fake_embedder) -> None:
    engineer = VideoFeatureEngineer(fake_embedder, REFERENCE).fit(training_videos)
    X = engineer.transform(training_videos)
    assert engineer.uses_embeddings
    assert X.shape[1] == len(engineer.feature_names)

    restored = pickle.loads(pickle.dumps(engineer))
    assert restored.embedder is None
    with pytest.raises(RuntimeError, match="embedder"):
        restored.transform(training_videos)
    restored.embedder = fake_embedder
    np.testing.assert_allclose(restored.transform(training_videos), X)


def test_engagement_score_and_category() -> None:
    videos = pd.DataFrame({"view_count": [100, 0], "like_count": [10, 5], "comment_count": [2, 0]})
    scores = engagement_score(videos)
    assert scores.iat[0] == pytest.approx(np.log1p((3 * 2 + 10) / 10))
    assert scores.iat[1] == pytest.approx(np.log1p(5))  # zero views treated as one
    assert engagement_category(pd.Series([0.1, 0.5, 0.9]), [0.3, 0.7]).tolist() == ["Low", "Medium", "High"]


def test_engineered_export(training_videos) -> None:
    export = engineered_export(training_videos, REFERENCE)
    assert len(export) == len(training_videos)
    assert {"engagement_score", "engagement_category", "has_transcript", "subject"} <= set(export.columns)
    assert export["has_transcript"].sum() == training_videos["transcript_text"].notna().sum()
