"""Pytest configuration and fixtures."""

import pandas as pd
import pytest


@pytest.fixture
def sample_raw_videos() -> pd.DataFrame:
    """Create sample raw video data for testing."""
    return pd.DataFrame(
        {
            "video_id": ["vid1", "vid2", "vid3"],
            "title": [
                "Math Bac 2025 - Integral Calculus",
                "Physics: Newton Laws Exercise",
                "Arabic Literature - Poetry Analysis",
            ],
            "description": [
                "Complete tutorial on integral calculus for Bac exam",
                "Solve physics problems about Newton laws",
                "Analysis of classical Arabic poetry",
            ],
            "publish_date": pd.to_datetime(
                [
                    "2024-01-15 18:00:00+00:00",
                    "2024-02-20 14:30:00+00:00",
                    "2024-03-10 20:15:00+00:00",
                ]
            ),
            "channel_id": ["ch1", "ch1", "ch2"],
            "channel_title": ["Math Channel", "Math Channel", "Arabic Channel"],
            "category_id": ["27", "27", "27"],
            "duration_iso": ["PT15M30S", "PT8M45S", "PT25M00S"],
            "duration_sec": [930, 525, 1500],
            "view_count": [5000, 3000, 1200],
            "like_count": [250, 180, 80],
            "comment_count": [45, 30, 15],
            "tags": [
                "math,bac,integral,calculus",
                "physics,newton,exercise",
                "arabic,literature,poetry",
            ],
            "thumbnail_url": ["url1", "url2", "url3"],
        }
    )


@pytest.fixture
def mock_youtube_response() -> dict:
    """Create mock YouTube API response."""
    return {
        "items": [
            {
                "id": "test_video_id",
                "snippet": {
                    "title": "Test Video",
                    "description": "Test description",
                    "publishedAt": "2024-01-15T18:00:00Z",
                    "channelId": "test_channel",
                    "channelTitle": "Test Channel",
                    "categoryId": "27",
                    "tags": ["test", "video"],
                    "thumbnails": {"default": {"url": "http://test.url"}},
                },
                "statistics": {
                    "viewCount": "1000",
                    "likeCount": "50",
                    "commentCount": "10",
                },
                "contentDetails": {"duration": "PT10M30S"},
            }
        ]
    }
