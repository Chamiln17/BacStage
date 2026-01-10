"""Tests for YouTube data collection module."""

from datetime import datetime
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.youtube_collector import YouTubeCollector


class TestYouTubeCollector:
    """Test suite for YouTubeCollector class."""

    def test_init_valid_api_key(self) -> None:
        """Test initialization with valid API key."""
        with patch("src.data.youtube_collector.build"):
            collector = YouTubeCollector("valid_api_key")
            assert collector.api_key == "valid_api_key"
            assert collector.quota_used == 0

    def test_init_empty_api_key(self) -> None:
        """Test initialization with empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            YouTubeCollector("")

    def test_convert_iso_duration_standard(self) -> None:
        """Test ISO duration conversion for standard formats."""
        assert YouTubeCollector._convert_iso_duration("PT5M32S") == 332
        assert YouTubeCollector._convert_iso_duration("PT1H15M30S") == 4530
        assert YouTubeCollector._convert_iso_duration("PT0S") == 0
        assert YouTubeCollector._convert_iso_duration("PT10M") == 600
        assert YouTubeCollector._convert_iso_duration("PT2H") == 7200

    def test_convert_iso_duration_invalid(self) -> None:
        """Test ISO duration conversion handles invalid input."""
        assert YouTubeCollector._convert_iso_duration("invalid") == 0
        assert YouTubeCollector._convert_iso_duration("") == 0

    @patch("src.data.youtube_collector.build")
    def test_get_video_metadata_success(
        self, mock_build: Mock, mock_youtube_response: dict
    ) -> None:
        """Test successful video metadata retrieval."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = mock_youtube_response
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        metadata = collector.get_video_metadata("test_video_id")

        assert metadata is not None
        assert metadata["video_id"] == "test_video_id"
        assert metadata["title"] == "Test Video"
        assert metadata["view_count"] == 1000
        assert metadata["duration_sec"] == 630  # 10m30s

    @patch("src.data.youtube_collector.build")
    def test_get_video_metadata_not_found(self, mock_build: Mock) -> None:
        """Test video metadata retrieval when video not found."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = {"items": []}
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        metadata = collector.get_video_metadata("nonexistent_video")

        assert metadata is None

    @patch("src.data.youtube_collector.build")
    def test_collect_from_channels_missing_column(self, mock_build: Mock) -> None:
        """Test collection fails with missing channel_id column."""
        mock_youtube = Mock()
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        invalid_df = pd.DataFrame({"name": ["Channel 1"]})

        with pytest.raises(ValueError, match="must contain 'channel_id' column"):
            collector.collect_from_channels(invalid_df)

    @patch("src.data.youtube_collector.build")
    def test_quota_tracking(self, mock_build: Mock) -> None:
        """Test that quota is tracked correctly."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = {
            "items": [
                {
                    "id": "vid1",
                    "snippet": {
                        "title": "Test",
                        "description": "",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "channelId": "ch1",
                        "channelTitle": "Channel",
                        "categoryId": "27",
                        "thumbnails": {"default": {"url": "url"}},
                    },
                    "statistics": {
                        "viewCount": "100",
                        "likeCount": "10",
                        "commentCount": "1",
                    },
                    "contentDetails": {"duration": "PT5M"},
                }
            ]
        }
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        initial_quota = collector.quota_used

        collector.get_video_metadata("test_id")

        assert collector.quota_used == initial_quota + 1


class TestUploadsPlaylistDiscovery:
    """Test suite for uploads playlist discovery methods."""

    @patch("src.data.youtube_collector.build")
    def test_get_uploads_playlist_id_success(self, mock_build: Mock) -> None:
        """Test successful uploads playlist ID retrieval."""
        mock_youtube = Mock()
        mock_youtube.channels().list().execute.return_value = {
            "items": [
                {
                    "id": "UC123",
                    "contentDetails": {
                        "relatedPlaylists": {
                            "uploads": "UU123"
                        }
                    }
                }
            ]
        }
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        playlist_id = collector.get_uploads_playlist_id("UC123")

        assert playlist_id == "UU123"
        assert collector.quota_used == 1  # channels.list costs 1 unit

    @patch("src.data.youtube_collector.build")
    def test_get_uploads_playlist_id_not_found(self, mock_build: Mock) -> None:
        """Test uploads playlist ID when channel not found."""
        mock_youtube = Mock()
        mock_youtube.channels().list().execute.return_value = {"items": []}
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        playlist_id = collector.get_uploads_playlist_id("nonexistent")

        assert playlist_id is None

    @patch("src.data.youtube_collector.build")
    def test_get_playlist_videos(self, mock_build: Mock) -> None:
        """Test playlist video enumeration."""
        mock_youtube = Mock()
        mock_youtube.playlistItems().list().execute.return_value = {
            "items": [
                {"contentDetails": {"videoId": "vid1"}},
                {"contentDetails": {"videoId": "vid2"}},
                {"contentDetails": {"videoId": "vid3"}},
            ],
            "nextPageToken": None
        }
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        video_ids = collector.get_playlist_videos("UU123")

        assert video_ids == ["vid1", "vid2", "vid3"]
        assert collector.quota_used == 1  # playlistItems.list costs 1 unit

    @patch("src.data.youtube_collector.build")
    def test_get_playlist_videos_with_max_limit(self, mock_build: Mock) -> None:
        """Test playlist video enumeration respects max_videos limit."""
        mock_youtube = Mock()
        mock_youtube.playlistItems().list().execute.return_value = {
            "items": [
                {"contentDetails": {"videoId": f"vid{i}"}}
                for i in range(50)
            ],
            "nextPageToken": "token123"
        }
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        video_ids = collector.get_playlist_videos("UU123", max_videos=5)

        assert len(video_ids) == 5
        assert video_ids == ["vid0", "vid1", "vid2", "vid3", "vid4"]


class TestBatchedEnrichment:
    """Test suite for batched videos.list enrichment."""

    @patch("src.data.youtube_collector.build")
    def test_get_videos_metadata_batch(self, mock_build: Mock) -> None:
        """Test batched video metadata retrieval."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = {
            "items": [
                {
                    "id": "vid1",
                    "snippet": {
                        "title": "Video 1",
                        "description": "Desc 1",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "channelId": "ch1",
                        "channelTitle": "Channel",
                        "categoryId": "27",
                        "thumbnails": {"default": {"url": "url1"}},
                    },
                    "statistics": {
                        "viewCount": "100",
                        "likeCount": "10",
                        "commentCount": "1",
                    },
                    "contentDetails": {"duration": "PT5M"},
                },
                {
                    "id": "vid2",
                    "snippet": {
                        "title": "Video 2",
                        "description": "Desc 2",
                        "publishedAt": "2024-01-02T00:00:00Z",
                        "channelId": "ch1",
                        "channelTitle": "Channel",
                        "categoryId": "27",
                        "thumbnails": {"default": {"url": "url2"}},
                    },
                    "statistics": {
                        "viewCount": "200",
                        "likeCount": "20",
                        "commentCount": "2",
                    },
                    "contentDetails": {"duration": "PT10M"},
                },
            ]
        }
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        snapshot_date = datetime(2024, 1, 15)
        
        metadata = collector.get_videos_metadata_batch(
            ["vid1", "vid2"],
            snapshot_date=snapshot_date,
            run_id="test123"
        )

        assert len(metadata) == 2
        assert metadata[0]["video_id"] == "vid1"
        assert metadata[0]["title"] == "Video 1"
        assert metadata[0]["view_count"] == 100
        assert metadata[0]["snapshot_date"] == snapshot_date.isoformat()
        assert metadata[0]["run_id"] == "test123"
        
        assert metadata[1]["video_id"] == "vid2"
        assert metadata[1]["view_count"] == 200
        
        # Only 1 API request for both videos (batched)
        assert collector.quota_used == 1

    @patch("src.data.youtube_collector.build")
    def test_batch_quota_efficiency(self, mock_build: Mock) -> None:
        """Test that batching is 50x more efficient than single requests."""
        mock_youtube = Mock()
        # Mock response with 50 videos
        mock_youtube.videos().list().execute.return_value = {
            "items": [
                {
                    "id": f"vid{i}",
                    "snippet": {
                        "title": f"Video {i}",
                        "description": "",
                        "publishedAt": "2024-01-01T00:00:00Z",
                        "channelId": "ch1",
                        "channelTitle": "Channel",
                        "categoryId": "27",
                        "thumbnails": {},
                    },
                    "statistics": {
                        "viewCount": "100",
                        "likeCount": "10",
                        "commentCount": "1",
                    },
                    "contentDetails": {"duration": "PT5M"},
                }
                for i in range(50)
            ]
        }
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        
        # Fetch 50 videos in a batch
        video_ids = [f"vid{i}" for i in range(50)]
        metadata = collector.get_videos_metadata_batch(video_ids)

        assert len(metadata) == 50
        # Only 1 API request for 50 videos
        assert collector.quota_used == 1

    @patch("src.data.youtube_collector.build")
    def test_batch_multiple_batches(self, mock_build: Mock) -> None:
        """Test that large requests are split into multiple batches."""
        mock_youtube = Mock()
        
        # Return different results for each batch
        def mock_execute():
            return {
                "items": [
                    {
                        "id": f"vid{i}",
                        "snippet": {
                            "title": f"Video {i}",
                            "description": "",
                            "publishedAt": "2024-01-01T00:00:00Z",
                            "channelId": "ch1",
                            "channelTitle": "Channel",
                            "categoryId": "27",
                            "thumbnails": {},
                        },
                        "statistics": {
                            "viewCount": "100",
                            "likeCount": "10",
                            "commentCount": "1",
                        },
                        "contentDetails": {"duration": "PT5M"},
                    }
                    for i in range(50)
                ]
            }
        
        mock_youtube.videos().list().execute.side_effect = mock_execute
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        
        # Request 100 videos (should be 2 batches)
        video_ids = [f"vid{i}" for i in range(100)]
        collector.get_videos_metadata_batch(video_ids)

        # 2 API requests for 100 videos (50 each)
        assert collector.quota_used == 2

    @patch("src.data.youtube_collector.build")
    def test_empty_batch(self, mock_build: Mock) -> None:
        """Test empty video list returns empty result."""
        mock_youtube = Mock()
        mock_build.return_value = mock_youtube

        collector = YouTubeCollector("test_key")
        metadata = collector.get_videos_metadata_batch([])

        assert metadata == []
        assert collector.quota_used == 0  # No API calls made
