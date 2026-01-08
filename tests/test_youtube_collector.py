"""Tests for YouTube data collection module."""

from unittest.mock import Mock, patch

import pandas as pd
import pytest

from src.data.youtube_collector import YouTubeCollector


class TestYouTubeCollector:
    """Test suite for YouTubeCollector class."""
    
    def test_init_valid_api_key(self) -> None:
        """Test initialization with valid API key."""
        with patch('src.data.youtube_collector.build'):
            collector = YouTubeCollector('valid_api_key')
            assert collector.api_key == 'valid_api_key'
            assert collector.quota_used == 0
    
    def test_init_empty_api_key(self) -> None:
        """Test initialization with empty API key raises ValueError."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            YouTubeCollector('')
    
    def test_convert_iso_duration_standard(self) -> None:
        """Test ISO duration conversion for standard formats."""
        assert YouTubeCollector._convert_iso_duration('PT5M32S') == 332
        assert YouTubeCollector._convert_iso_duration('PT1H15M30S') == 4530
        assert YouTubeCollector._convert_iso_duration('PT0S') == 0
        assert YouTubeCollector._convert_iso_duration('PT10M') == 600
        assert YouTubeCollector._convert_iso_duration('PT2H') == 7200
    
    def test_convert_iso_duration_invalid(self) -> None:
        """Test ISO duration conversion handles invalid input."""
        assert YouTubeCollector._convert_iso_duration('invalid') == 0
        assert YouTubeCollector._convert_iso_duration('') == 0
    
    @patch('src.data.youtube_collector.build')
    def test_get_video_metadata_success(
        self,
        mock_build: Mock,
        mock_youtube_response: dict
    ) -> None:
        """Test successful video metadata retrieval."""
        # Setup mock
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = mock_youtube_response
        mock_build.return_value = mock_youtube
        
        collector = YouTubeCollector('test_key')
        metadata = collector.get_video_metadata('test_video_id')
        
        assert metadata is not None
        assert metadata['video_id'] == 'test_video_id'
        assert metadata['title'] == 'Test Video'
        assert metadata['view_count'] == 1000
        assert metadata['duration_sec'] == 630  # 10m30s
    
    @patch('src.data.youtube_collector.build')
    def test_get_video_metadata_not_found(self, mock_build: Mock) -> None:
        """Test video metadata retrieval when video not found."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = {'items': []}
        mock_build.return_value = mock_youtube
        
        collector = YouTubeCollector('test_key')
        metadata = collector.get_video_metadata('nonexistent_video')
        
        assert metadata is None
    
    @patch('src.data.youtube_collector.build')
    def test_collect_from_channels_missing_column(self, mock_build: Mock) -> None:
        """Test collection fails with missing channel_id column."""
        mock_youtube = Mock()
        mock_build.return_value = mock_youtube
        
        collector = YouTubeCollector('test_key')
        invalid_df = pd.DataFrame({'name': ['Channel 1']})
        
        with pytest.raises(ValueError, match="must contain 'channel_id' column"):
            collector.collect_from_channels(invalid_df)
    
    @patch('src.data.youtube_collector.build')
    def test_quota_tracking(self, mock_build: Mock) -> None:
        """Test that quota is tracked correctly."""
        mock_youtube = Mock()
        mock_youtube.videos().list().execute.return_value = {
            'items': [{
                'id': 'vid1',
                'snippet': {
                    'title': 'Test',
                    'description': '',
                    'publishedAt': '2024-01-01T00:00:00Z',
                    'channelId': 'ch1',
                    'channelTitle': 'Channel',
                    'categoryId': '27',
                    'thumbnails': {'default': {'url': 'url'}}
                },
                'statistics': {'viewCount': '100', 'likeCount': '10', 'commentCount': '1'},
                'contentDetails': {'duration': 'PT5M'}
            }]
        }
        mock_build.return_value = mock_youtube
        
        collector = YouTubeCollector('test_key')
        initial_quota = collector.quota_used
        
        collector.get_video_metadata('test_id')
        
        assert collector.quota_used == initial_quota + 1
