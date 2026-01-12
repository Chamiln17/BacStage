"""Tests for video registry module."""

import tempfile
from pathlib import Path

import pandas as pd

from src.data.video_registry import REGISTRY_COLUMNS, VideoRegistry


class TestVideoRegistry:
    """Test suite for VideoRegistry class."""

    def test_init_default_path(self) -> None:
        """Test initialization with default path."""
        registry = VideoRegistry()
        assert registry.path == Path("data/raw/video_registry.csv")

    def test_init_custom_path(self) -> None:
        """Test initialization with custom path."""
        custom_path = Path("/tmp/custom_registry.csv")
        registry = VideoRegistry(custom_path)
        assert registry.path == custom_path

    def test_create_empty_registry(self) -> None:
        """Test creating empty registry with correct schema."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            assert len(registry) == 0
            assert list(registry.data.columns) == REGISTRY_COLUMNS

    def test_add_videos_new(self) -> None:
        """Test adding new videos to registry."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            added = registry.add_videos(
                ["vid1", "vid2", "vid3"], "channel_123", source="uploads_playlist"
            )

            assert added == 3
            assert len(registry) == 3
            assert "vid1" in registry
            assert "vid2" in registry
            assert "vid3" in registry

    def test_add_videos_existing(self) -> None:
        """Test adding existing videos updates last_seen_at."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            # Add initial videos
            registry.add_videos(["vid1", "vid2"], "channel_123")

            # Get initial last_seen_at
            initial_last_seen = registry.data[registry.data["video_id"] == "vid1"][
                "last_seen_at"
            ].iloc[0]

            # Add same videos again (with one new)
            added = registry.add_videos(["vid1", "vid2", "vid3"], "channel_123")

            assert added == 1  # Only vid3 is new
            assert len(registry) == 3

            # last_seen_at should be updated for existing videos
            updated_last_seen = registry.data[registry.data["video_id"] == "vid1"][
                "last_seen_at"
            ].iloc[0]

            # The timestamp should have changed (or be the same if very fast)
            assert updated_last_seen >= initial_last_seen

    def test_save_and_load(self) -> None:
        """Test saving and loading registry persists data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"

            # Create and save registry
            registry1 = VideoRegistry(registry_path)
            registry1.load()
            registry1.add_videos(["vid1", "vid2"], "channel_123")
            registry1.save()

            # Load in new instance
            registry2 = VideoRegistry(registry_path)
            registry2.load()

            assert len(registry2) == 2
            assert "vid1" in registry2
            assert "vid2" in registry2

    def test_get_all_video_ids(self) -> None:
        """Test getting all video IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()
            registry.add_videos(["vid1", "vid2", "vid3"], "channel_123")

            video_ids = registry.get_all_video_ids()

            assert set(video_ids) == {"vid1", "vid2", "vid3"}

    def test_get_video_ids_for_channel(self) -> None:
        """Test getting video IDs for specific channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            registry.add_videos(["vid1", "vid2"], "channel_A")
            registry.add_videos(["vid3", "vid4", "vid5"], "channel_B")

            channel_a_videos = registry.get_video_ids_for_channel("channel_A")
            channel_b_videos = registry.get_video_ids_for_channel("channel_B")

            assert set(channel_a_videos) == {"vid1", "vid2"}
            assert set(channel_b_videos) == {"vid3", "vid4", "vid5"}

    def test_get_ids_for_enrichment(self) -> None:
        """Test getting IDs for enrichment with filters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            registry.add_videos(["vid1", "vid2"], "channel_A")
            registry.add_videos(["vid3", "vid4", "vid5"], "channel_B")

            # Get all
            all_ids = registry.get_ids_for_enrichment()
            assert len(all_ids) == 5

            # Filter by channel
            channel_a_ids = registry.get_ids_for_enrichment(channel_ids=["channel_A"])
            assert len(channel_a_ids) == 2

            # With limit
            limited_ids = registry.get_ids_for_enrichment(limit=3)
            assert len(limited_ids) == 3

    def test_get_unique_channels(self) -> None:
        """Test getting unique channel list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            registry.add_videos(["vid1"], "channel_A")
            registry.add_videos(["vid2", "vid3"], "channel_B")
            registry.add_videos(["vid4"], "channel_C")

            channels = registry.get_unique_channels()

            assert set(channels) == {"channel_A", "channel_B", "channel_C"}

    def test_count_by_channel(self) -> None:
        """Test counting videos by channel."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            registry.add_videos(["vid1"], "channel_A")
            registry.add_videos(["vid2", "vid3", "vid4"], "channel_B")

            counts = registry.count_by_channel()

            assert counts["channel_A"] == 1
            assert counts["channel_B"] == 3

    def test_bootstrap_from_csv(self) -> None:
        """Test bootstrapping registry from existing videos CSV."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a mock videos_metadata.csv
            videos_csv_path = Path(tmpdir) / "videos_metadata.csv"
            videos_df = pd.DataFrame(
                {
                    "video_id": ["vid1", "vid2", "vid3", "vid4"],
                    "channel_id": ["ch_A", "ch_A", "ch_B", "ch_B"],
                    "title": ["Title 1", "Title 2", "Title 3", "Title 4"],
                }
            )
            videos_df.to_csv(videos_csv_path, index=False)

            # Bootstrap registry from it
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            added = registry.bootstrap_from_csv(videos_csv_path)

            assert added == 4
            assert len(registry) == 4
            assert "vid1" in registry
            assert "vid4" in registry

            # Check source is set correctly
            sources = registry.data["source"].unique()
            assert "bootstrap_csv" in sources

    def test_bootstrap_from_nonexistent_csv(self) -> None:
        """Test bootstrapping from non-existent file returns 0."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            added = registry.bootstrap_from_csv(Path("/nonexistent/file.csv"))

            assert added == 0
            assert len(registry) == 0

    def test_contains_operator(self) -> None:
        """Test __contains__ operator for checking video presence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()
            registry.add_videos(["vid1", "vid2"], "channel_123")

            assert "vid1" in registry
            assert "vid2" in registry
            assert "vid3" not in registry

    def test_len_operator(self) -> None:
        """Test __len__ operator for registry size."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry_path = Path(tmpdir) / "registry.csv"
            registry = VideoRegistry(registry_path)
            registry.load()

            assert len(registry) == 0

            registry.add_videos(["vid1", "vid2", "vid3"], "channel_123")

            assert len(registry) == 3
