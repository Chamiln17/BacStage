"""
Video registry manager for tracking known video IDs.

This module maintains a CSV-based registry of all discovered video IDs,
enabling efficient incremental updates without re-discovering videos.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Default registry path
DEFAULT_REGISTRY_PATH = Path("data/raw/video_registry.csv")

# Registry schema
REGISTRY_COLUMNS = [
    "video_id",
    "channel_id",
    "discovered_at",
    "last_seen_at",
    "source",
]


class VideoRegistry:
    """
    Manages a CSV-based registry of known video IDs.

    The registry tracks:
    - video_id: YouTube video ID
    - channel_id: Channel that uploaded the video
    - discovered_at: When the video was first discovered
    - last_seen_at: When the video was last seen in uploads playlist
    - source: How the video was discovered (uploads_playlist, bootstrap_csv, search)

    Usage:
        registry = VideoRegistry()
        registry.load()
        registry.add_videos(video_ids, channel_id, source="uploads_playlist")
        registry.save()
    """

    def __init__(self, path: Optional[Path] = None) -> None:
        """
        Initialize the video registry.

        Args:
            path: Path to the registry CSV file.
                  Defaults to data/raw/video_registry.csv
        """
        self.path = path or DEFAULT_REGISTRY_PATH
        self._data: Optional[pd.DataFrame] = None

    @property
    def data(self) -> pd.DataFrame:
        """Get the registry DataFrame, loading if necessary."""
        if self._data is None:
            self.load()
        return self._data

    def load(self) -> pd.DataFrame:
        """
        Load the registry from disk.

        Returns:
            DataFrame with registry data

        Note:
            Creates an empty registry if file doesn't exist.
        """
        if self.path.exists():
            try:
                self._data = pd.read_csv(self.path)
                # Ensure all columns exist
                for col in REGISTRY_COLUMNS:
                    if col not in self._data.columns:
                        self._data[col] = ""
                logger.info(
                    f"Loaded registry with {len(self._data)} videos from {self.path}"
                )
            except Exception as e:
                logger.error(f"Error loading registry: {e}")
                self._data = self._create_empty_registry()
        else:
            logger.info(f"Registry not found at {self.path}, creating new one")
            self._data = self._create_empty_registry()

        return self._data

    def save(self) -> None:
        """Save the registry to disk."""
        if self._data is None:
            logger.warning("No data to save")
            return

        # Ensure directory exists
        self.path.parent.mkdir(parents=True, exist_ok=True)

        self._data.to_csv(self.path, index=False)
        logger.info(f"Saved registry with {len(self._data)} videos to {self.path}")

    def add_videos(
        self,
        video_ids: List[str],
        channel_id: str,
        source: str = "uploads_playlist",
    ) -> int:
        """
        Add or update videos in the registry.

        Args:
            video_ids: List of video IDs to add
            channel_id: Channel ID these videos belong to
            source: How the videos were discovered

        Returns:
            Number of new videos added (not updates)
        """
        if self._data is None:
            self.load()

        now = datetime.now().isoformat()
        existing_ids = set(self._data["video_id"].tolist())
        new_count = 0

        new_records = []
        for video_id in video_ids:
            if video_id in existing_ids:
                # Update last_seen_at for existing videos
                self._data.loc[self._data["video_id"] == video_id, "last_seen_at"] = now
            else:
                # Add new video
                new_records.append(
                    {
                        "video_id": video_id,
                        "channel_id": channel_id,
                        "discovered_at": now,
                        "last_seen_at": now,
                        "source": source,
                    }
                )
                new_count += 1

        if new_records:
            new_df = pd.DataFrame(new_records)
            self._data = pd.concat([self._data, new_df], ignore_index=True)

        logger.info(
            f"Added {new_count} new videos, updated {len(video_ids) - new_count} existing"
        )
        return new_count

    def get_all_video_ids(self) -> List[str]:
        """
        Get all video IDs in the registry.

        Returns:
            List of all video IDs
        """
        return self.data["video_id"].tolist()

    def get_video_ids_for_channel(self, channel_id: str) -> List[str]:
        """
        Get all video IDs for a specific channel.

        Args:
            channel_id: YouTube channel ID

        Returns:
            List of video IDs for the channel
        """
        mask = self.data["channel_id"] == channel_id
        return self.data.loc[mask, "video_id"].tolist()

    def get_ids_for_enrichment(
        self,
        channel_ids: Optional[List[str]] = None,
        limit: Optional[int] = None,
    ) -> List[str]:
        """
        Get video IDs that need enrichment.

        Args:
            channel_ids: Optional list of channels to filter by
            limit: Maximum number of IDs to return

        Returns:
            List of video IDs to enrich
        """
        df = self.data

        if channel_ids:
            df = df[df["channel_id"].isin(channel_ids)]

        video_ids = df["video_id"].tolist()

        if limit:
            video_ids = video_ids[:limit]

        return video_ids

    def get_unique_channels(self) -> List[str]:
        """Get list of unique channel IDs in the registry."""
        return self.data["channel_id"].unique().tolist()

    def count(self) -> int:
        """Get total number of videos in registry."""
        return len(self.data)

    def count_by_channel(self) -> pd.Series:
        """Get video count per channel."""
        return self.data.groupby("channel_id").size()

    def bootstrap_from_csv(self, videos_csv_path: Path) -> int:
        """
        Bootstrap the registry from an existing videos_metadata.csv file.

        This is useful for migrating from the old system without re-discovering videos.

        Args:
            videos_csv_path: Path to existing videos_metadata.csv

        Returns:
            Number of videos added to registry
        """
        if not videos_csv_path.exists():
            logger.warning(f"CSV file not found: {videos_csv_path}")
            return 0

        try:
            existing_df = pd.read_csv(videos_csv_path)
        except Exception as e:
            logger.error(f"Error reading CSV: {e}")
            return 0

        if "video_id" not in existing_df.columns:
            logger.error("CSV missing required 'video_id' column")
            return 0

        # Group by channel and add to registry
        total_added = 0
        for channel_id in existing_df["channel_id"].unique():
            channel_videos = existing_df[existing_df["channel_id"] == channel_id][
                "video_id"
            ].tolist()
            added = self.add_videos(channel_videos, channel_id, source="bootstrap_csv")
            total_added += added

        logger.info(f"Bootstrapped {total_added} videos from {videos_csv_path}")
        return total_added

    def _create_empty_registry(self) -> pd.DataFrame:
        """Create an empty registry DataFrame with proper schema."""
        return pd.DataFrame(columns=REGISTRY_COLUMNS)

    def __len__(self) -> int:
        """Return number of videos in registry."""
        return self.count()

    def __contains__(self, video_id: str) -> bool:
        """Check if a video ID is in the registry."""
        return video_id in set(self.data["video_id"].tolist())
