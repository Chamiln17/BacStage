"""
YouTube data collection module.

This module provides classes and functions for collecting video metadata from
YouTube channels using the YouTube Data API v3.
"""

import logging
import re
import time
from typing import Dict, List, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class YouTubeCollector:
    """
    YouTube Data API collector for channel video metadata.

    This class handles all interactions with the YouTube Data API v3 to collect
    video metadata from specified channels.

    Attributes:
        api_key: YouTube Data API v3 key
        youtube: Google API client instance
        quota_used: Tracking variable for API quota usage
    """

    def __init__(self, api_key: str) -> None:
        """
        Initialize YouTube collector with API credentials.

        Args:
            api_key: YouTube Data API v3 key

        Raises:
            ValueError: If API key is empty or invalid
            Exception: If YouTube API client initialization fails
        """
        if not api_key or not api_key.strip():
            raise ValueError("API key cannot be empty")

        self.api_key = api_key
        self.quota_used = 0

        try:
            self.youtube = build("youtube", "v3", developerKey=api_key)
            logger.info("YouTube API client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize YouTube API client: {e}")
            raise

    def get_channel_videos(
        self,
        channel_id: str,
        max_videos: Optional[int] = None,
        max_results_per_page: int = 50,
    ) -> List[str]:
        """
        Get all video IDs from a YouTube channel.

        Args:
            channel_id: YouTube channel ID (e.g., 'UCxxxxxx')
            max_videos: Maximum number of videos to retrieve (None = all)
            max_results_per_page: Videos per API request (max 50)

        Returns:
            List of video IDs from the channel

        Raises:
            HttpError: If API request fails

        Note:
            API quota cost: ~100 units per search.list request
        """
        video_ids: List[str] = []
        next_page_token: Optional[str] = None
        request_count = 0

        logger.info(f"Fetching videos from channel {channel_id}")

        try:
            while True:
                # Check max_videos limit
                if max_videos and len(video_ids) >= max_videos:
                    break

                # Calculate results to fetch for this request
                results_to_fetch = max_results_per_page
                if max_videos:
                    results_to_fetch = min(
                        max_results_per_page, max_videos - len(video_ids)
                    )

                # Make API request
                request = self.youtube.search().list(
                    part="id",
                    channelId=channel_id,
                    maxResults=results_to_fetch,
                    pageToken=next_page_token,
                    order="date",
                    type="video",
                )
                response = request.execute()
                request_count += 1
                self.quota_used += 100  # search.list costs ~100 units

                # Extract video IDs
                for item in response.get("items", []):
                    if "videoId" in item["id"]:
                        video_ids.append(item["id"]["videoId"])

                # Handle pagination
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                # Rate limiting
                time.sleep(0.1)

                # Progress logging
                if len(video_ids) % 50 == 0:
                    logger.debug(f"Retrieved {len(video_ids)} video IDs so far")

            logger.info(
                f"Retrieved {len(video_ids)} video IDs from channel "
                f"({request_count} API requests)"
            )
            return video_ids

        except HttpError as e:
            logger.error(f"HTTP error fetching videos from channel {channel_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching videos from channel {channel_id}: {e}")
            raise

    def get_video_metadata(self, video_id: str) -> Optional[Dict]:
        """
        Fetch detailed metadata for a single video.

        Args:
            video_id: YouTube video ID

        Returns:
            Dictionary containing video metadata with keys:
                - video_id, title, description, publish_date
                - channel_id, channel_title, category_id
                - duration_iso, duration_sec
                - view_count, like_count, comment_count
                - tags, thumbnail_url
            Returns None if video not found or error occurs

        Raises:
            HttpError: If API request fails

        Note:
            API quota cost: ~1 unit per videos.list request
        """
        try:
            request = self.youtube.videos().list(
                part="snippet,statistics,contentDetails", id=video_id
            )
            response = request.execute()
            self.quota_used += 1  # videos.list costs ~1 unit

            if not response.get("items"):
                logger.warning(f"Video {video_id} not found or unavailable")
                return None

            video = response["items"][0]

            # Extract metadata
            metadata = {
                "video_id": video_id,
                "title": video["snippet"].get("title", ""),
                "description": video["snippet"].get("description", ""),
                "publish_date": video["snippet"].get("publishedAt", ""),
                "channel_id": video["snippet"].get("channelId", ""),
                "channel_title": video["snippet"].get("channelTitle", ""),
                "category_id": video["snippet"].get("categoryId", ""),
                "duration_iso": video["contentDetails"].get("duration", "PT0S"),
                "view_count": int(video["statistics"].get("viewCount", 0)),
                "like_count": int(video["statistics"].get("likeCount", 0)),
                "comment_count": int(video["statistics"].get("commentCount", 0)),
                "tags": ",".join(video["snippet"].get("tags", [])),
                "thumbnail_url": (
                    video["snippet"]["thumbnails"]["default"]["url"]
                    if "thumbnails" in video["snippet"]
                    else ""
                ),
            }

            # Convert ISO duration to seconds
            metadata["duration_sec"] = self._convert_iso_duration(
                metadata["duration_iso"]
            )

            return metadata

        except HttpError as e:
            logger.error(f"HTTP error fetching metadata for video {video_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching metadata for video {video_id}: {e}")
            return None

    @staticmethod
    def _convert_iso_duration(iso_duration: str) -> int:
        """
        Convert ISO 8601 duration format to seconds.

        Args:
            iso_duration: ISO 8601 duration string (e.g., 'PT5M32S', 'PT1H15M30S')

        Returns:
            Duration in seconds

        Examples:
            >>> YouTubeCollector._convert_iso_duration('PT5M32S')
            332
            >>> YouTubeCollector._convert_iso_duration('PT1H15M30S')
            4530
            >>> YouTubeCollector._convert_iso_duration('PT0S')
            0
        """
        pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
        match = re.match(pattern, iso_duration)

        if match:
            hours, minutes, seconds = match.groups()
            hours = int(hours) if hours else 0
            minutes = int(minutes) if minutes else 0
            seconds = int(seconds) if seconds else 0
            return hours * 3600 + minutes * 60 + seconds

        return 0

    def collect_from_channels(
        self,
        channels_df: pd.DataFrame,
        max_videos_per_channel: Optional[int] = None,
        max_quota: int = 8000,
    ) -> pd.DataFrame:
        """
        Collect video metadata from multiple channels.

        Args:
            channels_df: DataFrame with 'channel_id' column (and optional 'channel_name')
            max_videos_per_channel: Maximum videos per channel (None = all)
            max_quota: Maximum API quota units to use

        Returns:
            DataFrame with collected video metadata

        Raises:
            ValueError: If channels_df missing 'channel_id' column

        Note:
            Total API quota usage: ~100 units per video (search) + 1 unit per video (metadata)
        """
        if "channel_id" not in channels_df.columns:
            raise ValueError("channels_df must contain 'channel_id' column")

        all_videos: List[Dict] = []
        total_channels = len(channels_df)

        logger.info(f"Starting collection from {total_channels} channels")
        logger.info(f"Max videos per channel: {max_videos_per_channel or 'All'}")
        logger.info(f"Quota limit: {max_quota} units")

        for idx, channel_row in channels_df.iterrows():
            channel_id = channel_row["channel_id"]
            channel_name = channel_row.get("channel_name", f"Channel_{idx+1}")

            logger.info(f"[{idx+1}/{total_channels}] Processing: {channel_name}")

            # Check quota before starting
            if self.quota_used >= max_quota:
                logger.warning(
                    f"Quota limit reached ({self.quota_used}/{max_quota}). "
                    "Stopping collection."
                )
                break

            # Get video IDs
            try:
                video_ids = self.get_channel_videos(
                    channel_id, max_videos=max_videos_per_channel
                )
            except Exception as e:
                logger.error(f"Failed to get videos from {channel_name}: {e}")
                continue

            if not video_ids:
                logger.warning(f"No videos found for {channel_name}")
                continue

            # Fetch metadata for each video
            logger.info(f"Fetching metadata for {len(video_ids)} videos")
            videos_collected = 0

            for video_idx, video_id in enumerate(video_ids, 1):
                # Check quota
                if self.quota_used >= max_quota:
                    logger.warning(
                        f"Quota limit reached at video {video_idx}/{len(video_ids)}"
                    )
                    break

                try:
                    metadata = self.get_video_metadata(video_id)
                    if metadata:
                        all_videos.append(metadata)
                        videos_collected += 1
                except Exception as e:
                    logger.error(f"Failed to get metadata for video {video_id}: {e}")
                    continue

                # Rate limiting
                time.sleep(0.05)

                # Progress logging every 10 videos
                if video_idx % 10 == 0:
                    logger.debug(
                        f"Progress: {video_idx}/{len(video_ids)} videos "
                        f"({self.quota_used} quota used)"
                    )

            logger.info(
                f"Collected {videos_collected} videos from {channel_name} "
                f"(Total quota: {self.quota_used}/{max_quota})"
            )

            if self.quota_used >= max_quota:
                break

        # Create DataFrame
        if all_videos:
            videos_df = pd.DataFrame(all_videos)
            videos_df["publish_date"] = pd.to_datetime(videos_df["publish_date"])

            logger.info(
                f"Collection complete: {len(videos_df)} videos from "
                f"{videos_df['channel_id'].nunique()} channels"
            )
            logger.info(f"Total API quota used: {self.quota_used}")

            return videos_df
        else:
            logger.warning("No videos collected")
            return pd.DataFrame()
