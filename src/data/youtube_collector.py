"""
YouTube data collection module.

This module provides classes and functions for collecting video metadata from
YouTube channels using the YouTube Data API v3.

Optimized for quota efficiency:
- Uses playlistItems.list (1 unit) instead of search.list (100 units) for discovery
- Batches videos.list requests (50 IDs per request) for enrichment
"""

import logging
import re
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)

# Maximum number of video IDs per videos.list request
BATCH_SIZE = 50


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

    # =========================================================================
    # DISCOVERY METHODS (Uploads Playlist - Low Cost)
    # =========================================================================

    def get_uploads_playlist_id(self, channel_id: str) -> Optional[str]:
        """
        Get the uploads playlist ID for a channel.

        Every YouTube channel has an automatically-generated uploads playlist
        that contains all public videos. This is much cheaper to enumerate
        than using search.list.

        Args:
            channel_id: YouTube channel ID (e.g., 'UCxxxxxx')

        Returns:
            Uploads playlist ID (e.g., 'UUxxxxxx'), or None if not found

        Note:
            API quota cost: 1 unit (channels.list)
        """
        try:
            request = self.youtube.channels().list(
                part="contentDetails",
                id=channel_id,
            )
            response = request.execute()
            self.quota_used += 1  # channels.list costs 1 unit

            if not response.get("items"):
                logger.warning(f"Channel {channel_id} not found")
                return None

            channel = response["items"][0]
            uploads_id = channel["contentDetails"]["relatedPlaylists"]["uploads"]

            logger.debug(f"Channel {channel_id} uploads playlist: {uploads_id}")
            return uploads_id

        except HttpError as e:
            logger.error(f"HTTP error getting uploads playlist for {channel_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting uploads playlist for {channel_id}: {e}")
            return None

    def get_playlist_videos(
        self,
        playlist_id: str,
        max_videos: Optional[int] = None,
        max_results_per_page: int = 50,
    ) -> List[str]:
        """
        Get all video IDs from a playlist using playlistItems.list.

        Args:
            playlist_id: YouTube playlist ID
            max_videos: Maximum number of videos to retrieve (None = all)
            max_results_per_page: Videos per API request (max 50)

        Returns:
            List of video IDs from the playlist

        Note:
            API quota cost: 1 unit per request (50 items max per request)
        """
        video_ids: List[str] = []
        next_page_token: Optional[str] = None
        request_count = 0

        try:
            while True:
                # Check max_videos limit
                if max_videos and len(video_ids) >= max_videos:
                    break

                # Calculate results to fetch for this request
                results_to_fetch = min(max_results_per_page, 50)
                if max_videos:
                    results_to_fetch = min(results_to_fetch, max_videos - len(video_ids))

                # Make API request
                request = self.youtube.playlistItems().list(
                    part="contentDetails",
                    playlistId=playlist_id,
                    maxResults=results_to_fetch,
                    pageToken=next_page_token,
                )
                response = request.execute()
                request_count += 1
                self.quota_used += 1  # playlistItems.list costs 1 unit

                # Extract video IDs
                for item in response.get("items", []):
                    video_id = item.get("contentDetails", {}).get("videoId")
                    if video_id:
                        video_ids.append(video_id)
                        if max_videos and len(video_ids) >= max_videos:
                            break

                # Handle pagination
                if max_videos and len(video_ids) >= max_videos:
                    break
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                # Rate limiting
                time.sleep(0.05)

            logger.debug(
                f"Retrieved {len(video_ids)} video IDs from playlist "
                f"({request_count} API requests, {request_count} quota units)"
            )
            return video_ids[:max_videos] if max_videos else video_ids

        except HttpError as e:
            logger.error(f"HTTP error fetching playlist {playlist_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error fetching playlist {playlist_id}: {e}")
            raise

    def get_channel_videos(
        self,
        channel_id: str,
        max_videos: Optional[int] = None,
        max_results_per_page: int = 50,
        use_search_fallback: bool = False,
    ) -> List[str]:
        """
        Get all video IDs from a YouTube channel.

        Uses the uploads playlist method by default (1 unit per 50 videos).
        Falls back to search.list if playlist method fails.

        Args:
            channel_id: YouTube channel ID (e.g., 'UCxxxxxx')
            max_videos: Maximum number of videos to retrieve (None = all)
            max_results_per_page: Videos per API request (max 50)
            use_search_fallback: Force use of search.list (not recommended)

        Returns:
            List of video IDs from the channel

        Note:
            Quota cost with playlist method: ~1 unit per 50 videos
            Quota cost with search method: ~100 units per 50 videos
        """
        logger.info(f"Fetching videos from channel {channel_id}")

        if not use_search_fallback:
            # Try uploads playlist method first (much cheaper)
            try:
                uploads_playlist_id = self.get_uploads_playlist_id(channel_id)
                if uploads_playlist_id:
                    video_ids = self.get_playlist_videos(
                        uploads_playlist_id,
                        max_videos=max_videos,
                        max_results_per_page=max_results_per_page,
                    )
                    logger.info(
                        f"Retrieved {len(video_ids)} video IDs using uploads playlist"
                    )
                    return video_ids
            except Exception as e:
                logger.warning(f"Uploads playlist method failed, trying search: {e}")

        # Fallback to search.list (expensive)
        return self._get_channel_videos_via_search(
            channel_id, max_videos, max_results_per_page
        )

    def _get_channel_videos_via_search(
        self,
        channel_id: str,
        max_videos: Optional[int] = None,
        max_results_per_page: int = 50,
    ) -> List[str]:
        """
        Get video IDs using search.list (expensive fallback).

        Args:
            channel_id: YouTube channel ID
            max_videos: Maximum number of videos to retrieve
            max_results_per_page: Videos per API request (max 50)

        Returns:
            List of video IDs

        Note:
            API quota cost: ~100 units per request (avoid if possible)
        """
        video_ids: List[str] = []
        next_page_token: Optional[str] = None
        request_count = 0

        logger.warning(
            f"Using search.list for channel {channel_id} (100 units per request)"
        )

        try:
            while True:
                if max_videos and len(video_ids) >= max_videos:
                    break

                results_to_fetch = max_results_per_page
                if max_videos:
                    results_to_fetch = min(
                        max_results_per_page, max_videos - len(video_ids)
                    )

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

                for item in response.get("items", []):
                    if "videoId" in item["id"]:
                        video_ids.append(item["id"]["videoId"])
                        if max_videos and len(video_ids) >= max_videos:
                            break

                if max_videos and len(video_ids) >= max_videos:
                    break
                next_page_token = response.get("nextPageToken")
                if not next_page_token:
                    break

                time.sleep(0.1)

            logger.info(
                f"Retrieved {len(video_ids)} video IDs via search "
                f"({request_count} requests, {request_count * 100} quota units)"
            )
            return video_ids[:max_videos] if max_videos else video_ids

        except HttpError as e:
            logger.error(f"HTTP error in search for channel {channel_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error in search for channel {channel_id}: {e}")
            raise

    # =========================================================================
    # ENRICHMENT METHODS (Batched videos.list - Low Cost)
    # =========================================================================

    def get_videos_metadata_batch(
        self,
        video_ids: List[str],
        snapshot_date: Optional[datetime] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch metadata for multiple videos in batches of 50.

        This is 50x more efficient than calling get_video_metadata() per video.

        Args:
            video_ids: List of video IDs to fetch
            snapshot_date: When this data was collected (defaults to now)
            run_id: Collection run identifier

        Returns:
            List of video metadata dictionaries

        Note:
            API quota cost: 1 unit per 50 videos
            Example: 1000 videos = 20 units (vs 1000 units single-video method)
        """
        if not video_ids:
            return []

        if snapshot_date is None:
            snapshot_date = datetime.now()

        snapshot_date_str = snapshot_date.isoformat()
        all_metadata: List[Dict[str, Any]] = []
        total_batches = (len(video_ids) + BATCH_SIZE - 1) // BATCH_SIZE

        logger.info(
            f"Fetching metadata for {len(video_ids)} videos in {total_batches} batches"
        )

        for batch_idx in range(0, len(video_ids), BATCH_SIZE):
            batch = video_ids[batch_idx : batch_idx + BATCH_SIZE]
            batch_num = batch_idx // BATCH_SIZE + 1

            try:
                # Join video IDs for batch request
                ids_string = ",".join(batch)

                request = self.youtube.videos().list(
                    part="snippet,statistics,contentDetails",
                    id=ids_string,
                )
                response = request.execute()
                self.quota_used += 1  # 1 unit per request, regardless of batch size

                # Parse response
                for item in response.get("items", []):
                    metadata = self._parse_video_item(item)
                    metadata["snapshot_date"] = snapshot_date_str
                    if run_id:
                        metadata["run_id"] = run_id
                    all_metadata.append(metadata)

                # Log progress
                if batch_num % 10 == 0 or batch_num == total_batches:
                    logger.debug(
                        f"Batch {batch_num}/{total_batches}: "
                        f"{len(all_metadata)} videos fetched"
                    )

                # Rate limiting
                time.sleep(0.05)

            except HttpError as e:
                logger.error(f"HTTP error in batch {batch_num}: {e}")
                # Continue with next batch instead of failing completely
                continue
            except Exception as e:
                logger.error(f"Error in batch {batch_num}: {e}")
                continue

        logger.info(
            f"Fetched metadata for {len(all_metadata)}/{len(video_ids)} videos "
            f"({total_batches} API requests, {total_batches} quota units)"
        )

        return all_metadata

    def get_video_metadata(
        self,
        video_id: str,
        snapshot_date: Optional[datetime] = None,
        run_id: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Fetch detailed metadata for a single video.

        Note: For multiple videos, use get_videos_metadata_batch() instead
        for 50x better quota efficiency.

        Args:
            video_id: YouTube video ID
            snapshot_date: When this data was collected
            run_id: Collection run identifier

        Returns:
            Dictionary containing video metadata, or None if not found

        Note:
            API quota cost: 1 unit per request
        """
        results = self.get_videos_metadata_batch(
            [video_id],
            snapshot_date=snapshot_date,
            run_id=run_id,
        )
        return results[0] if results else None

    def _parse_video_item(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """
        Parse a single video item from videos.list response.

        Args:
            item: Video item from API response

        Returns:
            Parsed video metadata dictionary
        """
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})

        duration_iso = content_details.get("duration", "PT0S")

        return {
            "video_id": item.get("id", ""),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "publish_date": snippet.get("publishedAt", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "category_id": snippet.get("categoryId", ""),
            "duration_iso": duration_iso,
            "duration_sec": self._convert_iso_duration(duration_iso),
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "tags": ",".join(snippet.get("tags", [])),
            "thumbnail_url": self._get_thumbnail_url(snippet),
        }

    @staticmethod
    def _get_thumbnail_url(snippet: Dict[str, Any]) -> str:
        """Extract best available thumbnail URL from snippet."""
        thumbnails = snippet.get("thumbnails", {})
        for quality in ["medium", "default", "high", "standard", "maxres"]:
            if quality in thumbnails:
                return thumbnails[quality].get("url", "")
        return ""

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

    # =========================================================================
    # HIGH-LEVEL COLLECTION METHODS
    # =========================================================================

    def collect_from_channels(
        self,
        channels_df: pd.DataFrame,
        max_videos_per_channel: Optional[int] = None,
        max_quota: int = 8000,
        snapshot_date: Optional[datetime] = None,
        run_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Collect video metadata from multiple channels using optimized methods.

        Uses uploads playlist for discovery and batched videos.list for enrichment.

        Args:
            channels_df: DataFrame with 'channel_id' column (and optional 'channel_name')
            max_videos_per_channel: Maximum videos per channel (None = all)
            max_quota: Maximum API quota units to use
            snapshot_date: When this data was collected
            run_id: Collection run identifier

        Returns:
            DataFrame with collected video metadata

        Raises:
            ValueError: If channels_df missing 'channel_id' column

        Note:
            Quota usage (optimized):
            - Discovery: ~1 unit per 50 videos (via uploads playlist)
            - Enrichment: ~1 unit per 50 videos (via batched videos.list)
        """
        if "channel_id" not in channels_df.columns:
            raise ValueError("channels_df must contain 'channel_id' column")

        if snapshot_date is None:
            snapshot_date = datetime.now()

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

            # Get video IDs using uploads playlist (cheap)
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

            # Check if we have enough quota for enrichment
            estimated_enrichment_quota = (len(video_ids) + BATCH_SIZE - 1) // BATCH_SIZE
            if self.quota_used + estimated_enrichment_quota > max_quota:
                # Limit video IDs to stay within quota
                remaining_quota = max_quota - self.quota_used
                max_videos_quota = remaining_quota * BATCH_SIZE
                video_ids = video_ids[:max_videos_quota]
                logger.warning(
                    f"Limiting to {len(video_ids)} videos to stay within quota"
                )

            # Fetch metadata in batches (cheap)
            logger.info(f"Fetching metadata for {len(video_ids)} videos (batched)")
            channel_metadata = self.get_videos_metadata_batch(
                video_ids,
                snapshot_date=snapshot_date,
                run_id=run_id,
            )
            all_videos.extend(channel_metadata)

            logger.info(
                f"Collected {len(channel_metadata)} videos from {channel_name} "
                f"(Total quota: {self.quota_used}/{max_quota})"
            )

            if self.quota_used >= max_quota:
                break

        # Create DataFrame
        if all_videos:
            videos_df = pd.DataFrame(all_videos)
            if "publish_date" in videos_df.columns:
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

    # =========================================================================
    # CHANNEL INFO METHODS
    # =========================================================================

    def get_channel_info(self, channel_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed information about a channel.

        Args:
            channel_id: YouTube channel ID

        Returns:
            Dictionary with channel info, or None if not found

        Note:
            API quota cost: 1 unit
        """
        try:
            request = self.youtube.channels().list(
                part="snippet,statistics,contentDetails",
                id=channel_id,
            )
            response = request.execute()
            self.quota_used += 1

            if not response.get("items"):
                logger.warning(f"Channel {channel_id} not found")
                return None

            channel = response["items"][0]
            snippet = channel.get("snippet", {})
            statistics = channel.get("statistics", {})
            content_details = channel.get("contentDetails", {})

            return {
                "channel_id": channel_id,
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "country": snippet.get("country", ""),
                "created_at": snippet.get("publishedAt", ""),
                "subscriber_count": int(statistics.get("subscriberCount", 0)),
                "video_count": int(statistics.get("videoCount", 0)),
                "view_count": int(statistics.get("viewCount", 0)),
                "uploads_playlist_id": content_details.get(
                    "relatedPlaylists", {}
                ).get("uploads", ""),
            }

        except HttpError as e:
            logger.error(f"HTTP error getting channel info for {channel_id}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error getting channel info for {channel_id}: {e}")
            return None
