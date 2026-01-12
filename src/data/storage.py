"""
Raw JSON storage utilities for YouTube API responses.

This module handles persisting raw API responses to disk for auditing,
reprocessing, and debugging purposes.
"""

import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default base path for raw API responses
DEFAULT_API_RESPONSES_PATH = Path("data/raw/api_responses")


def get_run_id() -> str:
    """
    Generate a unique run ID for this collection session.

    Returns:
        8-character UUID hex string
    """
    return uuid.uuid4().hex[:8]


def save_raw_response(
    endpoint: str,
    response: Dict[str, Any],
    run_id: str,
    base_path: Optional[Path] = None,
) -> Path:
    """
    Save a raw YouTube API response to disk as JSON.

    Args:
        endpoint: API endpoint name (e.g., 'videos_list', 'channels_list', 'playlistItems_list')
        response: Raw API response dictionary
        run_id: Unique identifier for this collection run
        base_path: Base directory for API responses (defaults to data/raw/api_responses)

    Returns:
        Path to the saved JSON file

    Example:
        >>> run_id = get_run_id()
        >>> path = save_raw_response('videos_list', response, run_id)
        >>> # Saves to: data/raw/api_responses/videos_list/abc123_20260110_153045.json
    """
    if base_path is None:
        base_path = DEFAULT_API_RESPONSES_PATH

    # Create endpoint directory if needed
    endpoint_path = base_path / endpoint
    endpoint_path.mkdir(parents=True, exist_ok=True)

    # Generate filename with run_id and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{run_id}_{timestamp}.json"
    file_path = endpoint_path / filename

    # Write JSON with proper formatting
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response, f, ensure_ascii=False, indent=2)

    logger.debug(f"Saved raw response to {file_path}")
    return file_path


def parse_videos_response(
    response: Dict[str, Any], snapshot_date: datetime, run_id: str
) -> List[Dict[str, Any]]:
    """
    Parse a videos.list API response into structured video metadata records.

    Args:
        response: Raw videos.list API response
        snapshot_date: When this data was collected
        run_id: Collection run identifier

    Returns:
        List of video metadata dictionaries with standardized schema
    """
    videos = []
    snapshot_date_str = snapshot_date.isoformat()

    for item in response.get("items", []):
        snippet = item.get("snippet", {})
        statistics = item.get("statistics", {})
        content_details = item.get("contentDetails", {})

        video = {
            "video_id": item.get("id", ""),
            "title": snippet.get("title", ""),
            "description": snippet.get("description", ""),
            "publish_date": snippet.get("publishedAt", ""),
            "channel_id": snippet.get("channelId", ""),
            "channel_title": snippet.get("channelTitle", ""),
            "category_id": snippet.get("categoryId", ""),
            "duration_iso": content_details.get("duration", "PT0S"),
            "view_count": int(statistics.get("viewCount", 0)),
            "like_count": int(statistics.get("likeCount", 0)),
            "comment_count": int(statistics.get("commentCount", 0)),
            "tags": ",".join(snippet.get("tags", [])),
            "thumbnail_url": _get_thumbnail_url(snippet),
            "snapshot_date": snapshot_date_str,
            "run_id": run_id,
        }
        videos.append(video)

    return videos


def parse_playlist_items_response(response: Dict[str, Any]) -> List[Dict[str, str]]:
    """
    Parse a playlistItems.list API response to extract video IDs.

    Args:
        response: Raw playlistItems.list API response

    Returns:
        List of dictionaries with video_id and publish_date
    """
    items = []
    for item in response.get("items", []):
        content_details = item.get("contentDetails", {})
        snippet = item.get("snippet", {})

        video_id = content_details.get("videoId", "")
        if video_id:
            items.append(
                {
                    "video_id": video_id,
                    "publish_date": snippet.get("publishedAt", ""),
                }
            )

    return items


def parse_channel_response(response: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Parse a channels.list API response to extract channel info and uploads playlist ID.

    Args:
        response: Raw channels.list API response

    Returns:
        Dictionary with channel info including uploads_playlist_id, or None if not found
    """
    items = response.get("items", [])
    if not items:
        return None

    channel = items[0]
    snippet = channel.get("snippet", {})
    statistics = channel.get("statistics", {})
    content_details = channel.get("contentDetails", {})
    related_playlists = content_details.get("relatedPlaylists", {})

    return {
        "channel_id": channel.get("id", ""),
        "uploads_playlist_id": related_playlists.get("uploads", ""),
        "title": snippet.get("title", ""),
        "description": snippet.get("description", ""),
        "country": snippet.get("country", ""),
        "created_at": snippet.get("publishedAt", ""),
        "subscriber_count": int(statistics.get("subscriberCount", 0)),
        "video_count": int(statistics.get("videoCount", 0)),
        "view_count": int(statistics.get("viewCount", 0)),
    }


def _get_thumbnail_url(snippet: Dict[str, Any]) -> str:
    """Extract best available thumbnail URL from snippet."""
    thumbnails = snippet.get("thumbnails", {})
    # Prefer medium, then default, then any available
    for quality in ["medium", "default", "high", "standard", "maxres"]:
        if quality in thumbnails:
            return thumbnails[quality].get("url", "")
    return ""


def load_raw_response(file_path: Path) -> Dict[str, Any]:
    """
    Load a raw JSON response from disk.

    Args:
        file_path: Path to the JSON file

    Returns:
        Parsed JSON as dictionary

    Raises:
        FileNotFoundError: If file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)
