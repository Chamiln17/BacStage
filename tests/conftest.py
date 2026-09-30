"""Pytest configuration and fixtures."""

from typing import List

import numpy as np
import pandas as pd
import pytest

TOPICS = [
    "الدالة الأسية",
    "الأعداد المركبة",
    "التكامل",
    "les nombres complexes",
    "la dérivée",
    "المناعة",
    "التركيب الضوئي",
    "Newton",
]


class FakeEmbedder:
    """Deterministic stand-in for AraBERT: 16-dim vectors derived from each text."""

    def __init__(self) -> None:
        self.calls = 0

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        self.calls += 1
        return np.array(
            [np.random.default_rng(sum(map(ord, t))).normal(size=16) for t in texts]
        )


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def training_videos() -> pd.DataFrame:
    """60 cleaned videos over 3 channels, half with transcripts, with statistics."""
    rng = np.random.default_rng(0)
    n = 60
    channels = np.array(["UC_math", "UC_bio", "UC_phys"])[np.arange(n) % 3]
    subjects = {"UC_math": "Maths", "UC_bio": "Natural Sciences", "UC_phys": "Physics"}
    views = rng.integers(200, 50_000, n)
    transcripts = [
        f"اليوم ندرس {TOPICS[i % 8]}. لماذا؟ لأن المثال مهم. مثلا نحل التمرين {i}."
        if i % 2 == 0
        else None
        for i in range(n)
    ]
    return pd.DataFrame(
        {
            "video_id": [f"vid_{i}" for i in range(n)],
            "title": [f"بكالوريا 2025 {TOPICS[i % 8]} الجزء {i}" + ("؟" if i % 5 == 0 else "") for i in range(n)],
            "description": [f"شرح {TOPICS[(i + 3) % 8]} للسنة الثالثة ثانوي" for i in range(n)],
            "tags": ["bac,3as" if i % 4 else "" for i in range(n)],
            "publish_date": pd.date_range("2024-01-01", periods=n, freq="3D", tz="UTC").strftime("%Y-%m-%dT%H:%M:%SZ"),
            "channel_id": channels,
            "channel_title": channels,
            "subject": [subjects[c] for c in channels],
            "duration_sec": rng.integers(300, 3600, n),
            "view_count": views,
            "like_count": (views * rng.uniform(0.01, 0.08, n)).astype(int),
            "comment_count": (views * rng.uniform(0.0, 0.01, n)).astype(int),
            "transcript_text": transcripts,
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
