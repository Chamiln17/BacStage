"""
Predict engagement for planned videos with a trained bundle.

A planned video needs ``title``, ``duration_sec`` and a ``channel_id`` or
``subject``. ``description``, ``tags``, ``publish_date`` and
``transcript_text`` are optional. Statistics are never needed.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from src.features.engineer import TextEmbedder, engagement_category
from src.models.model_utils import load_bundle

logger = logging.getLogger(__name__)


class EngagementPredictor:
    """Scores planned videos with the model and features saved by ``train``."""

    def __init__(
        self, model_dir: Union[str, Path] = "models", embedder: Optional[TextEmbedder] = None
    ) -> None:
        """
        Args:
            model_dir: Directory holding ``model.joblib``.
            embedder: Text embedder to use if the bundle was trained with one.
                Defaults to loading the embedding model named in the metadata.
        """
        self.engineer, self.model, self.metadata = load_bundle(model_dir)
        if self.engineer.uses_embeddings:
            if embedder is None:
                from src.features.text_embeddings import TextEmbeddingExtractor

                embedder = TextEmbeddingExtractor(self.metadata["embedding_model"])
            self.engineer.embedder = embedder
        self.thresholds = self.metadata["engagement_thresholds"]

    def predict(self, video: Dict[str, Any]) -> Dict[str, Any]:
        """Score one planned video.

        Returns:
            ``engagement_score``, ``engagement_category`` (Low/Medium/High by the
            training thresholds) and ``known_channel`` (False means the channel
            was unseen in training and channel features fell back to medians).
        """
        row = self.predict_batch(pd.DataFrame([video])).iloc[0]
        return {
            "engagement_score": float(row["predicted_engagement_score"]),
            "engagement_category": str(row["predicted_engagement_category"]),
            "known_channel": bool(row["known_channel"]),
        }

    def predict_batch(self, videos: pd.DataFrame) -> pd.DataFrame:
        """Score many videos; returns ``videos`` with three prediction columns added."""
        scores = pd.Series(self.model.predict(self.engineer.transform(videos)), index=videos.index)
        return videos.assign(
            predicted_engagement_score=scores,
            predicted_engagement_category=engagement_category(scores, self.thresholds),
            known_channel=self.engineer.known_channel(videos).to_numpy(),
        )
