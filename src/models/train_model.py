"""
Train the engagement model.

Splits the videos first, fits the feature engineer on the training split
only, evaluates on the held-out split, then refits features and model on all
videos for the saved production bundle.
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from src.features.engineer import TextEmbedder, VideoFeatureEngineer, engagement_score
from src.models.model_utils import evaluate_model, save_bundle

logger = logging.getLogger(__name__)

RANDOM_SEED = 42
MODEL_TYPES = ("catboost", "xgboost", "lightgbm", "rf")


def get_model(model_type: str, random_seed: int = RANDOM_SEED) -> Any:
    """Build an untrained regressor. Hyperparameters come from the notebook's Optuna runs."""
    if model_type == "xgboost":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=165,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.7,
            colsample_bytree=0.5,
            min_child_weight=14,
            reg_alpha=2.2,
            reg_lambda=1.7,
            gamma=0.2,
            n_jobs=-1,
            random_state=random_seed,
        )
    if model_type == "catboost":
        from catboost import CatBoostRegressor

        return CatBoostRegressor(
            iterations=357,
            depth=6,
            learning_rate=0.15,
            l2_leaf_reg=1.75,
            bagging_temperature=0.43,
            random_strength=0.48,
            verbose=False,
            random_seed=random_seed,
            allow_writing_files=False,
        )
    if model_type == "lightgbm":
        from lightgbm import LGBMRegressor

        return LGBMRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            n_jobs=-1,
            random_state=random_seed,
            verbose=-1,
        )
    if model_type == "rf":
        return RandomForestRegressor(
            n_estimators=100, max_depth=20, n_jobs=-1, random_state=random_seed
        )
    raise ValueError(f"Unknown model type: {model_type}. Choose from {MODEL_TYPES}")


class CachedEmbedder:
    """Embeds each distinct text once. Training transforms the same texts several times."""

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder
        self.cache: Dict[str, np.ndarray] = {}

    def get_embeddings(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        new = list(dict.fromkeys(t for t in texts if t not in self.cache))
        if new:
            for text, vector in zip(new, self.embedder.get_embeddings(new, batch_size), strict=True):
                self.cache[text] = vector
        return np.vstack([self.cache[t] for t in texts])


def train(
    videos: pd.DataFrame,
    output_dir: Union[str, Path],
    model_type: str = "catboost",
    embedder: Optional[TextEmbedder] = None,
    embedding_model: Optional[str] = None,
    test_ids: Optional[Iterable[str]] = None,
    random_seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Train, evaluate on held-out videos, and save the production bundle.

    Args:
        videos: Cleaned videos with statistics, optionally with ``transcript_text``.
        output_dir: Where ``model.joblib`` and ``metadata.json`` are written.
        model_type: One of ``MODEL_TYPES``.
        embedder: Text embedder, or None to train without embeddings.
        embedding_model: Name recorded so prediction can rebuild the embedder.
        test_ids: Video IDs to hold out. None holds out a random 20%.
        random_seed: Seed for the split, PCA and model.

    Returns:
        The metadata saved with the bundle, including ``test_metrics``.
    """
    # A video with no views has no meaningful engagement score.
    videos = videos[videos["view_count"].fillna(0) > 0].reset_index(drop=True)
    y = engagement_score(videos)

    if test_ids is not None:
        is_test = videos["video_id"].isin(set(test_ids)).to_numpy()
        split = "given test video_ids"
    else:
        _, test_idx = train_test_split(np.arange(len(videos)), test_size=0.2, random_state=random_seed)
        is_test = np.isin(np.arange(len(videos)), test_idx)
        split = "random 80/20"
    if not is_test.any() or is_test.all():
        raise ValueError("The split must leave both training and test videos")

    cached = CachedEmbedder(embedder) if embedder is not None else None
    reference_date = datetime.now(timezone.utc)

    def fit(rows: pd.DataFrame, target: pd.Series) -> Tuple[VideoFeatureEngineer, Any]:
        engineer = VideoFeatureEngineer(cached, reference_date, random_seed=random_seed).fit(rows)
        model = get_model(model_type, random_seed).fit(engineer.transform(rows), target)
        return engineer, model

    train_rows, test_rows = videos[~is_test], videos[is_test]
    logger.info(f"Training {model_type} on {len(train_rows)} videos, testing on {len(test_rows)} ({split})")
    engineer, model = fit(train_rows, y[~is_test])
    metrics = evaluate_model(model, engineer.transform(test_rows), y[is_test])
    logger.info(f"Test metrics: {metrics}")

    logger.info("Refitting on all videos for the production bundle")
    engineer, model = fit(videos, y)
    metadata = {
        "model_type": model_type,
        "trained_at": reference_date.isoformat(),
        "split": split,
        "n_train": int((~is_test).sum()),
        "n_test": int(is_test.sum()),
        "test_metrics": metrics,
        "features": engineer.feature_names,
        "embedding_model": embedding_model if embedder is not None else None,
        "engagement_thresholds": y.quantile([0.33, 0.67]).tolist(),
        "random_seed": random_seed,
    }
    save_bundle(output_dir, engineer, model, metadata)
    return metadata
