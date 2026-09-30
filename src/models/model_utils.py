"""
Shared utilities for model training and prediction.

A trained model is saved as one bundle: ``model.joblib`` holds the fitted
``VideoFeatureEngineer`` and the model together, so they cannot drift apart,
and ``metadata.json`` holds a readable copy of the metrics and settings.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.features.engineer import VideoFeatureEngineer

logger = logging.getLogger(__name__)

BUNDLE_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load a CSV, failing clearly if it is missing."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)


def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Regression metrics on a held-out set."""
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    return {
        "mae": float(mean_absolute_error(y, y_pred)),
        "mse": float(mse),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y, y_pred)),
    }


def save_bundle(
    output_dir: Union[str, Path],
    engineer: VideoFeatureEngineer,
    model: Any,
    metadata: Dict[str, Any],
) -> Path:
    """Write ``model.joblib`` and ``metadata.json`` to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / BUNDLE_FILE
    joblib.dump({"engineer": engineer, "model": model, "metadata": metadata}, path)
    (output_dir / METADATA_FILE).write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8"
    )
    logger.info(f"Saved model bundle to {path}")
    return path


def load_bundle(model_dir: Union[str, Path]) -> Tuple[VideoFeatureEngineer, Any, Dict[str, Any]]:
    """Load a bundle written by ``save_bundle``.

    Only load bundles you trained yourself: joblib files can execute code.
    """
    path = Path(model_dir) / BUNDLE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"No {BUNDLE_FILE} in {model_dir}. Train one with `run_pipeline.py train`."
        )
    bundle = joblib.load(path)
    return bundle["engineer"], bundle["model"], bundle["metadata"]
