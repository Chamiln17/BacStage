"""
Shared utilities for model training and prediction.
"""

import logging
import json
import pickle
import joblib
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

logger = logging.getLogger(__name__)

def load_data(path: Union[str, Path]) -> pd.DataFrame:
    """Load data from CSV file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    
    logger.info(f"Loading data from {path}")
    return pd.read_csv(path)

def save_model_artifacts(
    output_dir: Union[str, Path],
    model: Any,
    artifacts: Dict[str, Any],
    metadata: Dict[str, Any]
) -> None:
    """
    Save model and associated artifacts to directory.
    
    Args:
        output_dir: Directory to save artifacts
        model: Trained model object
        artifacts: Dictionary of additional artifacts (scaler, vectorizers, etc.)
        metadata: Dictionary of training metadata
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / f"{metadata.get('model_type', 'model')}_production.pkl"
    joblib.dump(model, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # Save artifacts
    for name, artifact in artifacts.items():
        if artifact is not None:
            artifact_path = output_dir / f"{name}_production.pkl"
            joblib.dump(artifact, artifact_path)
            logger.info(f"Saved artifact {name} to {artifact_path}")
            
    # Save metadata
    metadata_path = output_dir / "training_metadata.json"
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"Saved metadata to {metadata_path}")

def load_model_artifacts(
    model_dir: Union[str, Path], 
    model_type: str = "catboost"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Load model and artifacts for inference.
    
    Args:
        model_dir: Directory containing artifacts
        model_type: Type of model to load (e.g. 'catboost', 'xgboost')
        
    Returns:
        Tuple of (model, artifacts_dict)
    """
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
        
    # Load model
    model_path = model_dir / f"{model_type}_production.pkl"
    if not model_path.exists():
        # Try finding any model pickle if specific type not found
        pkl_files = list(model_dir.glob("*_production.pkl"))
        potential_models = [f for f in pkl_files if "scaler" not in f.name and "tfidf" not in f.name and "pca" not in f.name]
        if potential_models:
            model_path = potential_models[0]
            logger.warning(f"Requested model type '{model_type}' not found, falling back to {model_path.name}")
        else:
            raise FileNotFoundError(f"No model found in {model_dir}")
            
    logger.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    
    # Load artifacts
    artifacts = {}
    artifact_names = ["scaler", "tfidf", "pca", "arabert_cache", "encoder"]
    
    for name in artifact_names:
        path = model_dir / f"{name}_production.pkl"
        if path.exists():
            artifacts[name] = joblib.load(path)
            logger.info(f"Loaded artifact: {name}")
            
    # Load feature columns if available
    features_path = model_dir / "feature_columns.json"
    if features_path.exists():
        with open(features_path, 'r') as f:
            artifacts["feature_columns"] = json.load(f)
            
    return model, artifacts

def evaluate_model(model: Any, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Calculate regression metrics."""
    y_pred = model.predict(X)
    
    metrics = {
        "mae": float(mean_absolute_error(y, y_pred)),
        "mse": float(mean_squared_error(y, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, y_pred))),
        "r2": float(r2_score(y, y_pred))
    }
    
    return metrics

def select_features_vif(
    X: pd.DataFrame, 
    vif_threshold: float = 10.0, 
    correlation_threshold: float = 0.95
) -> List[str]:
    """
    Select features using VIF and correlation analysis.
    
    Args:
        X: DataFrame of features
        vif_threshold: Max VIF (remove if higher)
        correlation_threshold: Max inter-feature correlation
        
    Returns:
        List of selected feature names
    """
    X_clean = X.copy().fillna(0)
    
    # Step 1: Remove highly correlated features
    corr_matrix = X_clean.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    to_drop = set()
    for col in upper.columns:
        if any(upper[col] > correlation_threshold):
            to_drop.add(col)
            
    if to_drop:
        logger.info(f"Dropping {len(to_drop)} highly correlated features: {list(to_drop)[:5]}...")
        X_clean = X_clean.drop(columns=list(to_drop))
        
    # Step 2: Iterative VIF removal
    while len(X_clean.columns) >= 2:
        vif_data = []
        for i, col in enumerate(X_clean.columns):
            try:
                vif = variance_inflation_factor(X_clean.values, i)
                vif_data.append((col, vif))
            except Exception:
                vif_data.append((col, 0))
                
        if not vif_data:
            break
            
        vif_df = pd.DataFrame(vif_data, columns=['feature', 'VIF'])
        max_vif = vif_df.loc[vif_df['VIF'].idxmax()]
        
        if max_vif['VIF'] > vif_threshold:
            logger.debug(f"Dropping {max_vif['feature']} (VIF={max_vif['VIF']:.2f})")
            X_clean = X_clean.drop(columns=[max_vif['feature']])
        else:
            break
            
    selected = X_clean.columns.tolist()
    logger.info(f"VIF selection retained {len(selected)}/{len(X.columns)} features")
    return selected
