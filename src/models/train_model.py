
"""
Production Training Script for Engagement Prediction Model.

Handles:
1. Data loading and preprocessing
2. Feature engineering (Numerical + Text)
3. Model training (XGBoost, CatBoost, LightGBM, Random Forest)
4. Evaluation and artifact saving
"""

import argparse
import joblib
import json
import logging
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor

# Optional imports for specific models
try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from catboost import CatBoostRegressor
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

from src.models.model_utils import (
    save_model_artifacts, 
    evaluate_model,
    select_features_vif
)
from src.features.text_embeddings import TextEmbeddingExtractor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Constants
RANDOM_SEED = 42
TARGET_COL = "engagement_score"
EXCLUDE_COLS = [
    'video_id', 'channel_id', 'channel_title',
    TARGET_COL, 'engagement_category',
    'view_count', 'like_count', 'comment_count',
    'like_ratio', 'comment_ratio',
    'channel_avg_engagement',
    'title', 'description', 'tags', 'publish_date', 'publish_day_of_week',
    "channel_video_avg_duration", 
    #"channel_video_count", 
    #"channel_avg_views", 
    #"channel_age_days"
]

def parse_args():
    parser = argparse.ArgumentParser(description="Train Engagement Prediction Model")
    parser.add_argument("--input", type=Path, default=Path("data/processed/videos_engineered.csv"), help="Input CSV path")
    parser.add_argument("--output-dir", type=Path, default=Path("models"), help="Output directory for model artifacts")
    parser.add_argument("--model-type", type=str, default="catboost", choices=["catboost", "xgboost", "lightgbm", "rf"], help="Model type to train")
    parser.add_argument("--cv-folds", type=int, default=0, help="Number of CV folds (0 to disable)")
    parser.add_argument("--no-arabert", action="store_true", help="Skip AraBERT embeddings")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED, help="Random seed")
    return parser.parse_args()

class ModelTrainer:
    def __init__(self, output_dir: Path, random_seed: int = RANDOM_SEED):
        self.output_dir = output_dir
        self.random_seed = random_seed
        self.scaler = StandardScaler()
        self.tfidf = None
        self.pca = None
        self.feature_columns = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_numerical_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare numerical features and target."""
        # one-hot encode categorical features like 'subject' if available
        # Check for categorical columns that should be one-hot encoded
        categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c not in EXCLUDE_COLS]
        
        df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # Handle missing values (fill with 0 as per notebook strategy)
        # This is critical for features like transcript counts where missing implies 0
        df_encoded = df_encoded.fillna(0)
        
        # Select numerical features
        numeric_cols = df_encoded.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in EXCLUDE_COLS and c != TARGET_COL]
        
        # VIF Selection
        logger.info(f"Running VIF selection on {len(feature_cols)} features...")
        selected_features = select_features_vif(df_encoded[feature_cols])
        self.feature_columns = selected_features
        
        return df_encoded[selected_features], df_encoded[TARGET_COL]

    def extract_text_features(self, df: pd.DataFrame, use_arabert: bool = True) -> np.ndarray:
        """Extract text features using TF-IDF and AraBERT."""
        # Combine title and description
        text_data = (df["title"].fillna("") + " " + df["description"].fillna("")).tolist()
        
        logger.info("Extracting TF-IDF features...")
        self.tfidf = TfidfVectorizer(
            max_features=30,
            min_df=1 if len(df) < 20 else 10, # Adjust min_df for small datasets (testing)
            max_df=1.0 if len(df) < 20 else 0.5,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        tfidf_features = self.tfidf.fit_transform(text_data).toarray()
        
        arabert_features = None
        if use_arabert:
            logger.info("Extracting AraBERT embeddings...")
            try:
                extractor = TextEmbeddingExtractor(device='cuda' if joblib.cpu_count() > 0 else 'cpu') 
                embeddings = extractor.get_embeddings(text_data, batch_size=32)
                
                # Reduce dimensionality with PCA
                n_samples = len(df)
                n_components = min(100, n_samples, embeddings.shape[1])
                logger.info(f"Reducing AraBERT dimensions (768 -> {n_components}) with PCA...")
                
                self.pca = PCA(n_components=n_components, random_state=self.random_seed)
                arabert_features = self.pca.fit_transform(embeddings)
            except Exception as e:
                logger.error(f"Failed to extract AraBERT features: {e}")
                # For testing/automation, we should NOT prompt user input.
                # Just fail if it's a critical error, or skip if optional.
                # Here we skip.
                logger.warning("Skipping AraBERT features due to error.")
                arabert_features = None
        
        # Combine features
        if arabert_features is not None:
            return np.hstack([tfidf_features, arabert_features])
        return tfidf_features

    def get_model(self, model_type: str):
        """Factory for model creation."""
        if model_type == "xgb" or model_type == "xgboost":
            if not XGBOOST_AVAILABLE:
                raise ImportError("XGBoost not installed. Run `uv add xgboost`")
            # Hyperparameters from notebook Optuna tuning (approximate best)
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
                random_state=self.random_seed
            )
        elif model_type == "cat" or model_type == "catboost":
            if not CATBOOST_AVAILABLE:
                raise ImportError("CatBoost not installed. Run `uv add catboost`")
            # Hyperparameters from notebook Optuna tuning
            return CatBoostRegressor(
                iterations=357,
                depth=6,
                learning_rate=0.15,
                l2_leaf_reg=1.75,
                bagging_temperature=0.43,
                random_strength=0.48,
                verbose=False,
                random_seed=self.random_seed,
                allow_writing_files=False
            )
        elif model_type == "lgbm" or model_type == "lightgbm":
            if not LIGHTGBM_AVAILABLE:
                raise ImportError("LightGBM not installed. Run `uv add lightgbm`")
            return LGBMRegressor(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.05,
                n_jobs=-1,
                random_state=self.random_seed,
                verbose=-1
            )
        elif model_type == "rf":
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=20,
                n_jobs=-1,
                random_state=self.random_seed
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def train(self, df: pd.DataFrame, model_type: str = "catboost", use_arabert: bool = True):
        """Execute full training pipeline."""
        logger.info(f"Starting training pipeline with {len(df)} samples")
        
        # 1. Prepare Numerical Features
        X_num_df, y = self.prepare_numerical_features(df)
        X_num = X_num_df.values
        
        # Scaling
        X_num = self.scaler.fit_transform(X_num)
        
        # 2. Extract Text Features
        X_text = self.extract_text_features(df, use_arabert=use_arabert)
        
        # 3. Combine Features
        X = np.hstack([X_num, X_text])
        logger.info(f"Total features: {X.shape[1]} (Numeric: {X_num.shape[1]}, Text: {X_text.shape[1]})")
        
        # 4. Split Data (60/20/20 split logic from notebook, simplified here to Train/Test for production)
        # However, for production we typically train on ALL data or a large train/val split.
        # Let's use a standard Train/Test split for verification metrics.
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_seed
        )
        
        # 5. Train Model
        logger.info(f"Training {model_type} model...")
        model = self.get_model(model_type)
        model.fit(X_train, y_train)
        
        # 6. Evaluate
        metrics = evaluate_model(model, X_test, y_test)
        logger.info(f"Validation Metrics: {metrics}")
        
        # 7. Save Artifacts
        artifacts = {
            "scaler": self.scaler,
            "tfidf": self.tfidf,
            "pca": self.pca,
        }
        
        metadata = {
            "model_type": model_type,
            "training_date": datetime.now().isoformat(),
            "n_samples": len(df),
            "n_features": X.shape[1],
            "metrics": metrics,
            "features": self.feature_columns,
            "use_arabert": use_arabert
        }
        
        # If best model, we might want to re-train on ALL data? 
        # For now, let's save the trained model as is to avoid overfitting or complexity.
        # Ideally, we verify on test, then train production model on full dataset.
        # Let's do a final train on full dataset for the 'production' model
        logger.info("Retraining on full dataset for production artifact...")
        prod_model = self.get_model(model_type)
        prod_model.fit(X, y)
        
        save_model_artifacts(self.output_dir, prod_model, artifacts, metadata)
        logger.info("Training complete successfully.")

def main():
    args = parse_args()
    from src.models.model_utils import load_data
    
    # Load data
    try:
        df = load_data(args.input)
    except FileNotFoundError:
        logger.error(f"Input file not found: {args.input}")
        return 1
        
    trainer = ModelTrainer(args.output_dir, random_seed=args.random_seed)
    
    try:
        trainer.train(
            df, 
            model_type=args.model_type, 
            use_arabert=not args.no_arabert
        )
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1
        
    return 0

if __name__ == "__main__":
    from typing import Tuple 
    # Fix Tuple import for older python versions if needed in class method signature
    # But it's already imported from typing
    exit(main())
