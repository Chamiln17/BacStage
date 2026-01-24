"""
Inference Script for Engagement Prediction Model.

Provides EngagementPredictor class for single-instance and batch predictions.
"""

import sys
import logging
import json
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Union, List, Optional, Any

# Ensure project root is in path
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.models.model_utils import load_model_artifacts
from src.features.text_embeddings import TextEmbeddingExtractor

logger = logging.getLogger(__name__)

class EngagementPredictor:
    """
    Predicts engagement scores for educational YouTube videos.
    """
    
    def __init__(self, model_dir: Union[str, Path] = "models", model_type: str = "catboost", device: str = None):
        """
        Initialize predictor by loading model and artifacts.
        
        Args:
            model_dir: Directory containing trained model artifacts
            model_type: Type of model to load
            device: Device for AraBERT (cpu/cuda)
        """
        self.model_dir = Path(model_dir)
        self.device = device
        
        try:
            self.model, self.artifacts = load_model_artifacts(self.model_dir, model_type)
            self.feature_columns = self.artifacts.get("feature_columns", [])
            self.scaler = self.artifacts.get("scaler")
            self.tfidf = self.artifacts.get("tfidf")
            self.pca = self.artifacts.get("pca")
            
            # Load AraBERT extractor only if needed (PCA present)
            if self.pca:
                self.arabert_extractor = TextEmbeddingExtractor(device=self.device)
            else:
                self.arabert_extractor = None
                
            logger.info("EngagementPredictor initialized successfully")
            
        except FileNotFoundError as e:
            logger.error(f"Initialization failed: {e}")
            raise

    def preprocess_input(self, video_data: Dict[str, Any]) -> np.ndarray:
        """
        Transform raw video data into model input vector.
        
        Args:
            video_data: Dictionary containing video metadata
            
        Returns:
            Numpy array of features
        """
        # Convert to DataFrame for easier processing
        df = pd.DataFrame([video_data])
        
        # 1. Feature Engineering (Simplified for inference)
        # We need to replicate the exact features used in training.
        # This is tricky because training used VIF selection on a batch.
        # Ideally, we should reuse the VIF-selected columns list.
        
        # Basic numerical features
        # Note: This requires the input to ALREADY have the engineered numerical features 
        # OR we need to re-implement feature extraction here.
        # Given the complexity of engineer.py, we should ideally use it.
        
        from src.features.engineer import VideoFeatureEngineer
        
        # Initialize engineer (date shouldn't matter too much for single prediction score, 
        # but for 'days_since_publish' it matters! 
        # For prediction of NEW videos, days_since_publish is effectively 0 or small.)
        engineer = VideoFeatureEngineer()
        
        # Mock missing columns required by engineer if they don't exist
        required_cols = ["view_count", "like_count", "comment_count", "publish_date"]
        for col in required_cols:
            if col not in df.columns:
                if col == "publish_date":
                    df[col] = pd.Timestamp.now(tz="UTC")
                else:
                    df[col] = 0 # Default stats to 0 for new videos
        
        # Engineer features
        df_engineered = engineer.fit_transform(df)
        
        # One-hot encoding (align with training columns)
        # This is difficult for single-instance prediction if we rely on pd.get_dummies
        # because we might miss categories. 
        # For V1, let's assume we handle numericals + text primarily.
        # Better approach: initialize empty DataFrame with all feature_columns as 0, then fill.
        
        input_data = pd.DataFrame(0, index=[0], columns=self.feature_columns)
        
        # Fill available numeric features
        for col in self.feature_columns:
            if col in df_engineered.columns:
                input_data[col] = df_engineered[col]
            else:
                # Handle categorical one-hot columns manually if needed
                # e.g. subject_Maths
                pass
                
        X_num = input_data.values
        
        # Scale
        if self.scaler:
            X_num = self.scaler.transform(X_num)
            
        # 2. Text Features
        text = (str(video_data.get("title", "")) + " " + str(video_data.get("description", "")))
        
        X_tfidf = np.empty((1, 0))
        if self.tfidf:
            X_tfidf = self.tfidf.transform([text]).toarray()
            
        X_bert = np.empty((1, 0))
        if self.pca and self.arabert_extractor:
            embedding = self.arabert_extractor.get_embeddings([text], batch_size=1)
            X_bert = self.pca.transform(embedding)
            
        # Combine
        X_final = np.hstack([X_num, X_tfidf, X_bert])
        return X_final

    def predict(self, video_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict engagement score for a single video.
        
        Args:
            video_data: Dict with keys: title, description, duration_sec, etc.
            
        Returns:
            Dict containing engagement_score and metadata
        """
        try:
            X = self.preprocess_input(video_data)
            score = self.model.predict(X)[0]
            
            # Unlog score if it was log-transformed in training (it was: np.log1p)
            # engagement_score = np.expm1(score) # Optional: return raw log score or actual scale?
            # Notebook target was np.log1p(score). So model predicts log score.
            # Returning model raw output (log score) allows comparison with "Low/Medium/High" thresholds trained on log scores.
            
            return {
                "engagement_score": float(score),
                "predicted_metric": float(np.expm1(score)), # Approximate real interactions
                "model_version": "1.0"
            }
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise

    def predict_batch(self, videos_df: pd.DataFrame) -> pd.DataFrame:
        """Batch prediction."""
        results = []
        for _, row in videos_df.iterrows():
            try:
                res = self.predict(row.to_dict())
                results.append(res["engagement_score"])
            except Exception:
                results.append(None)
        
        videos_df["predicted_score"] = results
        return videos_df

def main():
    """CLI Entry point for prediction."""
    import argparse
    parser = argparse.ArgumentParser(description="Predict Engagement Score")
    parser.add_argument("--input", required=True, help="Input file (JSON/CSV)")
    parser.add_argument("--model-dir", default="models", help="Model directory")
    parser.add_argument("--output", default="stdout", help="Output path")
    args = parser.parse_args()
    
    try:
        predictor = EngagementPredictor(model_dir=args.model_dir)
        
        # Load input
        if args.input.endswith(".json"):
            with open(args.input, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    # Batch (list of dicts) - convert to DataFrame for now or loop
                    # For CLI batch, let's assume CSV is better, but handle list JSON
                    df = pd.DataFrame(data)
                    res_df = predictor.predict_batch(df)
                    output_data = res_df.to_dict(orient='records')
                else:
                    # Single dict
                    result = predictor.predict(data)
                    output_data = result
        elif args.input.endswith(".csv"):
            df = pd.read_csv(args.input)
            res_df = predictor.predict_batch(df)
            output_data = res_df.to_dict(orient='records')
        
        # Output
        if args.output == "stdout":
            print(json.dumps(output_data, indent=2, ensure_ascii=False))
        else:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
                
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    return 0

if __name__ == "__main__":
    exit(main())
