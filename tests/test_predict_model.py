
import pytest
import pandas as pd
import numpy as np
import joblib
import json
from unittest.mock import MagicMock, patch
from src.models.predict_model import EngagementPredictor
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA

# Dummy model class that can be pickled
class MockModel:
    def predict(self, X):
        return np.array([4.2] * X.shape[0])

@pytest.fixture
def mock_artifacts(tmp_path):
    """Create mock model artifacts using real sklearn objects."""
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    
    # Mock Model
    model = MockModel()
    
    # Real Scaler (fit on dummy data)
    scaler = StandardScaler()
    scaler.fit(np.zeros((10, 5)))
    
    # Real TF-IDF
    tfidf = TfidfVectorizer()
    tfidf.fit(["dummy text"])
    
    # Real PCA
    pca = PCA(n_components=1) # Minimal component
    pca.fit(np.zeros((10, 768))) # Fit on 768 dim data (AraBERT output)
    
    feature_columns = ["view_count", "like_count", "duration_sec", "subject_Maths", "feature_5"]
    
    # Save artifacts
    joblib.dump(model, model_dir / "catboost_production.pkl")
    joblib.dump(scaler, model_dir / "scaler_production.pkl")
    joblib.dump(tfidf, model_dir / "tfidf_production.pkl")
    joblib.dump(pca, model_dir / "pca_production.pkl")
    
    with open(model_dir / "feature_columns.json", 'w') as f:
        json.dump(feature_columns, f)
        
    return model_dir

@patch("src.models.predict_model.TextEmbeddingExtractor")
@patch("src.features.engineer.VideoFeatureEngineer") 
def test_predictor_initialization(mock_engineer_cls, mock_extractor_cls, mock_artifacts):
    """Test loading of artifacts."""
    predictor = EngagementPredictor(model_dir=mock_artifacts, model_type="catboost")
    
    assert predictor.model is not None
    assert predictor.scaler is not None
    assert predictor.tfidf is not None
    assert predictor.pca is not None
    assert len(predictor.feature_columns) == 5

@patch("src.models.predict_model.TextEmbeddingExtractor")
@patch("src.features.engineer.VideoFeatureEngineer")
def test_predict_single(mock_engineer_cls, mock_extractor_cls, mock_artifacts):
    """Test single video prediction."""
    # Setup mocks
    mock_engineer = MagicMock()
    # engineer.fit_transform should return a DataFrame with some columns
    mock_engineer.fit_transform.return_value = pd.DataFrame({
        "view_count": [1000], "like_count": [50], "duration_sec": [120], "subject_Maths": [1], "feature_5": [0]
    })
    mock_engineer_cls.return_value = mock_engineer
    
    mock_extractor = MagicMock()
    mock_extractor.get_embeddings.return_value = np.zeros((1, 768))
    mock_extractor_cls.return_value = mock_extractor
    
    # Initialize
    predictor = EngagementPredictor(model_dir=mock_artifacts)
    
    # Predict
    input_data = {
        "title": "Test Video",
        "description": "Test Descr",
        "duration_sec": 120,
        "view_count": 1000,
        "like_count": 50
    }
    result = predictor.predict(input_data)
    
    assert "engagement_score" in result
    assert result["engagement_score"] == 4.2
    assert "predicted_metric" in result

@patch("src.models.predict_model.TextEmbeddingExtractor")
@patch("src.features.engineer.VideoFeatureEngineer")
def test_predict_batch(mock_engineer_cls, mock_extractor_cls, mock_artifacts):
    """Test batch prediction."""
    mock_engineer = MagicMock()
    mock_engineer.fit_transform.return_value = pd.DataFrame({
        "view_count": [1000], "like_count": [50], "duration_sec": [120], "subject_Maths": [0], "feature_5": [0]
    })
    mock_engineer_cls.return_value = mock_engineer
    
    # Mock extractor
    mock_extractor = MagicMock()
    mock_extractor.get_embeddings.return_value = np.zeros((1, 768))
    mock_extractor_cls.return_value = mock_extractor
    
    predictor = EngagementPredictor(model_dir=mock_artifacts)
    
    df = pd.DataFrame([
        {"title": "V1", "description": "D1"},
        {"title": "V2", "description": "D2"}
    ])
    
    res_df = predictor.predict_batch(df)
    
    assert "predicted_score" in res_df.columns
    assert len(res_df) == 2
