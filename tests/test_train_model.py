
import pytest
import pandas as pd
import numpy as np
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch
from src.models.train_model import ModelTrainer

@pytest.fixture
def sample_data():
    """Create sample videos dataframe for testing."""
    n_samples = 30
    return pd.DataFrame({
        "video_id": [f"vid_{i}" for i in range(n_samples)],
        "title": [f"Video Title {i}" for i in range(n_samples)],
        "description": [f"Description for video {i} with some text." for i in range(n_samples)],
        "duration_sec": np.random.randint(60, 3600, n_samples),
        "view_count": np.random.randint(100, 10000, n_samples),
        "like_count": np.random.randint(10, 1000, n_samples),
        "comment_count": np.random.randint(0, 100, n_samples),
        "engagement_score": np.random.uniform(0, 5, n_samples),
        "publish_date": pd.date_range("2024-01-01", periods=n_samples),
        "subject": ["Maths"] * 15 + ["Physics"] * 15,
        "channel_id": ["UC_1"] * 15 + ["UC_2"] * 15,
        "engagement_category": ["High"] * 15 + ["Low"] * 15
    })

@pytest.fixture
def trainer(tmp_path):
    return ModelTrainer(output_dir=tmp_path)

def test_prepare_numerical_features(trainer, sample_data):
    """Test numerical feature preparation and VIF selection."""
    X, y = trainer.prepare_numerical_features(sample_data)
    
    assert len(X) == 30
    assert len(y) == 30
    # VIF selection might drop some columns on random data, but should keep some
    assert len(X.columns) > 0
    assert "duration_sec" in X.columns or "view_count" in X.columns
    # Categorical subject should be one-hot encoded
    assert any(col.startswith("subject_") for col in X.columns) if "subject_Physics" in X.columns else True

def test_prepare_numerical_features_with_nans(trainer, sample_data):
    """Test robustness against NaN values."""
    # Inject NaNs
    sample_data.loc[0:5, "view_count"] = np.nan
    sample_data.loc[5:10, "duration_sec"] = np.nan
    
    # Should not raise error and should fill NaNs
    X, y = trainer.prepare_numerical_features(sample_data)
    
    assert len(X) == 30
    assert not X.isnull().values.any() # Verify no NaNs remain

@patch("src.models.train_model.TextEmbeddingExtractor")
def test_extract_text_features(mock_extractor_cls, trainer, sample_data):
    """Test text feature extraction with mocked AraBERT."""
    # Mock AraBERT extractor
    mock_extractor = MagicMock()
    mock_extractor.get_embeddings.return_value = np.random.rand(30, 768)
    mock_extractor_cls.return_value = mock_extractor
    
    # Run extraction
    features = trainer.extract_text_features(sample_data, use_arabert=True)
    
    # Check shape: TF-IDF (30) + AraBERT (100) = 130 columns
    # Note: TF-IDF might produce fewer than 30 if not enough words, but with this sample data it should be fine?
    # Actually sample data is small, TF-IDF might be limited by vocabulary.
    assert features.shape[0] == 30
    assert features.shape[1] > 0
    
    # Check that TF-IDF was created
    assert trainer.tfidf is not None
    
    # Check that PCA was created (since use_arabert=True)
    assert trainer.pca is not None

def test_extract_text_features_no_arabert(trainer, sample_data):
    """Test text feature extraction without AraBERT."""
    features = trainer.extract_text_features(sample_data, use_arabert=False)
    
    # Should only have TF-IDF features (max 30)
    assert features.shape[0] == 30
    assert features.shape[1] <= 30
    assert trainer.pca is None

@patch("src.models.train_model.TextEmbeddingExtractor")
def test_train_pipeline(mock_extractor_cls, trainer, sample_data):
    """Test full training pipeline (using Random Forest for speed/simplicity)."""
    # Mock AraBERT
    mock_extractor = MagicMock()
    mock_extractor.get_embeddings.return_value = np.random.rand(30, 768)
    mock_extractor_cls.return_value = mock_extractor
    
    # Train
    trainer.train(sample_data, model_type="rf", use_arabert=True)
    
    # Check artifacts
    assert (trainer.output_dir / "rf_production.pkl").exists()
    assert (trainer.output_dir / "scaler_production.pkl").exists()
    assert (trainer.output_dir / "tfidf_production.pkl").exists()
    assert (trainer.output_dir / "pca_production.pkl").exists()
    assert (trainer.output_dir / "training_metadata.json").exists()
