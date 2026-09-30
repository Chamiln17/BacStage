"""Train → save → load → predict round trips, through the library and the real CLI."""

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.models.model_utils import load_bundle
from src.models.predict_model import EngagementPredictor
from src.models.train_model import train

ROOT = Path(__file__).resolve().parents[1]
PLANNED = {"title": "مراجعة بكالوريا الأعداد المركبة", "duration_sec": 1200, "channel_id": "UC_math"}


def test_train_then_predict_with_embeddings(tmp_path, training_videos, fake_embedder) -> None:
    test_ids = training_videos["video_id"].iloc[::5]
    metadata = train(
        training_videos, tmp_path, model_type="rf", embedder=fake_embedder,
        embedding_model="fake", test_ids=test_ids,
    )
    assert metadata["n_test"] == len(test_ids)
    assert metadata["split"] == "given test video_ids"
    assert set(metadata["test_metrics"]) == {"mae", "mse", "rmse", "r2"}
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))["model_type"] == "rf"

    engineer, _, _ = load_bundle(tmp_path)
    assert engineer.embedder is None  # never pickled

    predictor = EngagementPredictor(tmp_path, embedder=fake_embedder)
    result = predictor.predict(PLANNED)
    assert set(result) == {"engagement_score", "engagement_category", "known_channel"}
    assert result["engagement_category"] in {"Low", "Medium", "High"}
    assert result["known_channel"] is True

    batch = predictor.predict_batch(training_videos.drop(columns=["view_count", "like_count", "comment_count"]))
    assert batch["predicted_engagement_score"].notna().all()


def test_embedder_is_called_once_per_text_during_training(tmp_path, training_videos, fake_embedder) -> None:
    train(training_videos, tmp_path, model_type="rf", embedder=fake_embedder, embedding_model="fake")
    # Fit + transforms on train, test and all rows reuse the cache: one call for
    # the training split's texts, one for the test split's, none after that.
    assert fake_embedder.calls == 2


def test_unviewed_videos_are_not_trained_on(tmp_path, training_videos) -> None:
    videos = training_videos.copy()
    videos.loc[:9, "view_count"] = 0
    metadata = train(videos, tmp_path, model_type="rf")
    assert metadata["n_train"] + metadata["n_test"] == len(videos) - 10


def test_missing_bundle_fails_clearly(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="run_pipeline.py train"):
        EngagementPredictor(tmp_path)


def _cli(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(ROOT / "run_pipeline.py"), *args],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )


def test_cli_train_then_predict(tmp_path, training_videos) -> None:
    """The documented commands work end to end (the train import used to crash)."""
    videos_csv, transcripts_csv = tmp_path / "videos.csv", tmp_path / "transcripts.csv"
    training_videos.drop(columns="transcript_text").to_csv(videos_csv, index=False)
    training_videos[["video_id", "transcript_text"]].dropna().to_csv(transcripts_csv, index=False)
    model_dir = tmp_path / "model"

    trained = _cli("train", "--input", str(videos_csv), "--transcripts", str(transcripts_csv),
                   "--output-dir", str(model_dir), "--model-type", "rf", "--no-arabert")
    assert trained.returncode == 0, trained.stderr + trained.stdout
    assert (model_dir / "model.joblib").exists()

    planned_json, out_json = tmp_path / "planned.json", tmp_path / "out.json"
    planned_json.write_text(json.dumps(PLANNED, ensure_ascii=False), encoding="utf-8")
    predicted = _cli("predict", "--input", str(planned_json), "--model-dir", str(model_dir),
                     "--output", str(out_json))
    assert predicted.returncode == 0, predicted.stderr + predicted.stdout
    assert json.loads(out_json.read_text(encoding="utf-8"))["known_channel"] is True

    batch_csv = tmp_path / "planned.csv"
    pd.DataFrame([PLANNED, {**PLANNED, "channel_id": "UC_new", "subject": "Maths"}]).to_csv(batch_csv, index=False)
    out_csv = tmp_path / "out.csv"
    predicted = _cli("predict", "--input", str(batch_csv), "--model-dir", str(model_dir), "--output", str(out_csv))
    assert predicted.returncode == 0, predicted.stderr + predicted.stdout
    assert pd.read_csv(out_csv)["known_channel"].tolist() == [True, False]
