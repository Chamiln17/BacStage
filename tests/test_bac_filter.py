"""Tests for the Bac 3AS filter: decision rules, discovery, and the one-call filter."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.bac_filter_balanced import (
    BalancedBacFilter,
    build_channel_priors,
    discover_bac_terms,
    latest_snapshots,
    load_filter_config,
    run_bac_filter,
    score_against_labels,
    validation_sample,
)

ROOT = Path(__file__).resolve().parents[1]
MARKERS = {
    "bac": ["bac", "بكالوريا"],
    "non_bac": ["1as", "متوسط"],
    "strong_bac_intent": ["مراجعة بكالوريا"],
}
CONFIG = {
    "markers": MARKERS,
    "channel_prior": {"bac_threshold": 0.6, "non_bac_max": 0.2},
    "soft_positives": {"duration_min": 300, "require_duration": False},
    "tfidf": {"min_df": 1, "max_df": 1.0, "top_n_terms": 10, "analyzer": "word", "ngram_range": [1, 1]},
}


@pytest.fixture
def bac_filter() -> BalancedBacFilter:
    return BalancedBacFilter(
        bac_markers=MARKERS["bac"],
        non_bac_markers=MARKERS["non_bac"],
        strong_bac_intent=MARKERS["strong_bac_intent"],
        bac_heavy_channels={"UC_heavy"},
        tfidf_terms=["الدالة"],
        channel_subjects={"UC_heavy": "Maths"},
        duration_min=300,
    )


@pytest.mark.parametrize(
    "title, channel, duration, is_bac, category",
    [
        ("درس 1as الدوال", "UC_other", 900, False, "non_bac"),  # rule 1: non-Bac marker
        ("مراجعة بكالوريا 1as", "UC_other", 900, True, "bac_3as"),  # rule 2: conflict, strong intent wins
        ("bac متوسط", "UC_other", 900, False, "non_bac"),  # rule 2: conflict, no strong intent
        ("تحضير بكالوريا 2025", "UC_other", 900, True, "bac_3as"),  # rule 3: explicit Bac marker
        ("الدالة الأسية", "UC_other", 900, False, "unknown"),  # rule 4: no markers, channel not Bac-heavy
        ("الدالة الأسية", "UC_heavy", 900, True, "bac_3as_ambiguous"),  # rule 4: Bac-heavy + soft positive
    ],
)
def test_filter_rules(bac_filter, title, channel, duration, is_bac, category) -> None:
    result = bac_filter.filter_video(title, "", "", channel, duration)
    assert (result["is_bac_3as"], result["filter_category"]) == (is_bac, category)


def test_required_duration_is_a_hard_gate(bac_filter) -> None:
    bac_filter.require_duration = True
    result = bac_filter.filter_video("تحضير بكالوريا", "", "", "UC_heavy", 60)
    assert result["is_bac_3as"] is False and "too short" in result["filter_reason"]


def _videos() -> pd.DataFrame:
    rows = [("UC_heavy", f"بكالوريا الدالة {i}") for i in range(60)]
    rows += [("UC_heavy", "الدالة الأسية شرح")]  # no grade marker, Bac-heavy channel
    rows += [("UC_mixed", f"متوسط درس {i}") for i in range(25)]
    rows += [("UC_mixed", "bac تمرين")]
    return pd.DataFrame(
        {
            "video_id": [f"v{i}" for i in range(len(rows))],
            "channel_id": [c for c, _ in rows],
            "title": [t for _, t in rows],
            "description": "",
            "tags": "",
            "duration_sec": 900,
        }
    )


def test_channel_priors() -> None:
    priors = build_channel_priors(_videos(), CONFIG).set_index("channel_id")
    assert priors.loc["UC_heavy", "count_bac_marked"] == 60
    assert bool(priors.loc["UC_heavy", "is_bac_heavy"]) is True
    assert bool(priors.loc["UC_mixed", "is_bac_heavy"]) is False


def test_discovered_terms_favour_bac_titles() -> None:
    terms = discover_bac_terms(_videos(), CONFIG)
    assert "الدالة" in terms
    assert "درس" not in terms


def test_discovery_needs_enough_marked_titles() -> None:
    assert discover_bac_terms(_videos().iloc[:30], CONFIG) == []


def test_run_bac_filter_end_to_end() -> None:
    result = run_bac_filter(_videos(), CONFIG, {"UC_heavy": "Maths"})
    by_title = result.videos.set_index("title")
    assert by_title.loc["الدالة الأسية شرح", "filter_category"] == "bac_3as_ambiguous"
    assert by_title.loc["الدالة الأسية شرح", "subject"] == "Maths"
    assert not by_title.loc["متوسط درس 0", "is_bac_3as"]
    assert result.terms


def test_cached_priors_and_terms_skip_discovery() -> None:
    priors = pd.DataFrame({"channel_id": ["UC_mixed"], "is_bac_heavy": [True]})
    result = run_bac_filter(_videos(), CONFIG, {}, priors=priors, terms=[])
    assert result.priors is priors and result.terms == []
    ambiguous = result.videos.set_index("title").loc["الدالة الأسية شرح"]
    assert ambiguous["filter_category"] == "unknown"  # UC_heavy is no longer Bac-heavy


def test_latest_snapshots() -> None:
    videos = pd.DataFrame(
        {"video_id": ["a", "a", "b"], "snapshot_date": ["2026-01-01", "2026-01-05", "2026-01-02"], "view_count": [1, 5, 2]}
    )
    latest = latest_snapshots(videos).set_index("video_id")
    assert latest["view_count"].to_dict() == {"a": 5, "b": 2}


def test_validation_sample_leaves_label_columns_empty() -> None:
    filtered = run_bac_filter(_videos(), CONFIG, {}).videos
    sample = validation_sample(filtered, {"validation": {"sample_bac_3as": 5, "sample_non_bac": 3}})
    assert (sample["sample_type"] == "bac_3as").sum() == 5
    assert set(sample["manual_is_bac"]) == {""}


def test_score_against_labels() -> None:
    labels = pd.DataFrame(
        {"is_bac_3as": [True, True, False, False, True], "manual_is_bac": ["True", "False", "False", "True", ""]}
    )
    assert score_against_labels(labels) == {"labelled": 4, "precision": 0.5, "recall": 0.5, "accuracy": 0.5}


def test_config_is_required(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_filter_config(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("markers:\n  bac: [bac]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="non_bac"):
        load_filter_config(bad)


def test_repo_config_loads() -> None:
    config = load_filter_config(ROOT / "config" / "filter_config.yaml")
    assert "الأعداد المركبة" in config["markers"]["bac"]
