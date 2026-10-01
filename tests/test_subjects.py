"""Every place that names a subject must use the canonical spellings."""

from pathlib import Path

import pandas as pd
import pytest

from src.features.bac_keywords import (
    SUBJECT_KEYWORDS,
    SUBJECTS,
    canonical_subject,
    count_domain_keywords,
)

ROOT = Path(__file__).resolve().parents[1]
CHANNELS = ROOT / "data" / "raw" / "channels.csv"
KNOWLEDGE_BASE = ROOT / "knowledge_base"


@pytest.mark.parametrize(
    "name, canonical",
    [
        ("Maths", "Maths"),
        ("maths ", "Maths"),
        ("Mathematics", "Maths"),
        ("Math", "Maths"),
        ("Science", "Natural Sciences"),
        ("History", "History & Geography"),
        ("history & geography", "History & Geography"),
    ],
)
def test_aliases(name: str, canonical: str) -> None:
    assert canonical_subject(name) == canonical


def test_unknown_subject_raises() -> None:
    with pytest.raises(ValueError, match="Unknown subject"):
        canonical_subject("Chemistry")


# Collected data and the knowledge base are not published (ADR 0001); these run locally.
@pytest.mark.skipif(not CHANNELS.exists(), reason="data/raw/channels.csv is not in the public repo")
def test_channels_csv_uses_canonical_subjects() -> None:
    subjects = pd.read_csv(CHANNELS)["subjects"]
    assert set(subjects.str.strip()) <= set(SUBJECTS)


@pytest.mark.skipif(not KNOWLEDGE_BASE.exists(), reason="knowledge_base/ is not in the public repo")
def test_every_subject_has_a_knowledge_base_file() -> None:
    names = {p.name.removesuffix("_best_practices.json") for p in KNOWLEDGE_BASE.glob("*_best_practices.json")}
    assert set(SUBJECTS) <= names


def test_keyword_lists_use_canonical_names() -> None:
    assert set(SUBJECT_KEYWORDS) <= set(SUBJECTS)


def test_maths_uses_the_maths_list() -> None:
    # "التكامل" is a Maths keyword; "المناعة" is Natural Sciences only.
    assert count_domain_keywords("التكامل والمناعة", "Maths") == 1
    assert count_domain_keywords("التكامل والمناعة", "Arabic") == 2  # no list: all subjects
