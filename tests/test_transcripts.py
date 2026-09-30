"""Tests for the transcript module: validity rule and feature extraction."""

import pytest

from src.data.transcript_collector import clean_vtt_text
from src.features.transcript_features import (
    _empty_features,
    extract_transcript_features,
    is_valid_transcript,
)

PAGE_CODE = 'window.WIZ_global_data = {"a": 1}; var ytcfg = {d: function() {}};'


@pytest.mark.parametrize(
    "text, valid",
    [
        ("سنشرح اليوم الدالة الأسية خطوة بخطوة", True),
        ("On commence par la dérivée. Ensuite on étudie les variations.", True),
        ("", False),
        ("   ", False),
        (None, False),
        (float("nan"), False),
        (PAGE_CODE, False),
        # One marker alone is speech that happens to contain it, not page code.
        ("the javascript course starts now", True),
    ],
)
def test_is_valid_transcript(text: object, valid: bool) -> None:
    assert is_valid_transcript(text) is valid


def test_invalid_transcript_yields_empty_features() -> None:
    assert extract_transcript_features(PAGE_CODE, 600) == _empty_features()
    assert extract_transcript_features("", 600) == _empty_features()


def test_valid_transcript_features() -> None:
    text = "What is a derivative? Because the slope changes. For example, x squared."
    f = extract_transcript_features(text, duration_sec=6)

    assert f["transcript_word_count"] == 12
    assert f["transcript_sentence_count"] == 3
    assert f["question_count"] == 1
    assert f["speech_rate_wpm"] == pytest.approx(120.0)
    assert f["speech_rate_optimal"] == 1
    assert set(f) == set(_empty_features())


def test_speech_rate_needs_duration() -> None:
    f = extract_transcript_features("some real words here", duration_sec=None)
    assert f["speech_rate_wpm"] is None


def test_clean_vtt_text() -> None:
    vtt = "WEBVTT\n\n00:00:01.000 --> 00:00:05.000\nHello world\n"
    assert clean_vtt_text(vtt) == "Hello world"
    assert not is_valid_transcript(clean_vtt_text(PAGE_CODE))
