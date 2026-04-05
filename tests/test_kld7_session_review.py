"""Tests for offline K-LD7 session review helpers."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from kld7_session_review_lib import analyze_session, load_session


SESSION_PATH = (
    Path(__file__).parent.parent
    / "session_logs"
    / "session_20260403_133805_range.jsonl"
)
NO_KLD7_SESSION_PATH = (
    Path(__file__).parent.parent
    / "session_logs"
    / "session_20260310_150412_range.jsonl"
)


def test_load_session_indexes_shots():
    """Session loader should index the expected shot records."""
    session_meta, shots = load_session(SESSION_PATH)

    assert session_meta["type"] == "session_start"
    assert session_meta["mode"] == "rolling-buffer"
    assert len(shots) == 10
    assert sorted(shots) == list(range(1, 11))
    assert "buffer" in shots[1]
    assert "capture" in shots[1]
    assert "shot" in shots[1]


def test_analyze_session_finds_recoverable_profiles():
    """The angle-offset range session should yield multiple strong profiles."""
    _, results = analyze_session(SESSION_PATH)
    quality_by_shot = {result.shot_number: result.quality for result in results}

    assert len(results) == 10
    assert sum(result.quality == "strong" for result in results) >= 4
    assert quality_by_shot[2] == "strong"
    assert quality_by_shot[8] == "weak"


def test_analyze_session_rejects_logs_without_kld7_buffers():
    """Session review should fail clearly when the session has no K-LD7 buffers."""
    with pytest.raises(ValueError, match="no kld7_buffer entries"):
        analyze_session(NO_KLD7_SESSION_PATH)
