"""Tests for ReviewState."""

from pathlib import Path

import pytest

from review_state import ReviewState


def test_review_state_get_set() -> None:
    state = ReviewState()
    assert state.get("/a.mp3") is None
    state.set("/a.mp3", "ok")
    assert state.get("/a.mp3") == "ok"
    state.set("/a.mp3", "swap")
    assert state.get("/a.mp3") == "swap"


def test_review_state_save_and_load(tmp_path: Path) -> None:
    state = ReviewState()
    state.set("/foo/track.mp3", "ok")
    state.set("/bar/other.cdg", "edit")
    path = tmp_path / "review-state.json"
    state.save(path)

    assert path.exists()
    loaded = path.read_text(encoding="utf-8")
    assert "ok" in loaded
    assert "edit" in loaded
    assert "by_path" in loaded

    state2 = ReviewState()
    state2.load(path)
    assert state2.get("/foo/track.mp3") == "ok"
    assert state2.get("/bar/other.cdg") == "edit"


def test_review_state_load_nonexistent() -> None:
    state = ReviewState()
    state.set("/x.mp3", "ok")
    state.load("/nonexistent/review-state.json")
    assert state.get("/x.mp3") is None


def test_review_state_load_invalid_version(tmp_path: Path) -> None:
    path = tmp_path / "review-state.json"
    path.write_text('{"version": 99, "by_path": {}}', encoding="utf-8")

    state = ReviewState()
    with pytest.raises(ValueError, match="Unsupported review-state version 99"):
        state.load(path)
