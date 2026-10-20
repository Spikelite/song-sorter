"""Tests for key_detect's optional IO layer, with librosa stubbed.

librosa itself is not installed in CI (it pulls in numba/scipy, and the DSP maths
is librosa's to get right). What is ours, and tested here, is everything around
it: the first-verse window, the short-track retry, the silence gate, the
decode-failure fallback, the empty-profile guard, the fd-level stderr
suppression, and the analyze_local worker entry point.
"""

from __future__ import annotations

import importlib
import os
import sys
import types

import pytest

# numpy comes from requirements-dev.txt; without it (e.g. a bare local venv)
# skip this module rather than break collection for the whole suite
np = pytest.importorskip("numpy")

import key_detect
from key_detect import KEY_NAMES, _KS_MINOR

SR = 22050


def _loud(seconds: float = 2.0):
    return np.full(int(SR * seconds), 0.5, dtype=np.float32)


def _chroma_for_key(profile=_KS_MINOR, tonic="A", frames=8):
    """A (12, frames) chromagram whose per-bin mean IS a rotated key template."""
    pc = KEY_NAMES.index(tonic)
    col = np.array([profile[(i - pc) % 12] for i in range(12)], dtype=np.float32)
    return np.tile(col[:, None], (1, frames))


class FakeLibrosa:
    """Scripted stand-in: `loads` is the (y, sr) returned by each load() call."""

    def __init__(self, loads=(), chroma=None, load_error=None):
        self._loads = list(loads)
        self._chroma = _chroma_for_key() if chroma is None else chroma
        self._load_error = load_error
        self.calls: list[dict] = []
        self.feature = types.SimpleNamespace(chroma_cqt=self._chroma_cqt)

    def load(self, path, mono=True, offset=None, duration=None):
        self.calls.append({"path": path, "mono": mono,
                           "offset": offset, "duration": duration})
        if self._load_error is not None:
            raise self._load_error
        return self._loads.pop(0)

    def _chroma_cqt(self, y, sr):
        return self._chroma


@pytest.fixture
def librosa_stub(monkeypatch):
    def install(fake):
        monkeypatch.setattr(key_detect, "librosa", fake, raising=False)
        monkeypatch.setattr(key_detect, "_np", np, raising=False)
        monkeypatch.setattr(key_detect, "HAVE_LIBROSA", True)
        return fake
    return install


# --- detect_key_offline ----------------------------------------------------

def test_returns_none_without_librosa(monkeypatch) -> None:
    monkeypatch.setattr(key_detect, "HAVE_LIBROSA", False)
    assert key_detect.detect_key_offline("x.mp3") is None


def test_detects_key_from_first_verse_window(librosa_stub) -> None:
    fake = librosa_stub(FakeLibrosa(loads=[(_loud(), SR)]))
    key, conf = key_detect.detect_key_offline("song.mp3")
    assert key == "A minor"
    assert conf > 0
    # analysed the first verse, skipping the lead-in, not the whole track
    assert fake.calls == [{"path": "song.mp3", "mono": True,
                           "offset": key_detect._OFFSET_SECONDS,
                           "duration": key_detect._WINDOW_SECONDS}]


def test_short_track_retries_from_the_start(librosa_stub) -> None:
    """A track shorter than the lead-in offset decodes to (almost) nothing, so
    the detector must retry from 0 rather than give up."""
    fake = librosa_stub(FakeLibrosa(loads=[(np.full(10, 0.5), SR), (_loud(), SR)]))
    assert key_detect.detect_key_offline("short.mp3")[0] == "A minor"
    assert [c["offset"] for c in fake.calls] == [key_detect._OFFSET_SECONDS, None]


def test_none_decode_also_retries(librosa_stub) -> None:
    fake = librosa_stub(FakeLibrosa(loads=[(None, SR), (_loud(), SR)]))
    assert key_detect.detect_key_offline("x.mp3") is not None
    assert len(fake.calls) == 2


def test_too_short_even_from_start_is_none(librosa_stub) -> None:
    librosa_stub(FakeLibrosa(loads=[(np.full(10, 0.5), SR), (np.full(10, 0.5), SR)]))
    assert key_detect.detect_key_offline("tiny.mp3") is None


def test_silent_window_is_rejected(librosa_stub) -> None:
    """Silence has no pitch content; a key derived from it would be noise."""
    librosa_stub(FakeLibrosa(loads=[(np.zeros(SR * 2), SR)]))
    assert key_detect.detect_key_offline("silent.mp3") is None


def test_decode_failure_degrades_to_none(librosa_stub) -> None:
    librosa_stub(FakeLibrosa(load_error=RuntimeError("no backend")))
    assert key_detect.detect_key_offline("broken.mp3") is None


def test_all_zero_profile_is_rejected(librosa_stub) -> None:
    librosa_stub(FakeLibrosa(loads=[(_loud(), SR)], chroma=np.zeros((12, 4))))
    assert key_detect.detect_key_offline("x.mp3") is None


# --- _quiet_native_stderr edge cases ---------------------------------------

def test_quiet_stderr_is_a_noop_when_fd2_is_unusable(monkeypatch) -> None:
    """No usable descriptor to redirect: must still yield, not raise."""
    monkeypatch.setattr(key_detect, "_STDERR_FD", 987654)   # EBADF on dup
    entered = []
    with key_detect._quiet_native_stderr():
        entered.append(True)
    assert entered == [True]


def test_quiet_stderr_survives_a_stderr_that_cannot_flush(monkeypatch, capfd) -> None:
    class NoFlush:
        def flush(self):
            raise OSError("closed")
        def write(self, s):
            return len(s)
    monkeypatch.setattr(sys, "stderr", NoFlush())
    with key_detect._quiet_native_stderr():
        os.write(2, b"HIDDEN")
    _out, err = capfd.readouterr()
    assert "HIDDEN" not in err


# --- analyze_local (the process-pool worker) --------------------------------

def test_analyze_local_reads_tag_and_offline_from_a_real_mp3(tmp_path, librosa_stub) -> None:
    pytest.importorskip("track_inspect")
    from conftest import write_mp3
    write_mp3(tmp_path / "A - S.mp3", initialkey="Em")
    librosa_stub(FakeLibrosa(loads=[(_loud(), SR)]))
    out = key_detect.analyze_local(str(tmp_path / "A - S.cdg"), ["mp3", "cdg"])
    assert out["tag"] == "Em"
    assert out["offline"][0] == "A minor"


def test_analyze_local_without_librosa_still_reads_the_tag(tmp_path, monkeypatch) -> None:
    pytest.importorskip("track_inspect")
    from conftest import write_mp3
    monkeypatch.setattr(key_detect, "HAVE_LIBROSA", False)
    write_mp3(tmp_path / "A - S.mp3", initialkey="C")
    out = key_detect.analyze_local(str(tmp_path / "A - S.cdg"), ["mp3"])
    assert out == {"tag": "C", "offline": None}


def test_analyze_local_survives_a_detector_that_raises(tmp_path, monkeypatch) -> None:
    """The worker contract is 'never raise': a blowing-up detector must still
    yield a result dict, keeping whatever was already read (the tag)."""
    pytest.importorskip("track_inspect")
    from conftest import write_mp3
    write_mp3(tmp_path / "A - S.mp3", initialkey="G")

    def boom(path):
        raise MemoryError("decoder exploded")
    monkeypatch.setattr(key_detect, "detect_key_offline", boom)
    out = key_detect.analyze_local(str(tmp_path / "A - S.cdg"), ["mp3"])
    assert out == {"tag": "G", "offline": None}


# --- the guarded import itself ---------------------------------------------

def test_librosa_is_detected_when_importable() -> None:
    """Covers the import-success branch: with a librosa importable, the module
    reports HAVE_LIBROSA and binds numpy for the detector."""
    fake = types.ModuleType("librosa")
    sys.modules["librosa"] = fake
    try:
        reloaded = importlib.reload(key_detect)
        assert reloaded.HAVE_LIBROSA is True
        assert reloaded._np is np
    finally:
        del sys.modules["librosa"]
        importlib.reload(key_detect)
        # reload re-executes in the SAME namespace, so bindings from the
        # successful import survive it; drop them to leave the module pristine
        for stale in ("librosa", "_np"):
            key_detect.__dict__.pop(stale, None)
    assert key_detect.HAVE_LIBROSA is False
