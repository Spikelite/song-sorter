"""main.py: Key-detect orchestration (detect_keys) and its helpers.

The per-track worker (key_detect.analyze_local) and the online lookup
(key_online.lookup_online) are stubbed at their module seams; the key logic they
feed is tested in test_key_detect*.py. ProcessPoolExecutor is swapped for a
thread pool so the parallel path runs in-process and deterministically.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import key_detect
import key_online
from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


@pytest.fixture
def kd(m, monkeypatch):
    state = {"local": {}, "online": {}, "online_calls": [], "worker_error": set()}

    def fake_analyze(path, file_types):
        if path in state["worker_error"]:
            raise RuntimeError("worker died")
        return state["local"].get(path, {"tag": None, "offline": None})

    def fake_lookup(artist, title, cache=None):
        state["online_calls"].append((artist, title))
        out = state["online"].get((artist, title), (None, "no match"))
        if isinstance(out, BaseException):
            raise out
        return out

    monkeypatch.setattr(key_detect, "analyze_local", fake_analyze)
    monkeypatch.setattr(key_online, "lookup_online", fake_lookup)
    monkeypatch.setattr(m, "_is_online", lambda *a, **k: True)
    monkeypatch.setattr(key_detect, "HAVE_LIBROSA", True)
    monkeypatch.setattr(m, "ProcessPoolExecutor", ThreadPoolExecutor)
    return state


def _t(path, artist="Abba", song="Waterloo", **md):
    md.setdefault("mp3_hash", f"h-{path}")
    return make_track(path, artist, song, file_types=("zip",), **md)


# Answers for the common prompt path: online? recheck? export-only? workers?
def _run(prompts, *, online=True, recheck=False, export_only=True, workers="1"):
    prompts.queue(online, recheck, export_only, workers)


# --- _export_candidates ----------------------------------------------------

def test_export_candidates_mirror_final_final(m, store) -> None:
    store.add(_t("/big", mp3_size=900))
    store.add(_t("/mid", mp3_size=500))
    store.add(_t("/small", mp3_size=100))
    store.add(_t("/unk", artist="Unknown", song="X"))
    store.add(_t("/blank", artist="", song="Y"))
    assert [t.path for t in m._export_candidates(store, 1)] == ["/big"]
    assert [t.path for t in m._export_candidates(store, 2)] == ["/big", "/mid"]
    assert [t.path for t in m._export_candidates(store, 0)] == ["/big"]   # floor of 1


def test_export_candidates_on_an_empty_store(m, store) -> None:
    assert m._export_candidates(store, 3) == []


# --- _apply_key_result -----------------------------------------------------

def test_apply_key_result_writes_and_clears(m) -> None:
    t = _t("/a")
    m._apply_key_result(t, {"key": "A minor", "confidence": 0.8, "source": "auto",
                            "detail": "d"}, "sig1")
    assert t.metadata["key"] == "A minor"
    assert t.metadata["key_confidence"] == "0.800"
    assert t.metadata["key_camelot"] == "8A"
    assert t.metadata["key_sig"] == "sig1"
    m._apply_key_result(t, {"key": None, "confidence": 0.0, "source": "none"}, "sig2")
    for k in ("key", "key_confidence", "key_camelot"):
        assert k not in t.metadata
    assert t.metadata["key_source"] == "none"
    assert t.metadata["key_detail"] == ""


def test_apply_key_result_drops_a_stale_camelot(m) -> None:
    t = _t("/a", key_camelot="8A")
    m._apply_key_result(t, {"key": "not a key", "confidence": 1.0, "source": "manual"}, "s")
    assert "key_camelot" not in t.metadata


# --- detect_keys: setup branches -------------------------------------------

def test_offline_and_no_librosa_can_be_declined(m, store, prompts, kd, monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_online", lambda *a, **k: False)
    monkeypatch.setattr(key_detect, "HAVE_LIBROSA", False)
    store.add(_t("/a"))
    prompts.queue(False)                      # "only tags/overrides. Continue?" -> no
    m.detect_keys(store)
    assert "librosa not installed" in prompts.output
    assert "unreachable" in prompts.output
    assert "key_source" not in store.get("/a").metadata


def test_tags_only_run_when_nothing_else_is_available(m, store, prompts, kd, monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_online", lambda *a, **k: False)
    monkeypatch.setattr(key_detect, "HAVE_LIBROSA", False)
    store.add(_t("/a"))
    kd["local"]["/a"] = {"tag": "Am", "offline": None}
    prompts.queue(True, False, True)          # continue, no recheck, export-only
    m.detect_keys(store)                      # no workers prompt without librosa
    assert store.get("/a").metadata["key_source"] == "tag"


@pytest.mark.parametrize("cancel_at", ["export_only", "workers"])
def test_cancelling_a_setup_prompt_aborts(m, store, prompts, kd, cancel_at) -> None:
    store.add(_t("/a"))
    if cancel_at == "export_only":
        prompts.queue(True, False, None)
    else:
        prompts.queue(True, False, True, None)
    m.detect_keys(store)
    assert "key_source" not in store.get("/a").metadata


def test_nothing_to_do(m, store, prompts, kd) -> None:
    store.add(make_track("/cdg", "Abba", "SOS", file_types=("cdg",)))   # no audio
    prompts.queue(True, False, True)
    m.detect_keys(store)
    assert "No tracks need key detection" in prompts.output


def test_warns_about_tracks_without_mp3_hash(m, store, prompts, kd) -> None:
    store.add(make_track("/a", "Abba", "Waterloo", file_types=("zip",)))
    _run(prompts)
    m.detect_keys(store)
    assert "1 track(s) have no mp3_hash" in prompts.output


def test_bad_worker_count_falls_back_to_the_default(m, store, prompts, kd) -> None:
    store.add(_t("/a"))
    kd["local"]["/a"] = {"tag": None, "offline": ("C major", 0.9)}
    _run(prompts, workers="lots")
    m.detect_keys(store)
    assert store.get("/a").metadata["key"] == "C major"


# --- detect_keys: signal fusion end to end ---------------------------------

def test_confident_offline_read_skips_the_network(m, store, prompts, kd) -> None:
    store.add(_t("/a"))
    kd["local"]["/a"] = {"tag": None, "offline": ("E minor", 0.9)}
    _run(prompts)
    m.detect_keys(store)
    md = store.get("/a").metadata
    assert (md["key"], md["key_source"], md["key_sig"]) == ("E minor", "auto", "h-/a")
    assert kd["online_calls"] == []                        # above the emit floor
    assert "Scoped to 1 exported copies" in prompts.output


def test_weak_offline_read_is_corroborated_once_per_pair(m, store, prompts, kd) -> None:
    store.add(_t("/a", mp3_size=9))
    store.add(_t("/b", mp3_size=5))                        # same song, second copy
    for p in ("/a", "/b"):
        kd["local"][p] = {"tag": None, "offline": ("E minor", 0.3)}
    kd["online"][("Abba", "Waterloo")] = ("E minor", "acousticbrainz x")
    _run(prompts, export_only=False)                        # key every copy
    m.detect_keys(store)
    assert kd["online_calls"] == [("Abba", "Waterloo")]    # memoised per pair
    assert all(store.get(p).metadata["key_confidence"] == "0.500" for p in ("/a", "/b"))


def test_online_fills_when_there_is_no_local_signal(m, store, prompts, kd) -> None:
    store.add(_t("/a"))
    kd["online"][("Abba", "Waterloo")] = ("A major", "acousticbrainz x")
    _run(prompts)
    m.detect_keys(store)
    assert store.get("/a").metadata["key_source"] == "online"


def test_online_lookup_failure_degrades_to_local(m, store, prompts, kd) -> None:
    store.add(_t("/a"))
    kd["local"]["/a"] = {"tag": "G", "offline": None}
    kd["online"][("Abba", "Waterloo")] = RuntimeError("network")
    _run(prompts)
    m.detect_keys(store)
    assert store.get("/a").metadata["key_source"] == "tag"


def test_declining_online_keeps_it_off(m, store, prompts, kd) -> None:
    store.add(_t("/a"))
    _run(prompts, online=False)
    m.detect_keys(store)
    assert kd["online_calls"] == []


def test_override_wins_without_touching_audio(m, store, prompts, kd) -> None:
    store.add(_t("/a"))
    kd["worker_error"].add("/a")                            # proves no analysis ran
    m._KEY_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    m._KEY_OVERRIDES_PATH.write_text(json.dumps({"Abba - Waterloo": "Dm"}), encoding="utf-8")
    _run(prompts)
    m.detect_keys(store)
    md = store.get("/a").metadata
    assert (md["key"], md["key_source"]) == ("D minor", "manual")
    assert "manual 1" in prompts.output


def test_an_override_reapplies_even_to_an_already_keyed_track(m, store, prompts, kd) -> None:
    store.add(_t("/a", key_sig="h-/a", key_source="auto", key="E minor",
                 key_confidence="0.95"))
    m._KEY_OVERRIDES_PATH.parent.mkdir(parents=True, exist_ok=True)
    m._KEY_OVERRIDES_PATH.write_text(json.dumps({"Abba - Waterloo": "C"}), encoding="utf-8")
    _run(prompts)
    m.detect_keys(store)
    assert store.get("/a").metadata["key"] == "C major"


# --- detect_keys: incremental / recheck ------------------------------------

def test_already_keyed_tracks_are_skipped(m, store, prompts, kd) -> None:
    store.add(_t("/a", key_sig="h-/a", key_source="auto", key_confidence="0.9"))
    prompts.queue(True, False, True)
    m.detect_keys(store)
    assert "No tracks need key detection" in prompts.output


def test_recheck_revisits_only_weak_results(m, store, prompts, kd) -> None:
    store.add(_t("/none", song="A", key_sig="h-/none", key_source="none"))
    store.add(_t("/online", song="B", key_sig="h-/online", key_source="online"))
    store.add(_t("/lowauto", song="C", key_sig="h-/lowauto", key_source="auto",
                 key_confidence="0.2"))
    store.add(_t("/junk", song="D", key_sig="h-/junk", key_source="auto",
                 key_confidence="n/a"))                      # unparseable -> 0.0
    store.add(_t("/good", song="E", key_sig="h-/good", key_source="auto",
                 key_confidence="0.9"))
    store.add(_t("/tag", song="F", key_sig="h-/tag", key_source="tag"))
    _run(prompts, recheck=True)
    m.detect_keys(store)
    assert "processed 4" in prompts.output


# --- detect_keys: parallel path, large-run heads-up, checkpoints -----------

def test_parallel_path_survives_a_crashing_worker(m, store, prompts, kd) -> None:
    store.add(_t("/ok", song="A"))
    store.add(_t("/dead", song="B"))
    kd["local"]["/ok"] = {"tag": None, "offline": ("F major", 0.9)}
    kd["worker_error"].add("/dead")
    _run(prompts, online=False, workers="2")
    m.detect_keys(store)
    assert store.get("/ok").metadata["key"] == "F major"
    assert store.get("/dead").metadata["key_source"] == "none"   # recorded, not lost


def test_large_online_run_warns_and_can_switch_online_off(m, store, prompts, kd) -> None:
    for i in range(501):
        store.add(_t(f"/t{i}", song=f"Song {i}"))
    prompts.queue(True, False, True, False, "1")   # ... keep online on? -> no
    m.detect_keys(store)
    assert "Heads-up: online corroboration" in prompts.output
    assert "Online corroboration off for this run" in prompts.output
    assert kd["online_calls"] == []


def test_large_online_run_can_keep_online_on(m, store, prompts, kd) -> None:
    for i in range(501):
        store.add(_t(f"/t{i}", song=f"Song {i}"))
    prompts.queue(True, False, True, True, "1")
    m.detect_keys(store)
    assert len(kd["online_calls"]) == 501


def test_long_runs_checkpoint(m, store, prompts, kd, monkeypatch) -> None:
    class Clock:
        t = 0.0

        def monotonic(self):
            Clock.t += 1000.0
            return Clock.t
    monkeypatch.setattr(m, "time", Clock())
    store.add(_t("/a"))
    kd["local"]["/a"] = {"tag": None, "offline": ("C major", 0.9)}
    _run(prompts)
    m.detect_keys(store)
    assert m._CACHE_PATH.exists()
