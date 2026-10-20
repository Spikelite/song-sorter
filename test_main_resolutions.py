"""main.py: Apply-resolutions and Unify-artists.

Both rewrite library data in bulk from a curated file, behind a dry run, so the
central property is pinned first: a dry run changes nothing on disk or in memory.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload),
                    encoding="utf-8")


def _review(m):
    from review_state import ReviewState
    rs = ReviewState()
    rs.load(m._REVIEW_STATE_PATH)
    return rs


# --- apply_resolutions ------------------------------------------------------

def test_resolutions_missing_file(m, store, prompts) -> None:
    m.apply_resolutions(store)
    assert "No resolutions file" in prompts.output


def test_resolutions_unreadable_file(m, store, prompts) -> None:
    _write(m._RESOLUTIONS_PATH, "{broken")
    m.apply_resolutions(store)
    assert "Could not read resolutions" in prompts.output


def test_resolutions_with_no_entries(m, store, prompts) -> None:
    _write(m._RESOLUTIONS_PATH, {"version": 1, "resolutions": {}})
    m.apply_resolutions(store)
    assert "has no entries" in prompts.output


def test_resolutions_dry_run_cancelled(m, store, prompts) -> None:
    store.add(make_track("/a", "Waterloo", "Abba"))
    _write(m._RESOLUTIONS_PATH, {"resolutions": {"/a": {"artist": "Abba", "song": "Waterloo"}}})
    prompts.queue(None)
    m.apply_resolutions(store)
    assert store.get("/a").artist == "Waterloo"


def _resolutions(m, store):
    store.add(make_track("/fix", "Waterloo", "Abba"))
    store.add(make_track("/part", "Abba", "Waterlo"))
    store.add(make_track("/same", "Abba", "SOS"))
    store.add(make_track("/same-unreviewed", "Abba", "Fernando"))
    from review_state import ReviewState
    rs = ReviewState()
    rs.set("/same", "ok")
    rs.save(m._REVIEW_STATE_PATH)
    _write(m._RESOLUTIONS_PATH, {"version": 1, "resolutions": {
        "/fix": {"artist": "Abba", "song": "Waterloo", "why": "reversed"},
        "/part": {"song": "Waterloo"},                     # artist omitted: kept
        "/same": {"artist": "Abba", "song": "SOS"},        # already set and ok'd
        "/same-unreviewed": {"artist": "Abba", "song": "Fernando"},
        "/not-in-store": {"artist": "X", "song": "Y"},
    }})


def test_resolutions_dry_run_changes_nothing(m, store, prompts) -> None:
    _resolutions(m, store)
    before = {t.path: (t.artist, t.song, dict(t.metadata)) for t in store.all()}
    rs_before = m._REVIEW_STATE_PATH.read_text(encoding="utf-8")
    prompts.queue(True)
    m.apply_resolutions(store)
    assert {t.path: (t.artist, t.song, dict(t.metadata)) for t in store.all()} == before
    assert m._REVIEW_STATE_PATH.read_text(encoding="utf-8") == rs_before
    assert not m._CACHE_PATH.exists()
    assert "[dry] fix" in prompts.output and "(reversed)" in prompts.output
    assert "Dry run: 5 entries (1 not in this store)" in prompts.output


def test_resolutions_apply_sets_fields_provenance_and_review(m, store, prompts) -> None:
    _resolutions(m, store)
    prompts.queue(False)
    m.apply_resolutions(store)
    g = {t.path: t for t in store.all()}
    assert (g["/fix"].artist, g["/fix"].song) == ("Abba", "Waterloo")
    assert g["/fix"].metadata["artist_from"] == "resolutions"
    assert (g["/part"].artist, g["/part"].song) == ("Abba", "Waterloo")
    assert "artist_from" not in g["/same"].metadata            # skipped: already set
    assert g["/same-unreviewed"].metadata["artist_from"] == "resolutions"   # marked ok
    rs = _review(m)
    assert rs.get("/fix") == rs.get("/part") == rs.get("/same-unreviewed") == "ok"
    assert m._CACHE_PATH.exists()
    assert "Applied 3; skipped 1 not-in-store, 1 already-set" in prompts.output


# --- unify_artists ----------------------------------------------------------

def test_unify_with_no_aliases(m, store, prompts) -> None:
    m.unify_artists(store)
    assert "No usable aliases" in prompts.output


def test_unify_dry_run_cancelled(m, store, prompts) -> None:
    _write(m._ARTIST_ALIASES_PATH, {"aliases": {"beatles": "The Beatles"}})
    store.add(make_track("/a", "beatles", "Help"))
    prompts.queue(None)
    m.unify_artists(store)
    assert store.get("/a").artist == "beatles"


def test_unify_dry_run_changes_nothing(m, store, prompts) -> None:
    _write(m._ARTIST_ALIASES_PATH, {"aliases": {"beatles": "The Beatles",
                                                "Unused": "Nobody"}})
    store.add(make_track("/a", "beatles", "Help"))
    store.add(make_track("/b", "beatles", "Yesterday"))
    prompts.queue(True)
    m.unify_artists(store)
    assert {t.artist for t in store.all()} == {"beatles"}
    assert not m._CACHE_PATH.exists()
    assert "[dry] 'beatles' -> 'The Beatles'  (2 track(s))" in prompts.output
    assert "Dry run: 1/2 aliases match tracks, 2 track(s) would be renamed" in prompts.output


def test_unify_apply_renames_and_saves(m, store, prompts) -> None:
    _write(m._ARTIST_ALIASES_PATH, {"aliases": {"beatles": "The Beatles"}})
    store.add(make_track("/a", "beatles", "Help"))
    store.add(make_track("/b", "Abba", "SOS"))
    prompts.queue(False)
    m.unify_artists(store)
    assert store.get("/a").artist == "The Beatles"
    assert store.get("/b").artist == "Abba"
    assert m._CACHE_PATH.exists()
    assert "Renamed 1 track(s) across 1 variant(s)" in prompts.output


def test_unify_truncates_a_long_rename_listing(m, store, prompts) -> None:
    aliases = {f"variant {i}": f"Canonical {i}" for i in range(85)}
    _write(m._ARTIST_ALIASES_PATH, {"aliases": aliases})
    for i in range(85):
        store.add(make_track(f"/{i}", f"variant {i}", "Song"))
    prompts.queue(True)
    m.unify_artists(store)
    assert "... and 5 more variant(s)" in prompts.output
    assert sum(1 for line in prompts.printed if line.startswith("[dry] ")) == 80
