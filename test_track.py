"""Tests for Track and TrackStore."""

import tempfile
from pathlib import Path

import pytest

from track import Track, TrackStore


def test_track_to_dict_roundtrip() -> None:
    t = Track(
        path="/foo/Artist - Song.mp3",
        file_types=["mp3", "cdg"],
        artist="Artist",
        song="Song",
        metadata={"length_seconds": "180"},
    )
    d = t.to_dict()
    assert d["path"] == "/foo/Artist - Song.mp3"
    assert d["file_types"] == ["mp3", "cdg"]
    assert d["artist"] == "Artist"
    assert d["song"] == "Song"
    assert d["metadata"] == {"length_seconds": "180"}

    t2 = Track.from_dict(d)
    assert t2.path == t.path
    assert t2.file_types == t.file_types
    assert t2.artist == t.artist
    assert t2.song == t.song
    assert t2.metadata == t.metadata


def test_track_from_dict_defaults() -> None:
    d = {"path": "/x.zip"}
    t = Track.from_dict(d)
    assert t.path == "/x.zip"
    assert t.file_types == []
    assert t.artist == ""
    assert t.song == ""
    assert t.metadata == {}


def test_store_add_and_all() -> None:
    store = TrackStore()
    assert store.all() == []

    t1 = Track(path="/a.mp3", file_types=["mp3"], artist="A", song="S1")
    store.add(t1)
    assert len(store.all()) == 1
    assert store.all()[0].path == "/a.mp3"

    t2 = Track(path="/b.zip", file_types=["zip"], artist="B", song="S2")
    store.add(t2)
    assert len(store.all()) == 2


def test_store_add_replaces_same_path() -> None:
    store = TrackStore()
    t1 = Track(path="/same.mp3", file_types=["mp3"], artist="A", song="S1")
    t2 = Track(path="/same.mp3", file_types=["mp3"], artist="B", song="S2")
    store.add(t1)
    store.add(t2)
    assert len(store.all()) == 1
    assert store.all()[0].artist == "B"


def test_store_save_and_load(tmp_path: Path) -> None:
    store = TrackStore()
    store.add(
        Track(
            path="/foo/Artist - Song.mp3",
            file_types=["mp3", "cdg"],
            artist="Artist",
            song="Song",
            metadata={"publisher": "KP"},
        )
    )
    cache = tmp_path / "cache.json"
    store.save(cache)

    assert cache.exists()
    loaded = cache.read_text(encoding="utf-8")
    assert "Artist" in loaded
    assert "version" in loaded

    store2 = TrackStore()
    store2.load(cache)
    tracks = store2.all()
    assert len(tracks) == 1
    assert tracks[0].path == "/foo/Artist - Song.mp3"
    assert tracks[0].file_types == ["mp3", "cdg"]
    assert tracks[0].artist == "Artist"
    assert tracks[0].song == "Song"
    assert tracks[0].metadata == {"publisher": "KP"}


def test_store_load_nonexistent() -> None:
    store = TrackStore()
    store.add(Track(path="/x.mp3", file_types=["mp3"], artist="", song=""))
    store.load("/nonexistent/path/cache.json")
    assert store.all() == []


def test_store_load_invalid_version(tmp_path: Path) -> None:
    cache = tmp_path / "cache.json"
    cache.write_text('{"version": 99, "tracks": []}', encoding="utf-8")

    store = TrackStore()
    with pytest.raises(ValueError, match="Unsupported cache version 99"):
        store.load(cache)


# ---------------------------------------------------------------------------
# uncomma_artist: safe 'Last, First' -> 'First Last' swaps

def test_uncomma_swaps_person_names() -> None:
    from track_index import uncomma_artist
    assert uncomma_artist("Jones, Tom") == "Tom Jones"
    assert uncomma_artist("Newton-John, Olivia") == "Olivia Newton-John"
    assert uncomma_artist("Van Dyke, Dick") == "Dick Van Dyke"
    assert uncomma_artist("Eckstine, Billy") == "Billy Eckstine"


def test_uncomma_relocates_generational_suffixes() -> None:
    from track_index import uncomma_artist
    assert uncomma_artist("Davis, Sammy Jr.") == "Sammy Davis Jr."
    assert uncomma_artist("Connick, Harry Jr.") == "Harry Connick Jr."


def test_uncomma_refuses_bands_and_multi_credits() -> None:
    from track_index import uncomma_artist
    # band names with commas must never blind-swap (the 'wind & fire earth'
    # incident): & / and / ft / feat / with mark a non-person credit
    assert uncomma_artist("Earth, Wind & Fire") is None
    assert uncomma_artist("Blood, Sweat & Tears") is None
    assert uncomma_artist("Crosby, Stills & Nash") is None
    assert uncomma_artist("Butler, Carl & Pearl") is None
    assert uncomma_artist("Andrews, Michael Ft. Gary Jules") is None
    assert uncomma_artist("Timberlake, Justin Feat. T.I") is None
    assert uncomma_artist("Brooks, Meredith with Queen Latifah") is None


def test_uncomma_refuses_malformed_forms() -> None:
    from track_index import uncomma_artist
    assert uncomma_artist("Tom Jones") is None          # no comma
    assert uncomma_artist("Peter, Paul, Mary") is None  # two commas
    assert uncomma_artist("Jones,") is None             # empty side
    assert uncomma_artist(", Tom") is None
    assert uncomma_artist("Puppets, The") is None       # trailing-article's job
