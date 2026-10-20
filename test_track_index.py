"""Tests for track_index's index structures.

ArtistIndex / SongIndex / TrackIndex are what Final-final, Stats, List, Browse and
every cleanup pass group tracks by, so their bucketing rules are load-bearing.
The string helpers in this module are covered in test_track.py.
"""

from __future__ import annotations

from conftest import make_track
from track import TrackStore
from track_index import (
    ArtistIndex,
    IndexNode,
    SongIndex,
    TrackIndex,
    fragments_match_title,
    iter_tracks,
    parse_artist_song,
    uncomma_artist,
)


def _store(*tracks) -> TrackStore:
    s = TrackStore()
    for t in tracks:
        s.add(t)
    return s


# --- IndexNode -------------------------------------------------------------

def test_index_node_builds_levels_and_counts() -> None:
    root = IndexNode()
    a = make_track("/1", "A", "S1")
    b = make_track("/2", "A", "S2")
    c = make_track("/3", "B", "S1")
    root.add(["x", "p", "t"], a)
    root.add(["x", "q", "t"], b)
    root.add(["y", "r", "t"], c)
    assert list(root.list_nodes()) == ["x", "y"]        # sorted keys
    assert not root.is_leaf()
    assert root.count() == 3
    assert root.count() == 3                             # cached path
    assert root.nodes["x"].count() == 2


def test_list_tracks_is_sorted_by_display_chain() -> None:
    node = IndexNode()
    for artist, song in [("B", "Z"), ("A", "Y"), ("A", "X")]:
        node.add(["leaf"], make_track(f"/{artist}{song}", artist, song))
    assert [f"{t.artist} - {t.song}" for t in node.list_tracks()] == \
        ["A - X", "A - Y", "B - Z"]


def test_iter_tracks_walks_every_level() -> None:
    root = IndexNode()
    tracks = [make_track(f"/{i}", "A", f"S{i}") for i in range(4)]
    root.add(["a", "x", "t"], tracks[0])
    root.add(["a", "y", "t"], tracks[1])
    root.add(["b", "z", "t"], tracks[2])
    root.add(["b", "z", "u"], tracks[3])
    assert {t.path for t in iter_tracks(root)} == {t.path for t in tracks}


# --- TrackIndex ------------------------------------------------------------

def test_track_index_groups_by_path_components() -> None:
    idx = TrackIndex(_store(make_track("/lib/SC/a.zip"), make_track("/lib/SC/b.zip"),
                            make_track("/lib/SF/c.zip")))
    root = idx.get_root()
    assert root.count() == 3
    lib = next(iter(root.list_nodes().values())).nodes["lib"]
    assert set(lib.list_nodes()) == {"SC", "SF"}
    assert lib.nodes["SC"].count() == 2


# --- ArtistIndex -----------------------------------------------------------

def test_artist_index_buckets_by_first_letter_and_clean_names() -> None:
    idx = ArtistIndex.from_store(_store(
        make_track("/1", "The Beatles", "Help"),
        make_track("/2", "Beatles", "Help!"),        # same clean artist
        make_track("/3", "10cc", "Rubber Bullets"),
        make_track("/4", "", "Mystery"),
    ))
    root = idx.get_root()
    assert set(root.list_nodes()) == {"b", "#", ""}
    assert "beatles" in root.nodes["b"].nodes
    assert root.nodes["b"].nodes["beatles"].count() == 2
    assert "10cc" in root.nodes["#"].nodes


def test_count_artists_uses_low_bound_and_skips_blank_bucket() -> None:
    tracks = [make_track(f"/a{i}", "Abba", f"S{i}") for i in range(3)]
    tracks += [make_track("/b", "Bread", "If")]
    tracks += [make_track(f"/n{i}", "", f"X{i}") for i in range(5)]
    idx = ArtistIndex(tracks)
    assert idx.count_artists() == ["abba"]              # > 1 track
    assert idx.count_artists(low_bound=0) == ["abba", "bread"]
    assert idx.count_artists(low_bound=5) == []


def test_single_artists_returns_one_track_per_song_of_thin_artists() -> None:
    tracks = [make_track(f"/a{i}", "Abba", f"S{i}") for i in range(5)]   # busy
    tracks += [make_track("/b1", "Bread", "If"), make_track("/b2", "Bread", "If")]
    idx = ArtistIndex(tracks)
    thin = idx.single_artists(max_songs=3)
    assert [(t.artist, t.song) for t in thin] == [("Bread", "If")]   # one per song


# --- SongIndex -------------------------------------------------------------

def test_song_index_skips_unknown_artists_and_buckets_titles() -> None:
    idx = SongIndex.from_store(_store(
        make_track("/1", "Abba", "Waterloo"),
        make_track("/2", "Unknown", "Waterloo"),
        make_track("/3", "", "Waterloo"),
        make_track("/4", "Prince", "1999"),
    ))
    root = idx.get_root()
    assert set(root.list_nodes()) == {"w", "#"}
    assert root.nodes["w"].count() == 1                  # only Abba's copy


# --- remaining string-helper branches ---------------------------------------

def test_parse_compact_stem_without_an_artist() -> None:
    assert parse_artist_song("DIS61201-13-JUSTATITLE") == ("", "JUSTATITLE")


def test_parse_blank_stem_is_unknown() -> None:
    assert parse_artist_song("   ") == ("", "Unknown")


def test_fragments_fail_when_a_later_fragment_is_missing() -> None:
    assert not fragments_match_title(["Heaven Is", "Nowhere"], "Heaven Is A Place On Earth")


def test_fragments_reject_empty_inputs() -> None:
    assert not fragments_match_title([], "Anything")
    assert not fragments_match_title(["x"], "")


def test_uncomma_refuses_implausibly_long_names() -> None:
    assert uncomma_artist("Smith, A B C D") is None
    assert uncomma_artist("A B C D, John") is None
