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


# ---------------------------------------------------------------------------
# majority_raw / safe_folder: display-name and export-path safety helpers

def test_majority_raw_prefers_most_common_spelling() -> None:
    from track_index import IndexNode, majority_raw
    node = IndexNode()
    for artist in ["Tom Jones", "Tom Jones", "tom jones", "TOM JONES", "Tom Jones"]:
        node.add(["x"], Track(path=f"/p/{artist}-{id(object())}", file_types=["cdg"],
                              artist=artist, song="Delilah"))
    assert majority_raw(node, "artist") == "Tom Jones"
    assert majority_raw(node, "song") == "Delilah"
    assert majority_raw(IndexNode(), "artist") == ""


def test_safe_folder_flattens_path_hostile_characters() -> None:
    from track_index import safe_folder
    assert safe_folder("ac/dc") == "ac-dc"
    assert safe_folder("back\slash") == "back-slash"
    assert safe_folder('a:b*c?d"e<f>g|h') == "a-b-c-d-e-f-g-h"
    assert safe_folder("plain name") == "plain name"
    assert safe_folder("///") == "---"


# ---------------------------------------------------------------------------
# strip_artist_echo: 'Artist - Artist-Title' display chains (issue #2)

def test_echo_strips_leading_artist() -> None:
    from track_index import strip_artist_echo
    assert strip_artist_echo("Madonna", "Madonna-Vogue") == "Vogue"
    assert strip_artist_echo("Dire Straits", "Dire Straits-Sultans Of Swing") \
        == "Sultans Of Swing"
    # comma-form echoes match order-insensitively
    assert strip_artist_echo("Chris Isaak", "Isaak, Chris-Wicked Game") == "Wicked Game"
    assert strip_artist_echo("Bob Marley", "Marley, Bob-Jammin'") == "Jammin'"
    # artist containing a dash: the split point walks past it
    assert strip_artist_echo("A-Ha", "A-Ha-Take On Me") == "Take On Me"


def test_echo_strips_trailing_artist() -> None:
    from track_index import strip_artist_echo
    assert strip_artist_echo("Tom Jones", "Delilah - Tom Jones") == "Delilah"
    assert strip_artist_echo("Tom Jones", "Delilah - Jones, Tom") == "Delilah"


def test_echo_never_bites_titles_that_mention_the_artist() -> None:
    from track_index import strip_artist_echo
    assert strip_artist_echo("Alabama", "My Home's In Alabama") is None
    assert strip_artist_echo("Big & Rich", "Rollin' (The Ballad Of Big & Rich)") is None
    assert strip_artist_echo("Raven", "That's So Raven") is None
    assert strip_artist_echo("McClymonts", "The McClymonts") is None  # self-titled
    assert strip_artist_echo("Blondie", "Heart Of Glass") is None
    # a dash segment that merely CONTAINS the artist's words plus more
    assert strip_artist_echo("Kiss", "Kiss Me Quick - Live") is None
    assert strip_artist_echo("", "Anything - At All") is None


# ---------------------------------------------------------------------------
# Restitch helpers: dash-elided disc filenames (FLY/SFKK style)

def test_split_stem_drops_catalog_id() -> None:
    from track_index import split_stem
    assert split_stem("FLY-03-06 - Belinda - Carlisle - Heaven Is A Pla - On Earth") \
        == ["Belinda", "Carlisle", "Heaven Is A Pla", "On Earth"]
    assert split_stem("SFKK-21-00 - AVRIL - Lavigne - Hot -") \
        == ["AVRIL", "Lavigne", "Hot"]
    assert split_stem("Plain Artist - Plain Song") == ["Plain Artist", "Plain Song"]


def test_rejoin_artist_reassembles_split_names() -> None:
    from track_index import rejoin_artist
    known = {"belinda carlisle", "foo fighters"}
    assert rejoin_artist(["Belinda", "Carlisle", "Heaven Is A Pla", "On Earth"], known) \
        == ("Belinda Carlisle", ["Heaven Is A Pla", "On Earth"])
    assert rejoin_artist(["FOO", "Fighters", "L", "Road To Ruin"], known) \
        == ("FOO Fighters", ["L", "Road To Ruin"])
    # a 1-segment match is the normal parse, not a rejoin
    assert rejoin_artist(["Madonna", "Vogue", "Extended"], {"madonna"}) is None
    assert rejoin_artist(["Nobody", "Known", "Here"], known) is None


def test_fragments_match_title_elisions() -> None:
    from track_index import fragments_match_title
    assert fragments_match_title(["Heaven Is A Pla", "On Earth"],
                                 "Heaven Is a Place on Earth")
    assert fragments_match_title(["Bel", "E Again"], "Believe Again")
    assert fragments_match_title(["Tur", "E Loose"], "Turn Me Loose")
    assert fragments_match_title(["Hate", "T I Love You"], "Hate That I Love You")
    # end-truncation: title may continue past the last fragment
    assert fragments_match_title(["Hea", "Nes (Friendship Never E"],
                                 "Headlines (Friendship Never Ends)")
    # must anchor at the start and stay in order
    assert not fragments_match_title(["On Earth", "Heaven"], "Heaven Is a Place on Earth")
    assert not fragments_match_title(["Place"], "Heaven Is a Place on Earth")
    assert not fragments_match_title(["Bel", "E Again"], "Born Again")
    assert not fragments_match_title([], "Anything")


# ---------------------------------------------------------------------------
# parse_artist_song / is_catalog_segment: catalog-LAST stems (Sunfly zips)

def test_is_catalog_segment() -> None:
    from track_index import is_catalog_segment
    for yes in ("SF 193-16", "SFKK-21-00", "EZH-31", "SC8121-03", "DCK927-10"):
        assert is_catalog_segment(yes), yes
    # titles that merely look catalog-ish must never qualify
    for no in ("Old 67", "U2", "Happy 70th Birthday", "Mambo No 5",
               "Plain Title", "72", ""):
        assert not is_catalog_segment(no), no


def test_parse_catalog_last_stems() -> None:
    from track_index import parse_artist_song
    # the Lauren Waterworth case: catalog at the END must not eat the artist
    assert parse_artist_song(
        "Lauren Waterworth - Baby Now That I've Found You - SF 193-16") \
        == ("Lauren Waterworth", "Baby Now That I've Found You")
    # dashy titles keep their inner segments
    assert parse_artist_song("A - B - C - SF 193-16") == ("A", "B - C")
    # title-only with trailing catalog
    assert parse_artist_song("Some Song Title - SF 193-16") \
        == ("", "Some Song Title")


def test_parse_catalog_first_unchanged() -> None:
    from track_index import parse_artist_song
    assert parse_artist_song("SC8121-03 - Beach Boys - Barbara Ann") \
        == ("Beach Boys", "Barbara Ann")
    assert parse_artist_song("Beach Boys - Barbara Ann") == ("", "Barbara Ann")
    assert parse_artist_song("Just A Song Title") == ("", "Just A Song Title")
    # catalog first AND a catalog-shaped title: first wins, title survives
    assert parse_artist_song("DCK927-10 - Elton John - Old 67") \
        == ("Elton John", "Old 67")
    # compact Disney-style stems still work
    assert parse_artist_song("DIS61201-13-MARY POPPINS-I LOVE TO LAUGH") \
        == ("MARY POPPINS", "I LOVE TO LAUGH")


def test_split_stem_drops_trailing_catalog() -> None:
    from track_index import split_stem
    assert split_stem("Lauren Waterworth - Baby Now - SF 193-16") \
        == ["Lauren Waterworth", "Baby Now"]
    assert split_stem("SC81 - Artist - Song") == ["Artist", "Song"]
