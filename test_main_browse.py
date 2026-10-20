"""main.py: the interactive browse / edit / fixup flows, and Fix-unknown.

Menus are driven through scripted questionary answers (see conftest). `pick`
selects by visible title, so tests navigate the way a person does.
"""

from __future__ import annotations

import pytest

from conftest import make_track, pick


@pytest.fixture
def m(mainmod):
    return mainmod


def _add(store, *tracks):
    for t in tracks:
        store.add(t)
    return store


def _busy(store, artist, n=6, prefix="/k"):
    """An artist with enough tracks to count as 'known' (count_artists > 5)."""
    for i in range(n):
        store.add(make_track(f"{prefix}/{artist}{i}", artist, f"Known {i}"))


# --- browse (by path) ------------------------------------------------------

def test_browse_descends_shows_tracks_and_climbs_back(m, store, prompts) -> None:
    _add(store, make_track("/lib/SC/a.zip", "Abba", "Waterloo"),
         make_track("/lib/SC/b.zip", "Abba", "SOS"),
         make_track("/lib/SF/c.zip", "Queen", "Radio Gaga"))
    prompts.queue(pick("/  #[3]"), pick("lib  #[3]"), pick("SC  #[2]"),
                  pick("Abba - SOS"),        # a track row: display only
                  "up", "up", "exit")
    m.browse(store)
    assert prompts.asked[0] == ("select", "Browse:")


def test_browse_cancel_exits(m, store, prompts) -> None:
    _add(store, make_track("/lib/a.zip"))
    prompts.queue(None)
    m.browse(store)


# --- browse_artist / browse_song -------------------------------------------

def test_browse_artist_reaches_track_details(m, store, prompts) -> None:
    _add(store, make_track("/lib/Abba - Waterloo.zip", "Abba", "Waterloo",
                           bitrate_bps=192000, length_seconds=165))
    prompts.queue(pick("a  #[1]"), pick("abba  #[1]"), pick("waterloo  #[1]"),
                  pick("bitrate 192"),       # leaf row carries metadata summary
                  "up", "exit")
    m.browse_artist(store)


def test_browse_artist_cancel(m, store, prompts) -> None:
    _add(store, make_track())
    prompts.queue(None)
    m.browse_artist(store)


def test_browse_song_groups_by_title(m, store, prompts) -> None:
    _add(store, make_track("/lib/1.zip", "Abba", "Waterloo"),
         make_track("/lib/2.zip", "Unknown", "Hidden"))       # excluded
    prompts.queue(pick("w  #[1]"), pick("waterloo  #[1]"), pick("Abba  #[1]"),
                  pick("sec"), "up", "exit")
    m.browse_song(store)


def test_browse_song_cancel(m, store, prompts) -> None:
    _add(store, make_track())
    prompts.queue(None)
    m.browse_song(store)


# --- _tracks ---------------------------------------------------------------

def test_tracks_yields_every_track_under_a_node(m, store) -> None:
    from track_index import ArtistIndex
    _add(store, make_track("/1", "Abba", "A"), make_track("/2", "Abba", "B"),
         make_track("/3", "Blur", "C"))
    root = ArtistIndex.from_store(store).get_root()
    assert {t.path for t in m._tracks(root)} == {"/1", "/2", "/3"}


# --- _bulk_edit_artist -----------------------------------------------------

def _artist_node(store, clean):
    from track_index import ArtistIndex
    root = ArtistIndex.from_store(store).get_root()
    return root.nodes[clean[0]].nodes[clean]


def test_bulk_edit_renames_every_track(m, store, prompts, capsys) -> None:
    _add(store, make_track("/1", "abba", "A"), make_track("/2", "abba", "B"))
    prompts.queue("edit", "ABBA")
    m._bulk_edit_artist(store, _artist_node(store, "abba"))
    assert {t.artist for t in store.all()} == {"ABBA"}
    assert "Changing abba to ABBA" in capsys.readouterr().out


def test_bulk_edit_blank_text_changes_nothing(m, store, prompts) -> None:
    _add(store, make_track("/1", "abba", "A"))
    prompts.queue("edit", "")
    m._bulk_edit_artist(store, _artist_node(store, "abba"))
    assert store.all()[0].artist == "abba"


def test_bulk_edit_uncomma_splits_feature(m, store, prompts) -> None:
    _add(store, make_track("/1", "Abba & Friends", "A"))
    prompts.queue("uncomma")
    m._bulk_edit_artist(store, _artist_node(store, "abba & friends"))
    t = store.all()[0]
    assert t.artist == "Abba"
    assert t.metadata["feature"] == "Friends"


def test_bulk_edit_uncomma_without_feature(m, store, prompts) -> None:
    _add(store, make_track("/1", "Abba", "A"))
    prompts.queue("uncomma")
    m._bulk_edit_artist(store, _artist_node(store, "abba"))
    assert "feature" not in store.all()[0].metadata


@pytest.mark.parametrize("answer", ["exit", None])
def test_bulk_edit_exit_or_cancel(m, store, prompts, answer) -> None:
    _add(store, make_track("/1", "Abba", "A"))
    prompts.queue(answer)
    m._bulk_edit_artist(store, _artist_node(store, "abba"))
    assert store.all()[0].artist == "Abba"


# --- _edit_track_details ---------------------------------------------------

def test_edit_track_swap(m, store, prompts) -> None:
    t = make_track("/1", "Waterloo", "Abba")
    store.add(t)
    prompts.queue("swap")
    m._edit_track_details(store, t)
    assert (t.artist, t.song) == ("Abba", "Waterloo")


def test_edit_track_unset_artist(m, store, prompts) -> None:
    t = make_track("/1", "Abba", "Waterloo")
    prompts.queue("unset-artist")
    m._edit_track_details(store, t)
    assert t.artist == ""


def test_edit_track_artist_and_song(m, store, prompts) -> None:
    t = make_track("/1", "Abba", "Waterloo")
    prompts.queue("artist", "ABBA")
    m._edit_track_details(store, t)
    prompts.queue("song", "Waterloo (Live)")
    m._edit_track_details(store, t)
    assert (t.artist, t.song) == ("ABBA", "Waterloo (Live)")


def test_edit_track_blank_answers_keep_values(m, store, prompts) -> None:
    t = make_track("/1", "Abba", "Waterloo")
    prompts.queue("artist", "", "song", None)
    m._edit_track_details(store, t)
    m._edit_track_details(store, t)
    assert (t.artist, t.song) == ("Abba", "Waterloo")


@pytest.mark.parametrize("answer", ["exit", None])
def test_edit_track_exit_or_cancel(m, store, prompts, answer) -> None:
    t = make_track("/1", "Abba", "Waterloo")
    prompts.queue(answer)
    m._edit_track_details(store, t)
    assert (t.artist, t.song) == ("Abba", "Waterloo")


# --- _tracks_to_review -----------------------------------------------------

def test_tracks_to_review_thin_artists_minus_reviewed(m, store) -> None:
    from review_state import ReviewState
    _busy(store, "Abba")                                  # 6 tracks: not thin
    _add(store, make_track("/t1", "Tiny", "One"), make_track("/t2", "Tiny", "Two"))
    rs = ReviewState()
    rs.set("/t1", "ok")
    got = m._tracks_to_review(store, rs)
    assert [t.path for t in got] == ["/t2"]


def test_always_review_pulls_in_a_busy_bucket(m, store) -> None:
    from review_state import ReviewState
    _busy(store, "Garbage Bucket")
    m._save_config({"always_review": ["garbage bucket"]})
    got = m._tracks_to_review(store, ReviewState())
    assert len(got) == 6


# --- browse_fixup ----------------------------------------------------------

def test_browse_fixup_edits_a_reviewable_track(m, store, prompts) -> None:
    t = make_track("/t1", "Waterloo", "Abba")
    _add(store, t)
    prompts.queue(pick("w  #[1]"), pick("waterloo  #[1]"), pick("abba  #[1]"),
                  pick("Waterloo - t1"), "swap",
                  "up", "exit")
    m.browse_fixup(store)
    assert (t.artist, t.song) == ("Abba", "Waterloo")


def test_browse_fixup_cancel(m, store, prompts) -> None:
    prompts.queue(None)
    m.browse_fixup(store)


# --- fix_artist ------------------------------------------------------------

def test_fix_artist_bulk_renames_from_the_artist_level(m, store, prompts) -> None:
    _add(store, make_track("/1", "abba", "A"), make_track("/2", "abba", "B"))
    prompts.queue(pick("a  #[2]"),                   # into the 'a' bucket
                  pick("abba  #[2]"), "edit", "ABBA",  # artist level -> bulk edit
                  "up", "exit")
    m.fix_artist(store)
    assert {t.artist for t in store.all()} == {"ABBA"}


def test_fix_artist_cancel(m, store, prompts) -> None:
    prompts.queue(None)
    m.fix_artist(store)


# --- fix_unknown -----------------------------------------------------------

def test_fix_unknown_recovers_every_split_shape(m, store, capsys, prompts) -> None:
    _busy(store, "Abba")
    _busy(store, "Queen", prefix="/q")
    cases = {
        # spaced dash
        "Abba - Waterloo": ("Abba", "Waterloo"),
        "SC1 - Abba - SOS": ("Abba", "SOS"),
        "SC - 1 - Queen - Bohemian": ("Queen", "Bohemian"),
        "Abba_-_Mamma": ("Abba", "Mamma"),              # underscores -> spaces
        # bare dash
        "Abba-Fernando": ("Abba", "Fernando"),
        "ID-Queen-Radio Gaga": ("Queen", "Radio Gaga"),
        # double space, both orientations
        "Abba  Chiquitita": ("Abba", "Chiquitita"),
        "Innuendo  Queen": ("Queen", "Innuendo"),
        # comma
        "Abba, Dancing Queen": ("Abba", "Dancing Queen"),
    }
    unresolved = ["Nobody - Song", "SC1 - Nobody - X", "SC - 1 - Nobody - X",
                  "a - b - c - d - e", "Nobody-Song", "ID-Nobody-X", "a-b-c-d",
                  "Foo  Bar", "a  b  c", "Nobody, X", "a, b, c",
                  "Abba Knowing Me", "Knowing Me Abba", "Nothing Here"]
    for i, song in enumerate(list(cases) + unresolved):
        store.add(make_track(f"/u{i}", "Unknown" if i % 2 else "", song))
    m.fix_unknown(store)

    got = {t.path: (t.artist, t.song) for t in store.all() if t.path.startswith("/u")}
    for i, (song, want) in enumerate(cases.items()):
        assert got[f"/u{i}"] == want, song
    for j, song in enumerate(unresolved, start=len(cases)):
        assert got[f"/u{j}"][1] == song, song          # untouched

    out = capsys.readouterr().out
    assert "Complex [5]" in out and "Complex space" in out
    # space-only stems are REPORTED as candidates, never written
    assert "candidate: 'Abba' - 'Knowing Me'" in out
    assert "candidate: 'Abba' - 'Knowing Me'  [Knowing Me Abba]" in out
    assert "Success 9" in prompts.output
