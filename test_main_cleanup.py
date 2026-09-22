"""main.py: the deterministic cleanup chain.

Clean, Tag-fill, Trailing-article, Uncomma, Fuzz, Fuzz_song, Unswap, Ungroup.
Each rewrites library data, so each rule is pinned, including the guards that
exist because of past incidents (the 'wind & fire earth' relics).
"""

from __future__ import annotations

import pytest

from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


def _many(store, artist, n, prefix):
    for i in range(n):
        store.add(make_track(f"{prefix}/{i}", artist, f"Song {i}"))


# --- clean: descriptor lists, brand tags, paren noise -----------------------

def test_clean_strips_descriptors_and_records_style(m, store, prompts) -> None:
    store.add(make_track("/1", "Tom Jones (Wobgv)", "Delilah (Duet) (Explicit)"))
    store.add(make_track("/2", "Adele", "Hello [SC Karaoke]"))
    store.add(make_track("/3", "Queen [Zoom Karaoke]", "Innuendo"))
    store.add(make_track("/4", "Blur (WOBGV)", "Song 2 (Mplx)"))      # case variants
    m.clean(store)
    got = {t.path: t for t in store.all()}
    assert got["/1"].artist == "Tom Jones"
    assert got["/1"].song == "Delilah"                  # every song indicator goes
    assert got["/2"].song == "Hello"
    assert got["/2"].metadata["style"] == "[SC Karaoke]"
    assert got["/3"].artist == "Queen"
    assert (got["/4"].artist, got["/4"].song) == ("Blur", "Song 2")
    assert "Stripped 2 [.. Karaoke ..] brand tags" in prompts.output


def test_clean_routes_bogus_artists_into_the_unknown_pipeline(m, store, prompts) -> None:
    store.add(make_track("/l/EZH-31 - 04 - Milkshake.zip", "04", "Milkshake"))
    store.add(make_track("/l/07 - Loner.zip", "07", "Loner"))           # no catalog part
    store.add(make_track("/l/x.zip", "SC\\SC-199\\SC-199-02", "Evita"))
    store.add(make_track("/l/y.zip", "", "Blank"))
    store.add(make_track("/l/z.zip", "911", "Bodyshakin"))              # a real band
    store.add(make_track("/l/u.zip", "Unknown", "Already"))
    m.clean(store)
    got = {t.path: t for t in store.all()}
    ms = got["/l/EZH-31 - 04 - Milkshake.zip"]
    assert ms.artist == "Unknown"
    assert (ms.metadata["track_no"], ms.metadata["catalog_id"]) == ("04", "EZH-31")
    assert "catalog_id" not in got["/l/07 - Loner.zip"].metadata
    assert got["/l/x.zip"].metadata["catalog_id"] == "SC\\SC-199\\SC-199-02"
    assert got["/l/y.zip"].artist == "Unknown"
    assert got["/l/z.zip"].artist == "911"
    assert "Cleared 4 track-number/catalog-path/empty artists" in prompts.output


def test_clean_strips_artist_echoes(m, store, prompts) -> None:
    store.add(make_track("/1", "Chris Isaak", "Isaak, Chris-Wicked Game"))
    store.add(make_track("/2", "Unknown", "Isaak, Chris-Wicked Game"))  # not attempted
    store.add(make_track("/3", "Alabama", "My Home's In Alabama"))      # never bitten
    m.clean(store)
    got = {t.path: t.song for t in store.all()}
    assert got == {"/1": "Wicked Game", "/2": "Isaak, Chris-Wicked Game",
                   "/3": "My Home's In Alabama"}


def test_clean_reparses_catalog_ids_parked_in_the_song(m, store, prompts) -> None:
    cases = {
        # artist recovered from a catalog-LAST stem
        ("/l/Abba - Waterloo - SF 193-16.zip", "Waterloo", "SF 193-16"): ("Abba", "Waterloo"),
        # artist slot held the title -> Unknown, never artist == song
        ("/l/Waterloo - SF 193-16.zip", "Waterloo", "SF 193-16"): ("Unknown", "Waterloo"),
        # a different, real artist is kept
        ("/l/2/Waterloo - SF 193-16.zip", "Somebody", "SF 193-16"): ("Somebody", "Waterloo"),
        # catalog-looking title that is not a trailing stem segment: untouched
        ("/l/Levert - Something Else.zip", "Levert", "ABC 123"): ("Levert", "ABC 123"),
        # re-parse yields another catalog id: untouched
        ("/l/SF 193-16 - SF 193-17.zip", "X", "SF 193-17"): ("X", "SF 193-17"),
    }
    for (path, a, s) in cases:
        store.add(make_track(path, a, s))
    m.clean(store)
    got = {t.path: (t.artist, t.song) for t in store.all()}
    for (path, _a, _s), want in cases.items():
        assert got[path] == want, path
    assert store.get("/l/Abba - Waterloo - SF 193-16.zip").metadata["catalog_id"] == "SF 193-16"


# --- fill_artist_from_tags (Tag-fill) ---------------------------------------

def test_tag_fill_fills_flags_and_never_overwrites(m, store, prompts) -> None:
    store.add(make_track("/fill", "Unknown", "A", tag_artist="Zac Brown Band (Wbgv)"))
    store.add(make_track("/num", "", "B", tag_artist="311"))
    store.add(make_track("/cat", "Unknown", "C", tag_artist="PS1254"))
    store.add(make_track("/nf", "Unknown", "D", tag_artist="<not-found>"))
    store.add(make_track("/none", "Unknown", "E"))
    store.add(make_track("/junk", "Unknown", "F", tag_artist="(duet)"))  # suffix only
    store.add(make_track("/keep", "Real Artist", "G", tag_artist="Other"))
    m.fill_artist_from_tags(store)
    g = {t.path: t for t in store.all()}
    assert (g["/fill"].artist, g["/fill"].metadata["artist_from"]) == ("Zac Brown Band", "tag")
    assert (g["/num"].artist, g["/num"].metadata["artist_review_reason"]) == ("", "numeric")
    assert g["/cat"].metadata["artist_review"] == "PS1254"
    assert g["/cat"].metadata["artist_review_reason"] == "catalog-id"
    for p in ("/nf", "/none", "/junk"):
        assert g[p].artist == "Unknown", p
    assert g["/keep"].artist == "Real Artist"
    assert "Filled 1 artists from tags; flagged 2" in prompts.output


# --- trailing article -------------------------------------------------------

@pytest.mark.parametrize("raw, want", [
    ("Beatles, The", "The Beatles"),
    ("Whole New World, A", "A Whole New World"),
    ("Apple, AN", "An Apple"),
    ("Earth, Wind & Fire", "Earth, Wind & Fire"),          # not an article
    ("Ace Of Base-Sign, The", "Ace Of Base-Sign, The"),    # dashed blob: skipped
    ("No Article", "No Article"),
])
def test_fix_trailing_article(m, raw, want) -> None:
    assert m._fix_trailing_article(raw) == want


def test_trailing_article_pass(m, store, prompts) -> None:
    store.add(make_track("/1", "Models, The", "Whole New World, A"))
    store.add(make_track("/2", "Abba", "Waterloo"))
    m.trailing_article(store)
    assert (store.get("/1").artist, store.get("/1").song) == ("The Models", "A Whole New World")
    assert "Fixed 1 trailing articles" in prompts.output


# --- standard_artist (Uncomma) ---------------------------------------------

def test_uncomma_needs_corroboration(m, store, prompts) -> None:
    import json
    _many(store, "Tom Jones", 2, "/tj")
    store.add(make_track("/swap", "Jones, Tom", "Delilah"))
    store.add(make_track("/alone", "Nobody, Such", "X"))            # no corroboration
    store.add(make_track("/band", "Earth, Wind & Fire", "Sept"))    # band: never swapped
    store.add(make_track("/alias", "The Fab Four", "Help"))
    m._ARTIST_ALIASES_PATH.parent.mkdir(parents=True, exist_ok=True)
    m._ARTIST_ALIASES_PATH.write_text(
        json.dumps({"aliases": {"The Fab Four": "The Beatles"}}), encoding="utf-8")
    m.standard_artist(store)
    assert store.get("/swap").artist == "Tom Jones"
    assert store.get("/alone").artist == "Nobody, Such"
    assert store.get("/band").artist == "Earth, Wind & Fire"
    assert store.get("/alias").artist == "The Beatles"
    assert "Swapped 2" in prompts.output


# --- fuzz_artist / fuzz_song -----------------------------------------------

def test_fuzz_artist_folds_the_rarer_spelling_into_the_common_one(m, store, prompts) -> None:
    _many(store, "The Beatles", 3, "/b")
    store.add(make_track("/typo", "Beatless", "Help"))
    _many(store, "Big Band Name", 6, "/busy")                       # > 5: never a source
    store.add(make_track("/blank", "", "No Letter"))
    m.fuzz_artist(store)
    assert store.get("/typo").artist == "The Beatles"               # majority RAW spelling
    assert all(t.artist == "Big Band Name" for t in store.all() if t.path.startswith("/busy"))


def test_fuzz_artist_breaks_ties_deterministically(m, store, prompts) -> None:
    store.add(make_track("/x", "Beatles", "A"))
    store.add(make_track("/y", "Beatless", "B"))
    m.fuzz_artist(store)
    # equal counts: the smaller key ('beatles') folds into the larger one
    assert {t.artist for t in store.all()} == {"Beatless"}


def test_fuzz_song_merges_near_duplicate_titles(m, store, prompts) -> None:
    store.add(make_track("/1", "Abba", "Waterloo"))
    store.add(make_track("/2", "Abba", "Waterloo"))
    store.add(make_track("/3", "Abba", "Waterlooo"))
    store.add(make_track("/4", "", "Ignored"))
    m.fuzz_song(store)
    assert {t.song for t in store.all() if t.artist == "Abba"} == {"Waterloo"}


def test_fuzz_song_breaks_ties_deterministically(m, store, prompts) -> None:
    store.add(make_track("/1", "Abba", "Waterloo"))
    store.add(make_track("/2", "Abba", "Waterlooo"))
    m.fuzz_song(store)
    assert {t.song for t in store.all()} == {"Waterlooo"}


# --- find_swapped (Unswap) --------------------------------------------------

def _reversed(store, folder, n, known="Avril Lavigne", copies=1):
    """n reversed tracks in one folder: artist = a title, song = a known artist."""
    for i in range(n):
        for c in range(copies):
            store.add(make_track(f"/{folder}/{i}-{c}.zip", f"Title {i}", known))


def test_unswap_applies_when_enough_tracks_in_a_folder_agree(m, store, prompts) -> None:
    _many(store, "Avril Lavigne", 6, "/k")
    _reversed(store, "disc", 5)
    _reversed(store, "few", 2)                                      # too few: left alone
    m.find_swapped(store)
    disc = [t for t in store.all() if t.path.startswith("/disc/")]
    assert all(t.artist == "Avril Lavigne" for t in disc)
    assert all(t.artist.startswith("Title") for t in store.all() if t.path.startswith("/few/"))
    assert "Completed Swaps 5" in prompts.output


@pytest.mark.xfail(strict=True, reason="GH #29: README promises Unswap applies where "
                   ">=3 tracks in a folder agree, but the code requires > 3 (i.e. >= 4)")
def test_unswap_threshold_matches_the_documented_three(m, store, prompts) -> None:
    _many(store, "Avril Lavigne", 6, "/k")
    _reversed(store, "disc", 3)
    m.find_swapped(store)
    assert all(t.artist == "Avril Lavigne" for t in store.all() if t.path.startswith("/disc/"))


@pytest.mark.xfail(strict=True, reason="GH #30: Unswap works from single_artists(), which "
                   "yields ONE copy per song, so duplicate copies are left reversed")
def test_unswap_fixes_every_copy_of_a_song(m, store, prompts) -> None:
    _many(store, "Avril Lavigne", 6, "/k")
    _reversed(store, "disc", 5, copies=2)
    m.find_swapped(store)
    left = [t.path for t in store.all()
            if t.path.startswith("/disc/") and t.artist != "Avril Lavigne"]
    assert not left, f"copies left reversed: {left}"


# --- ungroup_artist (Ungroup) -----------------------------------------------

def test_ungroup_moves_collaborators_into_feature(m, store, prompts) -> None:
    _many(store, "Abba", 2, "/abba")
    _many(store, "Tom Jones", 2, "/tj")
    store.add(make_track("/g", "Abba & Friends", "Waterloo"))
    store.add(make_track("/c", "Jones, Tom & Friends", "Delilah"))  # uncomma'd primary
    store.add(make_track("/n", "Nobody & Else", "X"))               # primary unknown
    m.ungroup_artist(store)
    assert (store.get("/g").artist, store.get("/g").metadata["feature"]) == ("Abba", "Friends")
    assert store.get("/c").artist == "Tom Jones"
    assert store.get("/n").artist == "Nobody & Else"


def test_ungroup_never_inverts_a_comma_band(m, store, prompts) -> None:
    """The known-artist corroboration is what protects comma bands here (the
    guard _auto_clean_artist lacks, see #25): 'wind earth' is no known artist."""
    _many(store, "Earth, Wind & Fire", 2, "/ewf")
    m.ungroup_artist(store)
    assert all(t.artist == "Earth, Wind & Fire" for t in store.all())


@pytest.mark.xfail(strict=True, reason="GH #30: Ungroup works from single_artists(), which "
                   "yields ONE copy per song, so duplicate copies keep the group credit")
def test_ungroup_fixes_every_copy_of_a_song(m, store, prompts) -> None:
    _many(store, "Abba", 2, "/abba")
    for c in range(3):
        store.add(make_track(f"/g/{c}", "Abba & Friends", "Waterloo"))
    m.ungroup_artist(store)
    left = [t.path for t in store.all() if t.artist == "Abba & Friends"]
    assert not left, f"copies left grouped: {left}"
