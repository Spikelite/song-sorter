"""main.py: ID3-tag-driven passes (Tag-review, Tag-swap) and their helpers.

These heuristics carry guards learned from real incidents (reversed rips with
reversed tags, band names with commas), so each rule is pinned individually.
"""

from __future__ import annotations

import pytest

from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


def _known(store, artist, n=4):
    """n >= 4 tracks: 'known' for the low_bound=3 checks these passes use."""
    for i in range(n):
        store.add(make_track(f"/known/{artist}/{i}", artist, f"Hit {i}"))


def _review(m):
    from review_state import ReviewState
    rs = ReviewState()
    rs.load(m._REVIEW_STATE_PATH)
    return rs


# --- _auto_clean_artist ----------------------------------------------------

@pytest.mark.parametrize("raw, want", [
    ("Abba", ("Abba", None)),
    ("Abba & Friends", ("Abba", "Friends")),
    ("Abba feat. Friends", ("Abba", "Friends")),
    ("Abba FT Friends", ("Abba", "Friends")),
    ("Abba featuring Friends", ("Abba", "Friends")),
    ("Abba With Friends", ("Abba", "Friends")),
    ("Jones, Tom and Friends", ("Tom Jones", "Friends")),     # uncomma'd primary
])
def test_auto_clean_artist(m, raw, want) -> None:
    assert m._auto_clean_artist(raw) == want


@pytest.mark.xfail(strict=True, reason="GH #25: comma band names are inverted "
                   "('Earth, Wind & Fire' -> 'Wind Earth') because the credit is "
                   "split on '&' before uncomma_artist can see the band guard")
@pytest.mark.parametrize("band", ["Earth, Wind & Fire", "Crosby, Stills & Nash",
                                  "Blood, Sweat & Tears", "Peter, Paul & Mary"])
def test_auto_clean_artist_leaves_comma_band_names_intact(m, band) -> None:
    primary, _feature = m._auto_clean_artist(band)
    assert primary != " ".join(reversed(band.split("&")[0].replace(",", "").split()))


# --- _artist_tokens --------------------------------------------------------

def test_artist_tokens_are_order_and_joiner_insensitive(m) -> None:
    assert m._artist_tokens("Murray, Pete") == m._artist_tokens("Pete Murray")
    assert m._artist_tokens("A & B") == m._artist_tokens("A And B") \
        == m._artist_tokens("A feat B") == m._artist_tokens("A ft B")


def test_artist_tokens_strip_catalog_prefix_and_karaoke_noise(m) -> None:
    assert m._artist_tokens("SFDU11-06 - Tom Jones (Karaoke) wvocal") == \
        frozenset({"tom", "jones"})
    assert m._artist_tokens(None) == frozenset()


# --- _song_as_known_artist -------------------------------------------------

def test_song_as_known_artist_prefers_the_comma_flip(m) -> None:
    known = {"celine dion", "dion, celine"}
    assert m._song_as_known_artist("Dion, Celine", known) == "Celine Dion"


def test_song_as_known_artist_direct_match_and_none(m) -> None:
    known = {"earth, wind & fire", "avril lavigne"}
    assert m._song_as_known_artist("Earth, Wind & Fire", known) == "Earth, Wind & Fire"
    assert m._song_as_known_artist("Avril Lavigne", known) == "Avril Lavigne"
    assert m._song_as_known_artist("Complicated", known) is None


# --- auto_ok_from_tags (Tag-review) ----------------------------------------

def test_tag_review_oks_corroborated_tracks_only(m, store, prompts) -> None:
    _known(store, "Avril Lavigne")
    store.add(make_track("/ok", "Murray, Pete", "Opportunity", tag_artist="Pete Murray"))
    store.add(make_track("/diff", "Pete Murray", "Better Days", tag_artist="Someone Else"))
    store.add(make_track("/none", "Pete Murray", "So Beautiful", tag_artist="<not-found>"))
    store.add(make_track("/blank", "Pete Murray", "Class A"))
    store.add(make_track("/noise", "(Karaoke)", "Weird", tag_artist="(Karaoke)"))
    m.auto_ok_from_tags(store)
    rs = _review(m)
    assert rs.get("/ok") == "ok"
    for p in ("/diff", "/none", "/blank", "/noise"):
        assert rs.get(p) is None, p
    assert "Auto-ok'd 1 tracks" in prompts.output
    assert "swap-suspect" not in prompts.output


def test_tag_review_never_trusts_a_tag_on_a_swap_suspect(m, store, prompts) -> None:
    """Reversed rips often carry reversed tags, so agreement proves nothing
    when the SONG field is a known artist and the artist field is not."""
    _known(store, "Avril Lavigne")
    store.add(make_track("/s", "Complicated", "Avril Lavigne", tag_artist="Complicated"))
    m.auto_ok_from_tags(store)
    assert _review(m).get("/s") is None
    assert "left 1 swap-suspect(s)" in prompts.output


# --- swap_from_tags (Tag-swap) ---------------------------------------------

def test_tag_swap_applies_each_kind_of_evidence(m, store, prompts) -> None:
    _known(store, "Avril Lavigne")
    _known(store, "Celine Dion")
    cases = {
        # tag names the SONG field
        "/tag": (("Complicated", "Avril Lavigne", "Avril Lavigne"),
                 ("Avril Lavigne", "Complicated")),
        # comma form, no tag at all
        "/comma": (("My Heart Will Go On", "Dion, Celine", None),
                   ("Celine Dion", "My Heart Will Go On")),
        # comma form beats a tag that 'agrees' with the reversed orientation
        "/rev": (("Sk8er Boi", "Lavigne, Avril", "Sk8er Boi"),
                 ("Avril Lavigne", "Sk8er Boi")),
    }
    for path, ((artist, song, tag), _want) in cases.items():
        md = {"tag_artist": tag} if tag else {}
        store.add(make_track(path, artist, song, **md))
    m.swap_from_tags(store)
    got = {t.path: (t.artist, t.song) for t in store.all()}
    rs = _review(m)
    for path, (_given, want) in cases.items():
        assert got[path] == want, path
        assert rs.get(path) == "ok", path              # swapped tracks are settled
    assert "Swapped 3 reversed" in prompts.output


@pytest.mark.parametrize("artist, song, tag", [
    ("Tiny Band", "Avril Lavigne", "Tiny Band"),     # agreeing tag vetoes
    ("Some Title", "Avril Lavigne", None),           # no evidence either way
    ("Some Title", "Avril Lavigne", "<not-found>"),  # sentinel == no tag
    ("Unknown", "Avril Lavigne", "Avril Lavigne"),   # unknown artist
    ("", "Avril Lavigne", "Avril Lavigne"),
    ("Celine Dion", "Avril Lavigne", "Avril Lavigne"),  # current artist is known
    ("X", "Not An Artist", "Not An Artist"),         # song is no known artist
    ("SC\\SC-199", "Avril Lavigne", "Avril Lavigne"),   # catalog-path artist
])
def test_tag_swap_leaves_these_alone(m, store, prompts, artist, song, tag) -> None:
    _known(store, "Avril Lavigne")
    _known(store, "Celine Dion")
    md = {"tag_artist": tag} if tag else {}
    store.add(make_track("/t", artist, song, **md))
    m.swap_from_tags(store)
    t = store.get("/t")
    assert (t.artist, t.song) == (artist, song)
    assert "Swapped 0 reversed" in prompts.output


def test_tag_swap_skips_a_missing_song(m, store, prompts) -> None:
    _known(store, "Avril Lavigne")
    store.add(make_track("/t", "Anything", ""))
    m.swap_from_tags(store)
    assert store.get("/t").artist == "Anything"
