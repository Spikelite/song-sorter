"""main.py: the sequential Review flow.

Only (ok) and accepting a suggestion mark a track reviewed and advance;
swap/edit/auto-clean change the track but stay on it; (skip) advances without
marking. Identical entries reuse an ok/skip decision within a session.
"""

from __future__ import annotations

import pytest

from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


def _review(m):
    from review_state import ReviewState
    rs = ReviewState()
    rs.load(m._REVIEW_STATE_PATH)
    return rs


def offered(sink: list, value):
    """Answer `value` to a select, recording the titles it offered."""
    def choose(kind, message, kwargs):
        sink.append([str(getattr(c, "title", c)) for c in kwargs["choices"]])
        return value
    return choose


def test_review_with_an_empty_queue(m, store, prompts) -> None:
    m.review_mode(store)
    assert "No tracks from artists with 5 or fewer" in prompts.output


def test_ok_marks_reviewed_and_session_cache_reuses_it(m, store, prompts) -> None:
    store.add(make_track("/a", "Abba", "Waterloo"))
    store.add(make_track("/b", "abba", "waterloo"))     # identical entry
    prompts.queue("ok")                                # asked only once
    m.review_mode(store)
    rs = _review(m)
    assert rs.get("/a") == rs.get("/b") == "ok"
    assert [k for k, _ in prompts.asked].count("select") == 1


def test_skip_advances_without_marking_and_is_cached(m, store, prompts) -> None:
    store.add(make_track("/a", "Abba", "Waterloo"))
    store.add(make_track("/b", "Abba", "Waterloo"))
    prompts.queue("skip")
    m.review_mode(store)
    rs = _review(m)
    assert rs.get("/a") is None and rs.get("/b") is None


@pytest.mark.parametrize("answer", ["exit", None])
def test_exit_or_cancel_saves_and_stops(m, store, prompts, answer) -> None:
    store.add(make_track("/a", "Abba", "Waterloo"))
    store.add(make_track("/b", "Blur", "Song 2"))
    prompts.queue(answer)
    m.review_mode(store)
    assert m._REVIEW_STATE_PATH.exists()
    assert _review(m).get("/a") is None


def test_swap_stays_on_the_track_until_ok(m, store, prompts) -> None:
    t = make_track("/a", "Waterloo", "Abba")
    store.add(t)
    prompts.queue("swap", "ok")
    m.review_mode(store)
    assert (t.artist, t.song) == ("Abba", "Waterloo")
    assert _review(m).get("/a") == "ok"


def test_edit_delegates_then_stays(m, store, prompts) -> None:
    t = make_track("/a", "Abba", "Waterlo")
    store.add(t)
    prompts.queue("edit", "song", "Waterloo", "ok")
    m.review_mode(store)
    assert t.song == "Waterloo"


def test_auto_clean_splits_the_feature(m, store, prompts) -> None:
    t = make_track("/a", "Jones, Tom & Friends", "Delilah")
    store.add(t)
    seen = []
    prompts.queue(offered(seen, "auto-clean"), "ok")
    m.review_mode(store)
    assert "auto [Tom Jones]" in seen[0]
    assert t.artist == "Tom Jones"
    assert t.metadata["feature"] == "Friends"


def test_auto_clean_without_a_feature(m, store, prompts) -> None:
    t = make_track("/a", "Jones, Tom", "Delilah")
    store.add(t)
    prompts.queue("auto-clean", "ok")
    m.review_mode(store)
    assert t.artist == "Tom Jones"
    assert "feature" not in t.metadata


def test_swapped_parse_warning(m, store, prompts) -> None:
    # 6 tracks: 'known' (> 3) yet too busy (> 5) to enter the review queue
    for i in range(6):
        store.add(make_track(f"/k{i}", "Avril Lavigne", f"Hit {i}"))
    store.add(make_track("/s", "Complicated", "Avril Lavigne"))
    prompts.queue("skip")
    m.review_mode(store)
    assert "probably swapped" in prompts.output


# Comparison is case/punctuation-insensitive (_norm_eq), so a suggestion only
# counts as different when it differs after normalisation -- hence 'Abbba'.
@pytest.mark.parametrize("artist, md, label, want", [
    ("Abbba", {"mb_artist": "ABBA", "mb_title": "Waterloo (Live)", "mb_match": "flag"},
     "use MB -> artist 'ABBA', song 'Waterloo (Live)'", ("ABBA", "Waterloo (Live)")),
    ("Abbba", {"mb_artist": "ABBA", "mb_title": "Waterloo"},
     "use MB -> artist 'ABBA'", ("ABBA", "Waterloo")),
    ("Abba", {"mb_artist": "ABBA", "mb_title": "Waterloo (Live)"},
     "use MB -> song 'Waterloo (Live)'", ("Abba", "Waterloo (Live)")),
])
def test_accepting_a_musicbrainz_suggestion(m, store, prompts, artist, md, label, want) -> None:
    t = make_track("/a", artist, "Waterloo", **md)
    store.add(t)
    seen = []
    prompts.queue(offered(seen, "mb-accept"))
    m.review_mode(store)
    assert label in seen[0]
    assert (t.artist, t.song) == want
    assert _review(m).get("/a") == "ok"


def test_matching_mb_suggestion_offers_nothing(m, store, prompts) -> None:
    t = make_track("/a", "Abba", "Waterloo", mb_artist="ABBA", mb_title="waterloo")
    store.add(t)
    seen = []
    prompts.queue(offered(seen, "skip"))
    m.review_mode(store)
    assert not [x for x in seen[0] if x.startswith("use MB")]


def test_mb_accept_without_a_suggestion_is_a_noop(m, store, prompts) -> None:
    """The branch guards against a stale choice; it must neither crash nor mark."""
    t = make_track("/a", "Abba", "Waterloo")
    store.add(t)
    prompts.queue(lambda k, msg, kw: "mb-accept", "exit")
    m.review_mode(store)
    assert _review(m).get("/a") is None


def test_accepting_the_id3_tag(m, store, prompts) -> None:
    t = make_track("/a", "Abbba", "Waterloo", tag_artist="ABBA (Karaoke)")
    store.add(t)
    seen = []
    prompts.queue(offered(seen, "tag-accept"))
    m.review_mode(store)
    assert "use tag -> artist 'ABBA'" in seen[0]
    assert t.artist == "ABBA"
    assert _review(m).get("/a") == "ok"


@pytest.mark.parametrize("tag", ["Abba", "<not-found>"])
def test_no_tag_suggestion_when_it_agrees_or_is_absent(m, store, prompts, tag) -> None:
    store.add(make_track("/a", "Abba", "Waterloo", tag_artist=tag))
    seen = []
    prompts.queue(offered(seen, "skip"))
    m.review_mode(store)
    assert not [x for x in seen[0] if x.startswith("use tag")]


def test_tag_accept_without_a_tag_is_a_noop(m, store, prompts) -> None:
    store.add(make_track("/a", "Abba", "Waterloo"))
    prompts.queue(lambda k, msg, kw: "tag-accept", "exit")
    m.review_mode(store)
    assert _review(m).get("/a") is None
