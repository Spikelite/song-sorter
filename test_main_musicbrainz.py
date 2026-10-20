"""main.py: the MusicBrainz corroboration pass, Restitch, and their helpers.

Fully offline. `mb` replaces _mb_fetch with a fake that decodes the real Lucene
query each helper built and hands it to a per-test handler, so query
construction is exercised too, not just response handling.
"""

from __future__ import annotations

import urllib.parse

import pytest

import net_retry
from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


def rec(artist, title):
    return {"title": title, "artist-credit": [{"name": artist, "joinphrase": ""}]}


@pytest.fixture
def mb(m, monkeypatch):
    state = {"handler": lambda q: [], "queries": []}

    def fake_fetch(url):
        q = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["query"][0]
        state["queries"].append(q)
        out = state["handler"](q)
        if isinstance(out, BaseException):
            raise out
        return {"recordings": out}

    monkeypatch.setattr(m, "_mb_fetch", fake_fetch)
    monkeypatch.setattr(m, "_is_online", lambda *a, **k: True)
    return state


class Clock:
    """Stand-in for main's `time`: each monotonic() jumps 100s, forcing every
    periodic-checkpoint branch, and sleep() is free."""

    def __init__(self):
        self.t = 0.0

    def monotonic(self):
        self.t += 100.0
        return self.t

    def sleep(self, s):
        pass


def _review(m):
    from review_state import ReviewState
    rs = ReviewState()
    rs.load(m._REVIEW_STATE_PATH)
    return rs


# --- small helpers ---------------------------------------------------------

def test_is_online_true_and_false(m, monkeypatch) -> None:
    import socket

    class Conn:
        def close(self):
            pass
    monkeypatch.setattr(socket, "create_connection", lambda *a, **k: Conn())
    assert m._is_online() is True

    def refuse(*a, **k):
        raise OSError("unreachable")
    monkeypatch.setattr(socket, "create_connection", refuse)
    assert m._is_online() is False


def test_mb_escape_and_credit_name(m) -> None:
    assert m._mb_escape(' a "b" \\c ') == "a  b   c"
    credit = [{"name": "Kenny Chesney", "joinphrase": " & "},
              {"artist": {"name": "Uncle Kracker"}}, "not-a-dict"]
    assert m._mb_credit_name(credit) == "Kenny Chesney & Uncle Kracker"
    assert m._mb_credit_name(None) == ""


def test_mb_norm_strips_noise_and_flips_only_the_primary(m) -> None:
    assert m._mb_norm("SF123-04 - Waterloo (Mplx) [SF Karaoke]") == "Waterloo"
    assert m._mb_norm("Chesney, Kenny & Uncle Kracker", uncomma=True) == \
        "Kenny Chesney & Uncle Kracker"
    assert m._mb_norm("Four Aces, The", uncomma=True) == "The Four Aces"
    assert m._mb_norm("A, B, C", uncomma=True) == "A, B, C"      # 2 commas: leave
    assert m._mb_norm(", Tom", uncomma=True) == ", Tom"          # empty side: leave
    assert m._mb_norm(None) == ""


def test_mb_sim_ignores_order_noise_articles_and_joiners(m) -> None:
    assert m._mb_sim("Murray, Pete", "Pete Murray") == 100
    assert m._mb_sim("Waterloo (Mplx)", "Waterloo") == 100
    assert m._mb_sim("Four Aces, The", "The Four Aces") == 100
    assert m._mb_sim("A & B", "A and B") == 100
    assert m._mb_sim("Abba", "Metallica") < 50


def test_mb_best_picks_the_highest_combined_score(m) -> None:
    best = m._mb_best([rec("Metallica", "One"), rec("Abba", "Waterloo")], "Abba", "Waterloo")
    assert best[2:] == ("Abba", "Waterloo") and best[0] == best[1] == 100
    assert m._mb_best([], "A", "B") == (0.0, 0.0, "", "")


def test_query_builders_shape_lucene_queries(m, mb) -> None:
    m._mb_search("Chesney, Kenny", 'Say "Hi"')
    m._mb_search_title("Waterloo")
    m._mb_search_release("Circle Of Life", "Lion King")
    m._mb_search_restitch("Belinda Carlisle", ["Heaven", "Earth"])
    assert mb["queries"] == [
        'artist:"Kenny Chesney" AND recording:"Say  Hi"',
        'recording:"Waterloo"',
        'recording:"Circle Of Life" AND release:"Lion King"',
        'artist:"Belinda Carlisle" AND recording:(Heaven AND Earth)',
    ]


def test_mb_throttle_waits_then_records(m, monkeypatch) -> None:
    clock = Clock()
    slept = []
    clock.sleep = slept.append
    monkeypatch.setattr(m, "time", clock)
    monkeypatch.setattr(m, "_MB_MIN_INTERVAL", 1000.0)
    monkeypatch.setattr(m, "_mb_last_call", 0.0)
    m._mb_throttle()           # 100s since t=0 < 1000s interval -> waits
    assert slept and slept[0] > 0
    assert m._mb_last_call > 0


def test_mb_fetch_routes_through_net_retry(m, monkeypatch, capsys) -> None:
    seen = {}

    def fake_fetch_json(url, *, user_agent, before_request, on_retry, **kw):
        seen.update(url=url, ua=user_agent)
        before_request()
        on_retry(1, 2.0, "HTTP 503")
        return {"recordings": []}
    monkeypatch.setattr(net_retry, "fetch_json", fake_fetch_json)
    assert m._mb_fetch("https://mb/x") == {"recordings": []}
    assert seen == {"url": "https://mb/x", "ua": m._MB_USER_AGENT}
    assert "retry 1 in 2.0s" in capsys.readouterr().out


# --- musicbrainz_lookup ----------------------------------------------------

def test_lookup_skips_cleanly_offline(m, store, prompts, monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_online", lambda *a, **k: False)
    m.musicbrainz_lookup(store)
    assert "skipped (offline)" in prompts.output


def test_lookup_with_nothing_to_do(m, store, prompts, mb) -> None:
    prompts.queue(False)                        # recheck?
    m.musicbrainz_lookup(store)
    assert "No tracks to look up" in prompts.output


def _answers(q):
    """A small fake MusicBrainz. Each track below exercises one outcome."""
    if 'artist:"Abba" AND recording:"Waterloo"' in q:
        return [rec("ABBA", "Waterloo")]                             # direct ok
    if 'artist:"Mamma Mia" AND recording:"Abba"' in q:
        return []                                                    # reversed
    if 'artist:"Abba" AND recording:"Mamma Mia"' in q:
        return [rec("ABBA", "Mamma Mia")]                            # -> swap
    if q == 'recording:"Dancing Queen"':
        return [rec("Someone", "Dancing Queen"), rec("ABBA", "Dancing Queen")]  # title
    if q == 'recording:"Circle Of Life" AND release:"Lion King"':
        return [rec("Elton John", "Circle Of Life")]                 # suggest
    if 'recording:"Fernand0"' in q:
        return [rec("ABBA", "Fernando")]                             # weak -> flag
    return []


def test_lookup_classifies_every_outcome(m, store, prompts, mb) -> None:
    mb["handler"] = _answers
    tracks = {
        "/ok": ("Abba", "Waterloo"),
        "/swap": ("Mamma Mia", "Abba"),
        "/title": ("Abba", "Dancing Queen"),
        "/sug": ("Lion King", "Circle Of Life"),
        "/flag": ("Abba", "Fernand0"),
        "/none": ("Nobody", "Nothing"),
        "/blank": ("", "Silent"),
    }
    for p, (a, s) in tracks.items():
        store.add(make_track(p, a, s))
    prompts.queue(False, True)                  # recheck? no.  verbose? yes.
    m.musicbrainz_lookup(store)

    md = {t.path: t.metadata for t in store.all()}
    assert md["/ok"]["mb_match"] == "ok"
    assert md["/swap"]["mb_match"] == "swap"
    assert (store.get("/swap").artist, store.get("/swap").song) == ("Abba", "Mamma Mia")
    assert md["/title"]["mb_match"] == "title"
    assert md["/sug"]["mb_match"] == "suggest"
    assert md["/sug"]["mb_artist"] == "Elton John"
    assert md["/flag"]["mb_match"] == "flag"
    assert md["/flag"]["mb_title"] == "Fernando"
    assert md["/none"]["mb_match"] == "none"
    assert "mb_checked" not in md["/blank"]                 # skipped, not scored
    rs = _review(m)
    assert {p for p in tracks if rs.get(p) == "ok"} == {"/ok", "/swap", "/title"}
    assert "ok'd 3 (incl 1 swapped), flagged 1, suggested 1, no-match 1" in prompts.output


def test_lookup_memoises_duplicate_pairs(m, store, prompts, mb) -> None:
    mb["handler"] = _answers
    store.add(make_track("/1", "Abba", "Waterloo"))
    store.add(make_track("/2", "abba", "waterloo"))           # same pair, other case
    prompts.queue(False, False)
    m.musicbrainz_lookup(store)
    assert len(mb["queries"]) == 1
    assert all(t.metadata["mb_match"] == "ok" for t in store.all())


def test_lookup_recheck_revisits_weak_results_only(m, store, prompts, mb) -> None:
    mb["handler"] = _answers
    store.add(make_track("/weak", "Abba", "Waterloo", mb_checked=1, mb_match="none"))
    store.add(make_track("/done", "Abba", "SOS", mb_checked=1, mb_match="ok"))
    prompts.queue(True, False)                  # recheck? yes
    m.musicbrainz_lookup(store)
    assert store.get("/weak").metadata["mb_match"] == "ok"     # re-scored
    assert len(mb["queries"]) == 1                              # /done untouched


def test_lookup_permanent_error_skips_without_a_strike(m, store, prompts, mb, capsys) -> None:
    def handler(q):
        if "Broken" in q:
            return net_retry.PermanentHTTPError(400, "u")
        return _answers(q)
    mb["handler"] = handler
    for i in range(6):                                   # more than the strike budget
        store.add(make_track(f"/b{i}", f"Band {i}", f"Broken {i}"))   # 1 track each: thin
    store.add(make_track("/ok", "Abba", "Waterloo"))
    prompts.queue(False, False)
    m.musicbrainz_lookup(store)
    assert store.get("/ok").metadata["mb_match"] == "ok"      # run was NOT aborted
    assert "HTTP 400" in capsys.readouterr().out


def test_lookup_stops_after_consecutive_exhausted_retries(m, store, prompts, mb, capsys) -> None:
    mb["handler"] = lambda q: net_retry.RetriesExhausted(5, OSError("down"))
    for i in range(8):
        store.add(make_track(f"/t{i}", f"Band {i}", f"Song {i}"))     # 1 track each: thin
    prompts.queue(False, False)
    m.musicbrainz_lookup(store)
    out = capsys.readouterr().out
    assert "still failing after 5 consecutive tracks" in out
    assert len(mb["queries"]) == m._MB_MAX_STRIKES           # stopped, did not grind on
    assert m._CACHE_PATH.exists()                             # progress saved on the way out


def test_lookup_checkpoints_periodically(m, store, prompts, mb, monkeypatch) -> None:
    monkeypatch.setattr(m, "time", Clock())
    mb["handler"] = _answers
    store.add(make_track("/ok", "Abba", "Waterloo"))
    prompts.queue(False, False)
    m.musicbrainz_lookup(store)
    assert m._REVIEW_STATE_PATH.exists()


# --- _reliable_words -------------------------------------------------------

def test_reliable_words_keep_only_certainly_complete_words(m) -> None:
    # 'Pla' may be cut at the dash; 'On' starts after a cut; 'Earth' is the end.
    assert m._reliable_words(["Heaven Is A Pla", "On Earth"]) == ["Heaven", "Earth"]
    assert m._reliable_words(["Hot"]) == []                    # sole word may be cut
    assert m._reliable_words(["Tur", "E Loose Now", "Baby Please"]) == ["Please", "Loose"]


@pytest.mark.xfail(strict=True, reason="GH #26: equal-length words are picked in "
                   "hash-seed order, so Restitch queries differ between runs")
def test_reliable_words_are_deterministic_under_ties(m) -> None:
    import subprocess
    import sys
    code = ("import main; print(main._reliable_words("
            "['Alpha Bravo Charl', 'Delta Echoo Foxtr', 'Golfo Hotel']))")
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                           env={"PYTHONHASHSEED": str(seed), "PATH": ""},
                           cwd=str(__import__("pathlib").Path(m.__file__).parent)).stdout
            for seed in range(1, 7)}
    assert len(outs) == 1, f"varies with hash seed: {outs}"


# --- restitch_titles -------------------------------------------------------

def _belinda(store, n=2):
    for i in range(n):
        store.add(make_track(f"/k/bc{i}", "Belinda Carlisle", f"Hit {i}"))


def test_restitch_skips_cleanly_offline(m, store, prompts, monkeypatch) -> None:
    monkeypatch.setattr(m, "_is_online", lambda *a, **k: False)
    m.restitch_titles(store)
    assert "skipped (offline)" in prompts.output


def test_restitch_with_nothing_to_do(m, store, prompts, mb) -> None:
    store.add(make_track("/short/A - B.zip", "A", "B"))        # < 3 segments
    prompts.queue(False)
    m.restitch_titles(store)
    assert "No dash-elided stems" in prompts.output


def test_restitch_applies_a_single_unambiguous_hit(m, store, prompts, mb) -> None:
    _belinda(store)
    t = make_track("/l/FLY-03-06 - Belinda - Carlisle - Heaven Is A Pla - On Earth.zip",
                   "Belinda", "Carlisle")
    store.add(t)
    mb["handler"] = lambda q: [rec("Belinda Carlisle", "Heaven Is A Place On Earth")]
    prompts.queue(False)
    m.restitch_titles(store)
    assert (t.artist, t.song) == ("Belinda Carlisle", "Heaven Is A Place On Earth")
    assert t.metadata["restitch"] == "ok"
    assert _review(m).get(t.path) == "ok"
    assert mb["queries"] == ['artist:"Belinda Carlisle" AND recording:(Heaven AND Earth)']


def test_restitch_suggests_when_ambiguous(m, store, prompts, mb) -> None:
    _belinda(store)
    t = make_track("/l/FLY-03-06 - Belinda - Carlisle - Heaven Is A Pla - On Earth.zip",
                   "Belinda", "Carlisle")
    store.add(t)
    mb["handler"] = lambda q: [rec("Belinda Carlisle", "Heaven Is A Place On Earth"),
                               rec("Belinda Carlisle", "Heaven Is A Place On Earth (Remix)")]
    prompts.queue(False)
    m.restitch_titles(store)
    assert t.metadata["restitch"] == "suggest"
    assert t.metadata["mb_match"] == "suggest"
    assert t.song == "Carlisle"                               # not auto-applied


def test_restitch_uses_a_known_current_artist(m, store, prompts, mb) -> None:
    for i in range(2):
        store.add(make_track(f"/k/q{i}", "Queen", f"Hit {i}"))
    t = make_track("/l/SFKK-01-01 - Queen - Bohemian Rhaps - Ody.zip", "Queen", "Bohemian")
    store.add(t)
    mb["handler"] = lambda q: [rec("Queen", "Bohemian Rhapsody")]
    prompts.queue(False)
    m.restitch_titles(store)
    assert t.song == "Bohemian Rhapsody"


def test_restitch_ignores_ineligible_tracks(m, store, prompts, mb) -> None:
    from review_state import ReviewState
    for i in range(2):
        store.add(make_track(f"/k/q{i}", "Queen", f"Hit {i}"))
    store.add(make_track("/l/SFKK-1 - Foo - Bar - Baz.zip", "Queen", "x"))     # no prefix match
    store.add(make_track("/l/SFKK-2 - Foo - Bar - Baz.zip", "Nobody", "x"))    # unknown artist
    store.add(make_track("/l/SFKK-3 - Queen - A - B.zip", "Queen", "x", restitch="none"))
    done = make_track("/l/SFKK-4 - Queen - A - B.zip", "Queen", "x")
    store.add(done)
    rs = ReviewState()
    rs.set(done.path, "ok")
    rs.save(m._REVIEW_STATE_PATH)
    prompts.queue(False)                                        # no recheck
    m.restitch_titles(store)
    assert "No dash-elided stems" in prompts.output


def test_restitch_recheck_and_no_reliable_words(m, store, prompts, mb) -> None:
    for i in range(2):
        store.add(make_track(f"/k/q{i}", "Queen", f"Hit {i}"))
    t = make_track("/l/SFKK-1 - Queen - Xy - Zq.zip", "Queen", "x", restitch="none")
    store.add(t)
    prompts.queue(True)                                         # recheck
    m.restitch_titles(store)
    assert t.metadata["restitch"] == "none"                     # words too short
    assert mb["queries"] == []                                  # never queried


def test_restitch_no_match_and_short_evidence(m, store, prompts, mb) -> None:
    _belinda(store)
    miss = make_track("/l/FLY-1 - Belinda - Carlisle - Heaven Is A Pla - On Earth.zip",
                      "Belinda", "Carlisle")
    thin = make_track("/l/FLY-2 - Belinda - Carlisle - Leave A - On.zip",
                      "Belinda", "Carlisle")
    store.add(miss)
    store.add(thin)

    def handler(q):
        if "Heaven" in q:
            return [rec("Someone Else", "Heaven Is A Place On Earth")]  # artist mismatch
        return [rec("Belinda Carlisle", "Leave A Light On")]            # < 10 chars evidence
    mb["handler"] = handler
    prompts.queue(False)
    m.restitch_titles(store)
    assert miss.metadata["restitch"] == "none"
    assert thin.metadata["restitch"] == "suggest"
    assert "applied 0, suggested 1" in prompts.output


def test_restitch_error_handling_and_checkpoints(m, store, prompts, mb, monkeypatch, capsys) -> None:
    monkeypatch.setattr(m, "time", Clock())
    _belinda(store)
    store.add(make_track("/l/FLY-0 - Belinda - Carlisle - Perm Anent - Failure.zip", "B", "C"))
    for i in range(1, 7):
        store.add(make_track(f"/l/FLY-{i} - Belinda - Carlisle - Heaven Is A Pla - On Earth{i}.zip",
                             "B", "C"))

    def handler(q):
        if "(Perm)" in q:
            return net_retry.PermanentHTTPError(400, "u")
        return net_retry.RetriesExhausted(5, OSError("down"))
    mb["handler"] = handler
    prompts.queue(False)
    m.restitch_titles(store)
    out = capsys.readouterr().out
    assert "HTTP 400" in out
    assert "still failing after 5 consecutive tracks" in out
    assert m._CACHE_PATH.exists()


def test_restitch_periodic_checkpoint_on_success(m, store, prompts, mb, monkeypatch) -> None:
    monkeypatch.setattr(m, "time", Clock())
    _belinda(store)
    store.add(make_track("/l/FLY-1 - Belinda - Carlisle - Heaven Is A Pla - On Earth.zip", "B", "C"))
    mb["handler"] = lambda q: []
    prompts.queue(False)
    m.restitch_titles(store)
    assert m._REVIEW_STATE_PATH.exists()
