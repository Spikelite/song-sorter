"""Tests for key_online: MusicBrainz -> AcousticBrainz key corroboration.

No network: net_retry.fetch_json is replaced by a scripted fake that, like the
real one, invokes the caller's before_request throttle hook. Backoff behaviour
itself is net_retry's and is tested there.
"""

from __future__ import annotations

import pytest

import key_online as ko
import net_retry


@pytest.fixture
def web(monkeypatch):
    """Route fetches by URL substring. Values are a response dict, an exception
    to raise, or a list consumed one per call."""
    routes: dict = {}
    calls: list[dict] = []

    def fetch(url, *, user_agent, before_request=None, **kwargs):
        calls.append({"url": url, "user_agent": user_agent, **kwargs})
        if before_request is not None:
            before_request()
        for fragment, resp in routes.items():
            if fragment in url:
                if isinstance(resp, list):
                    resp = resp.pop(0)
                if isinstance(resp, BaseException):
                    raise resp
                return resp
        raise AssertionError(f"unrouted fetch: {url}")

    monkeypatch.setattr(net_retry, "fetch_json", fetch)
    monkeypatch.setattr(ko, "_MB_MIN_INTERVAL", 0.0)
    monkeypatch.setattr(ko, "_AB_MIN_INTERVAL", 0.0)
    return routes, calls


def _recs(*pairs):
    """MusicBrainz search payload from (mbid, score) pairs."""
    return {"recordings": [{"id": mbid, "score": score} for mbid, score in pairs]}


def _ab(key, scale):
    return {"tonal": {"key_key": key, "key_scale": scale}}


# --- _throttle -------------------------------------------------------------

def test_throttle_waits_out_the_remaining_interval(monkeypatch) -> None:
    slept = []
    monkeypatch.setattr(ko.time, "sleep", slept.append)
    monkeypatch.setattr(ko, "_last_call", {"svc": ko.time.monotonic()})
    ko._throttle("svc", 5.0)
    assert len(slept) == 1 and 4.0 < slept[0] <= 5.0


def test_throttle_does_not_wait_when_interval_has_passed(monkeypatch) -> None:
    slept = []
    monkeypatch.setattr(ko.time, "sleep", slept.append)
    monkeypatch.setattr(ko, "_last_call", {})
    ko._throttle("svc", 5.0)
    assert slept == []


def test_throttle_is_per_service(monkeypatch) -> None:
    """A MusicBrainz call must not make an AcousticBrainz call wait."""
    slept = []
    monkeypatch.setattr(ko.time, "sleep", slept.append)
    monkeypatch.setattr(ko, "_last_call", {"musicbrainz": ko.time.monotonic()})
    ko._throttle("acousticbrainz", 5.0)
    assert slept == []


# --- _musicbrainz_mbids ----------------------------------------------------

def test_mbids_filter_by_score_and_dedupe(web) -> None:
    routes, calls = web
    routes["musicbrainz.org"] = _recs(("a", 100), ("a", 99), ("b", 90), ("c", 80))
    assert ko._musicbrainz_mbids("Adele", "Hello") == ["a", "b"]
    q = calls[0]["url"]
    assert "artist" in q and "Adele" in q and "Hello" in q
    assert calls[0]["user_agent"] == ko._USER_AGENT
    # key_online retries less than the interactive step: advisory, long batch
    assert calls[0]["max_attempts"] == 3 and calls[0]["max_delay"] == 10.0


def test_mbids_are_capped(web) -> None:
    routes, _ = web
    routes["musicbrainz.org"] = _recs(*[(f"m{i}", 100) for i in range(9)])
    assert ko._musicbrainz_mbids("A", "B") == [f"m{i}" for i in range(ko._MAX_CANDIDATES)]


def test_mbids_skip_records_without_an_id(web) -> None:
    routes, _ = web
    routes["musicbrainz.org"] = {"recordings": [{"score": 100}, {"id": "x", "score": 99}]}
    assert ko._musicbrainz_mbids("A", "B") == ["x"]


def test_mbids_on_fetch_failure_are_empty(web) -> None:
    routes, _ = web
    routes["musicbrainz.org"] = net_retry.RetriesExhausted(3, OSError("down"))
    assert ko._musicbrainz_mbids("A", "B") == []


def test_mbids_escape_lucene_quoting(web) -> None:
    routes, calls = web
    routes["musicbrainz.org"] = _recs()
    ko._musicbrainz_mbids('Say "Hi"', "a\\b")
    assert "%22Hi%22" not in calls[0]["url"]       # inner quotes neutralised


# --- _acousticbrainz_key ---------------------------------------------------

def test_ab_key_is_normalised(web) -> None:
    routes, calls = web
    routes["acousticbrainz.org"] = _ab("Bb", "minor")
    assert ko._acousticbrainz_key("mbid-1") == "A# minor"
    assert "mbid-1" in calls[0]["url"]


@pytest.mark.parametrize("payload", [{}, {"tonal": {}}, {"tonal": {"key_key": "C"}},
                                     {"tonal": {"key_scale": "major"}}])
def test_ab_incomplete_tonal_data_is_none(web, payload) -> None:
    routes, _ = web
    routes["acousticbrainz.org"] = payload
    assert ko._acousticbrainz_key("m") is None


def test_ab_no_data_404_is_none(web) -> None:
    routes, _ = web
    routes["acousticbrainz.org"] = net_retry.PermanentHTTPError(404, "u")
    assert ko._acousticbrainz_key("m") is None


# --- lookup_online ---------------------------------------------------------

def test_lookup_tries_candidates_until_one_has_a_key(web) -> None:
    routes, _ = web
    routes["musicbrainz.org"] = _recs(("m1", 100), ("m2", 95))
    routes["acousticbrainz.org"] = [net_retry.PermanentHTTPError(404, "u"), _ab("E", "minor")]
    assert ko.lookup_online("A", "B") == ("E minor", "acousticbrainz m2")


def test_lookup_uses_and_fills_the_cache(web) -> None:
    routes, calls = web
    routes["musicbrainz.org"] = _recs(("m1", 100), ("m2", 95))
    routes["acousticbrainz.org"] = [_ab("C", "major")]
    cache = {"m1": None}                     # m1 already known to have no data
    assert ko.lookup_online("A", "B", cache) == ("C major", "acousticbrainz m2")
    assert cache == {"m1": None, "m2": "C major"}
    ab_calls = [c for c in calls if "acousticbrainz" in c["url"]]
    assert len(ab_calls) == 1                # m1 served from cache, not refetched


def test_lookup_cache_hit_with_a_key_short_circuits(web) -> None:
    routes, calls = web
    routes["musicbrainz.org"] = _recs(("m1", 100))
    assert ko.lookup_online("A", "B", {"m1": "G major"}) == ("G major", "acousticbrainz m1")
    assert not [c for c in calls if "acousticbrainz" in c["url"]]


def test_lookup_when_no_candidate_has_data(web) -> None:
    routes, _ = web
    routes["musicbrainz.org"] = _recs(("m1", 100))
    routes["acousticbrainz.org"] = net_retry.PermanentHTTPError(404, "u")
    assert ko.lookup_online("A", "B") == (None, "no acousticbrainz key")


def test_lookup_when_musicbrainz_matches_nothing(web) -> None:
    routes, _ = web
    routes["musicbrainz.org"] = _recs(("m1", 10))          # below the score floor
    assert ko.lookup_online("A", "B") == (None, "no musicbrainz match")
