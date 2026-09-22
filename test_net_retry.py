"""Tests for net_retry: backoff maths and failure classification.

No network and no real sleeping -- the opener and sleep are injected.
"""

from __future__ import annotations

import email.message
import io
import json
import urllib.error

import pytest

from net_retry import (
    PermanentHTTPError,
    RetriesExhausted,
    backoff_delay,
    fetch_json,
    retry_after_seconds,
)


class _Resp(io.BytesIO):
    """Minimal stand-in for an HTTP response usable as a context manager."""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int, retry_after: str | None = None) -> urllib.error.HTTPError:
    hdrs = email.message.Message()
    if retry_after is not None:
        hdrs["Retry-After"] = retry_after
    return urllib.error.HTTPError("http://x", code, "boom", hdrs, None)


def _opener(script):
    """Fake urlopen driven by `script`: each entry is an exception to raise or
    a payload to return. Records how many times it was called."""
    calls = {"n": 0}

    def open_url(req, timeout=None):
        i = calls["n"]
        calls["n"] += 1
        item = script[min(i, len(script) - 1)]
        if isinstance(item, BaseException):
            raise item
        return _Resp(json.dumps(item).encode())

    open_url.calls = calls
    return open_url


# --- backoff_delay ----------------------------------------------------------

def test_backoff_doubles_from_base() -> None:
    assert backoff_delay(1, base=2, jitter=0.0) == 2
    assert backoff_delay(2, base=2, jitter=0.0) == 4
    assert backoff_delay(3, base=2, jitter=0.0) == 8
    assert backoff_delay(4, base=2, jitter=0.0) == 16


def test_backoff_is_capped() -> None:
    assert backoff_delay(10, base=2, cap=30, jitter=0.0) == 30


def test_backoff_honours_retry_after_within_cap() -> None:
    assert backoff_delay(1, base=2, cap=30, retry_after=7, jitter=0.0) == 7
    # A server asking for longer than the cap falls back to our own schedule.
    assert backoff_delay(1, base=2, cap=30, retry_after=999, jitter=0.0) == 2


def test_backoff_jitter_stays_in_band() -> None:
    for _ in range(50):
        d = backoff_delay(3, base=2, cap=60)      # nominal 8s, +/-25%
        assert 6.0 <= d <= 10.0


def test_retry_after_when_error_has_no_headers() -> None:
    """An HTTPError built without headers must not crash the backoff path."""
    err = urllib.error.HTTPError("http://x", 503, "boom", None, None)
    assert retry_after_seconds(err) is None


def test_retry_after_negative_is_rejected() -> None:
    assert retry_after_seconds(_http_error(503, "-5")) is None


def test_retry_after_parsing() -> None:
    assert retry_after_seconds(_http_error(503, "5")) == 5.0
    assert retry_after_seconds(_http_error(503)) is None
    assert retry_after_seconds(_http_error(503, "next Tuesday")) is None


# --- fetch_json classification ---------------------------------------------

def test_success_first_attempt() -> None:
    op = _opener([{"recordings": [1]}])
    out = fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None)
    assert out == {"recordings": [1]}
    assert op.calls["n"] == 1


def test_retries_then_succeeds() -> None:
    sleeps = []
    op = _opener([_http_error(503), _http_error(503), {"ok": True}])
    out = fetch_json("http://x", user_agent="t", opener=op, sleep=sleeps.append)
    assert out == {"ok": True}
    assert op.calls["n"] == 3
    assert len(sleeps) == 2          # slept before each retry, not after the win


def test_persistent_503_raises_retries_exhausted() -> None:
    sleeps = []
    op = _opener([_http_error(503)])
    with pytest.raises(RetriesExhausted) as e:
        fetch_json("http://x", user_agent="t", opener=op, sleep=sleeps.append,
                   max_attempts=4)
    assert op.calls["n"] == 4
    assert len(sleeps) == 3          # no sleep after the final failure
    assert e.value.attempts == 4


def test_permanent_4xx_is_not_retried() -> None:
    """A malformed query (400) can never succeed -- fail fast, don't burn retries."""
    sleeps = []
    op = _opener([_http_error(400)])
    with pytest.raises(PermanentHTTPError) as e:
        fetch_json("http://x", user_agent="t", opener=op, sleep=sleeps.append)
    assert op.calls["n"] == 1
    assert sleeps == []
    assert e.value.status == 400


def test_404_is_permanent() -> None:
    """AcousticBrainz answers 404 for recordings it has no data on -- that is a
    normal, immediate 'no', not something to retry."""
    op = _opener([_http_error(404)])
    with pytest.raises(PermanentHTTPError):
        fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None)
    assert op.calls["n"] == 1


def test_connection_errors_are_retried() -> None:
    op = _opener([urllib.error.URLError("dns"), {"ok": 1}])
    out = fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None)
    assert out == {"ok": 1}
    assert op.calls["n"] == 2


def test_garbled_json_is_retried() -> None:
    op = _opener([json.JSONDecodeError("bad", "", 0), {"ok": 1}])
    out = fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None)
    assert out == {"ok": 1}


def test_non_transport_exception_propagates() -> None:
    """A bug must not be laundered into 'lost connection'."""
    op = _opener([KeyError("artist-credit")])
    with pytest.raises(KeyError):
        fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None)
    assert op.calls["n"] == 1


def test_before_request_runs_on_every_attempt_including_retries() -> None:
    """Regression: the old throttle sat AFTER the request, so a failure skipped
    it and the loop accelerated during an outage. Pacing must precede retries."""
    ticks = []
    op = _opener([_http_error(503), _http_error(503), {"ok": 1}])
    fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None,
               before_request=lambda: ticks.append(1))
    assert len(ticks) == 3           # once per attempt, retries included


def test_on_retry_hook_reports_each_backoff() -> None:
    seen = []
    op = _opener([_http_error(503), {"ok": 1}])
    fetch_json("http://x", user_agent="t", opener=op, sleep=lambda s: None,
               on_retry=lambda attempt, delay, err: seen.append((attempt, delay)))
    assert len(seen) == 1
    assert seen[0][0] == 1 and seen[0][1] > 0
