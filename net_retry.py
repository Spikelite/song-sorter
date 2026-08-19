"""HTTP JSON fetch with failure classification and exponential backoff.

Shared by every outbound call song-sorter makes (MusicBrainz in `main.py`,
MusicBrainz + AcousticBrainz in `key_online.py`), so retry behaviour is defined
in exactly one place.

The problem this solves: the callers used to wrap requests in a bare
``except Exception`` and count consecutive failures, treating a rate-limit, a
malformed query and a genuine code bug identically. Worse, their throttle sat
*after* the request, so it was skipped whenever the request raised -- during an
outage the loop sped up instead of backing off, and burned its whole failure
budget in milliseconds.

Failures are sorted into three kinds:

* **Retryable** -- 429/5xx (MusicBrainz answers 503 when you exceed its ~1
  req/sec limit), connection errors, timeouts, and truncated/garbled JSON.
  These are retried with exponential backoff plus jitter, honouring a
  ``Retry-After`` header when the server sends one.
* **Permanent** -- other 4xx (a 400 from a malformed Lucene query, a 404 for a
  recording AcousticBrainz simply has no data for). Raised immediately as
  `PermanentHTTPError`; retrying cannot help, and a caller should skip that item
  rather than count it as a connectivity failure.
* **Anything else** -- propagated untouched. A ``KeyError`` in parsing is a bug,
  and reporting it as "lost connection" is how such bugs stay hidden.
"""

from __future__ import annotations

import json
import random
import socket
import time
import urllib.error
import urllib.request


class PermanentHTTPError(Exception):
    """The server refused in a way retrying cannot fix (non-retryable 4xx)."""

    def __init__(self, status: int, url: str) -> None:
        self.status = status
        self.url = url
        super().__init__(f"HTTP {status} (permanent)")


class RetriesExhausted(Exception):
    """A retryable failure that was still failing after the last attempt."""

    def __init__(self, attempts: int, last_error: BaseException) -> None:
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(f"{last_error} (after {attempts} attempts)")


# 503 is MusicBrainz's rate-limit/busy answer; 429 is the explicit rate limit;
# 5xx are server-side; 408/425 are timing-related. All worth another go.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

DEFAULT_MAX_ATTEMPTS = 5      # one initial try plus four retries
DEFAULT_BASE_DELAY = 2.0      # seconds before the first retry
DEFAULT_MAX_DELAY = 30.0      # ceiling for any single wait


def retry_after_seconds(err: urllib.error.HTTPError) -> float | None:
    """The server's requested wait from a ``Retry-After`` header, if usable.

    Only the delta-seconds form is honoured; the HTTP-date form is rare in
    practice and not worth mis-parsing. Returns None when absent or nonsense."""
    try:
        raw = err.headers.get("Retry-After")  # type: ignore[union-attr]
    except Exception:
        return None
    if not raw:
        return None
    try:
        secs = float(str(raw).strip())
    except (TypeError, ValueError):
        return None
    return secs if secs >= 0 else None


def backoff_delay(attempt: int, *, base: float = DEFAULT_BASE_DELAY,
                  cap: float = DEFAULT_MAX_DELAY,
                  retry_after: float | None = None,
                  jitter: float | None = None) -> float:
    """Seconds to wait before retry number `attempt` (1-based). Pure.

    Doubles per attempt from `base`, clamped to `cap`. A server-supplied
    `retry_after` wins when it is within the cap. Jitter of +/-25% is applied so
    a batch of failures doesn't resynchronise into a thundering herd; pass an
    explicit `jitter` (a fraction, e.g. 0.0) to make the result deterministic."""
    if retry_after is not None and retry_after <= cap:
        delay = retry_after
    else:
        delay = min(cap, base * (2 ** max(0, attempt - 1)))
    frac = random.uniform(-0.25, 0.25) if jitter is None else jitter
    return max(0.0, delay * (1.0 + frac))


def _classify(err: BaseException, url: str) -> bool:
    """True if `err` is worth retrying; raises PermanentHTTPError for a dead
    end; re-raises anything that isn't a network/transport failure at all."""
    if isinstance(err, urllib.error.HTTPError):
        if err.code in RETRYABLE_STATUS:
            return True
        raise PermanentHTTPError(err.code, url) from err
    if isinstance(err, (urllib.error.URLError, socket.timeout, TimeoutError,
                        ConnectionError, json.JSONDecodeError, UnicodeDecodeError)):
        return True
    raise err   # not a transport failure -- a bug, and it must stay visible


def fetch_json(url: str, *, user_agent: str, timeout: float = 15,
               max_attempts: int = DEFAULT_MAX_ATTEMPTS,
               base_delay: float = DEFAULT_BASE_DELAY,
               max_delay: float = DEFAULT_MAX_DELAY,
               sleep=time.sleep, opener=None,
               before_request=None, on_retry=None):
    """GET `url`, parse JSON, retrying transient failures with backoff.

    `before_request` (if given) is called before **every** attempt, retries
    included -- that is where a caller's rate-limit throttle belongs, so a
    failure can never skip pacing. `on_retry(attempt, delay, err)` is a logging
    hook. `sleep` and `opener` are injectable for tests.

    Raises `PermanentHTTPError` for a non-retryable 4xx, `RetriesExhausted` when
    every attempt failed, and passes any non-transport exception straight
    through."""
    open_url = opener or urllib.request.urlopen
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        if before_request is not None:
            before_request()
        try:
            req = urllib.request.Request(url, headers={"User-Agent": user_agent})
            with open_url(req, timeout=timeout) as resp:
                return json.load(resp)
        except Exception as err:      # noqa: BLE001 -- _classify re-raises non-transport
            _classify(err, url)       # raises PermanentHTTPError / non-transport bugs
            last_error = err
            if attempt >= max_attempts:
                break
            after = (retry_after_seconds(err)
                     if isinstance(err, urllib.error.HTTPError) else None)
            delay = backoff_delay(attempt, base=base_delay, cap=max_delay,
                                  retry_after=after)
            if on_retry is not None:
                on_retry(attempt, delay, err)
            sleep(delay)
    raise RetriesExhausted(max_attempts, last_error or RuntimeError("unknown"))
