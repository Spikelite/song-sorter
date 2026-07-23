"""Online key corroboration: MusicBrainz -> AcousticBrainz.

The *advisory* half of key detection (KriticalDJ #15). It resolves a recording
via MusicBrainz text search -- the same service song-sorter's Musicbrainz step
already uses, so no audio fingerprinting, no AcoustID, no `fpcalc` binary, and no
API key -- then reads that recording's estimated key from AcousticBrainz.

Important caveat, enforced by how `combine_key_signals` uses this: AcousticBrainz
reports the key of the *original commercial master*, not the (often transposed)
karaoke rip. So this is only ever used to corroborate/boost the offline read, or
to fill where there is no local signal at all -- never to override a confident
local result.

Everything here is best-effort and offline-safe: no internet, or a recording MB
can't match, all yield None rather than raising. AcousticBrainz is archived/
read-only but still serves data for recordings that were submitted, with spotty
coverage -- hence we try every candidate MBID until one has a key. It's plain
HTTP/JSON (urllib), so this module has no third-party dependencies.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from key_detect import normalize_key

_MB_URL = "https://musicbrainz.org/ws/2/recording"
_AB_URL = "https://acousticbrainz.org/api/v1/{}/low-level"
_USER_AGENT = "song-sorter/1.0 ( https://github.com/Spikelite/song-sorter )"

# Throttled per service, not globally: MusicBrainz asks for <= 1 request/second,
# but AcousticBrainz is a separate host with no such limit. Sharing one 1.1s
# throttle made a lookup that tried several candidate MBIDs cost seconds of
# needless waiting -- the dominant per-track cost on a large library.
_MB_MIN_INTERVAL = 1.1
_AB_MIN_INTERVAL = 0.34
_throttle_lock = threading.Lock()
_last_call: dict[str, float] = {}

# How confident MB must be in a text match before we trust the MBID (its search
# score is 0-100), and how many candidates to try against AcousticBrainz, whose
# coverage is per-MBID (many recordings 404).
_MB_MIN_SCORE = 85
_MAX_CANDIDATES = 5


def _throttle(service: str, min_interval: float) -> None:
    """Rate-limit calls to one service without penalising the other."""
    with _throttle_lock:
        wait = min_interval - (time.monotonic() - _last_call.get(service, 0.0))
        if wait > 0:
            time.sleep(wait)
        _last_call[service] = time.monotonic()


def _mb_escape(s: str) -> str:
    """Neutralise Lucene quoting -- we wrap the value in a quoted phrase."""
    return s.replace("\\", " ").replace('"', " ").strip()


def _musicbrainz_mbids(artist: str, title: str) -> list[str]:
    """Candidate recording MBIDs for an artist+title, best match first; [] on
    any failure or when nothing clears the match-score threshold."""
    qa, qt = _mb_escape(artist), _mb_escape(title)
    if not qa or not qt:
        return []
    q = f'artist:"{qa}" AND recording:"{qt}"'
    url = _MB_URL + "?" + urllib.parse.urlencode(
        {"query": q, "fmt": "json", "limit": "10"})
    _throttle("musicbrainz", _MB_MIN_INTERVAL)
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            recordings = json.load(resp).get("recordings", [])
    except Exception:
        return []
    mbids: list[str] = []
    for rec in recordings:
        if rec.get("score", 0) < _MB_MIN_SCORE:
            continue  # results are score-ordered, so the rest are weaker too
        mbid = rec.get("id")
        if mbid and mbid not in mbids:
            mbids.append(mbid)
        if len(mbids) >= _MAX_CANDIDATES:
            break
    return mbids


def _acousticbrainz_key(mbid: str) -> str | None:
    """AcousticBrainz's estimated key for a recording MBID, or None if it holds
    no data (404) / the request fails."""
    _throttle("acousticbrainz", _AB_MIN_INTERVAL)
    req = urllib.request.Request(_AB_URL.format(mbid),
                                 headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        return None if e.code == 404 else None
    except Exception:
        return None
    tonal = data.get("tonal", {})
    key = tonal.get("key_key")
    scale = tonal.get("key_scale")
    if not key or not scale:
        return None
    return normalize_key(f"{key} {scale}")


def lookup_online(artist: str, title: str,
                  mbid_cache: dict[str, str | None] | None = None) -> tuple[str | None, str]:
    """Best-effort online key for a recording named by artist+title.

    Returns ``(canonical_key_or_None, detail)``. Resolves candidate MBIDs via
    MusicBrainz text search, then tries each against AcousticBrainz until one
    yields a key. ``mbid_cache`` (optional) memoises AcousticBrainz results
    across the run."""
    mbids = _musicbrainz_mbids(artist, title)
    if not mbids:
        return None, "no musicbrainz match"
    for mbid in mbids:
        if mbid_cache is not None and mbid in mbid_cache:
            key = mbid_cache[mbid]
        else:
            key = _acousticbrainz_key(mbid)
            if mbid_cache is not None:
                mbid_cache[mbid] = key
        if key:
            return key, f"acousticbrainz {mbid}"
    return None, "no acousticbrainz key"
