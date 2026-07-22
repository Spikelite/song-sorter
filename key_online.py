"""Online key corroboration: fingerprint -> AcoustID -> AcousticBrainz.

The *advisory* half of key detection (KriticalDJ #15, v2). It fingerprints the
audio with Chromaprint (`fpcalc`), resolves the recording's MusicBrainz IDs via
AcoustID, then asks AcousticBrainz for that recording's estimated key.

Important caveat, enforced by how `combine_key_signals` uses this: AcousticBrainz
reports the key of the *original commercial master*, not the (often transposed)
karaoke rip. So this is only ever used to corroborate/boost the offline read, or
to fill where there is no local signal at all -- never to override a confident
local result.

Everything here is best-effort and offline-safe: a missing `fpcalc`/`pyacoustid`,
no API key, or no internet all yield None rather than raising. AcoustID (which
holds no key data itself) and AcousticBrainz are separate services; AcousticBrainz
is archived/read-only but still serves data for recordings that were submitted,
with spotty coverage -- hence we try every candidate MBID until one has a key.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

from key_detect import normalize_key

try:
    import acoustid  # needs the fpcalc/Chromaprint binary on PATH
    HAVE_ACOUSTID = True
except Exception:  # pragma: no cover - exercised only where acoustid is absent
    HAVE_ACOUSTID = False

_AB_URL = "https://acousticbrainz.org/api/v1/{}/low-level"
_USER_AGENT = "song-sorter/1.0 ( https://github.com/Spikelite/song-sorter )"

# Be a good citizen: AcoustID permits a few requests/sec, AcousticBrainz is a
# volunteer archive. One shared throttle keeps us well under any limit.
_MIN_INTERVAL = 0.34
_throttle_lock = threading.Lock()
_last_call = 0.0


def _throttle() -> None:
    global _last_call
    with _throttle_lock:
        wait = _MIN_INTERVAL - (time.monotonic() - _last_call)
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def fingerprint(mp3_path: str) -> tuple[int, bytes] | None:
    """(duration, chromaprint fingerprint) for an audio file, or None.

    None when pyacoustid/fpcalc isn't available or the file can't be
    fingerprinted."""
    if not HAVE_ACOUSTID:
        return None
    try:
        duration, fp = acoustid.fingerprint_file(mp3_path)
        return int(duration), fp
    except Exception:
        return None


def _acoustid_mbids(api_key: str, duration: int, fp: bytes) -> list[str]:
    """Recording MBIDs for a fingerprint, best score first; [] on any failure."""
    _throttle()
    try:
        res = acoustid.lookup(api_key, fp, duration, meta="recordingids")
    except Exception:
        return []
    if res.get("status") != "ok":
        return []
    mbids: list[str] = []
    for result in sorted(res.get("results", []),
                         key=lambda r: r.get("score", 0), reverse=True):
        for rec in result.get("recordings", []):
            mbid = rec.get("id")
            if mbid and mbid not in mbids:
                mbids.append(mbid)
    return mbids


def _acousticbrainz_key(mbid: str) -> str | None:
    """AcousticBrainz's estimated key for a recording MBID, or None if it holds
    no data (404) / the request fails."""
    _throttle()
    req = urllib.request.Request(_AB_URL.format(mbid),
                                 headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        return None
    except Exception:
        return None
    tonal = data.get("tonal", {})
    key = tonal.get("key_key")
    scale = tonal.get("key_scale")
    if not key or not scale:
        return None
    return normalize_key(f"{key} {scale}")


def lookup_online(mp3_path: str, api_key: str,
                  mbid_cache: dict[str, str | None] | None = None) -> tuple[str | None, str]:
    """Best-effort online key for an audio file.

    Returns ``(canonical_key_or_None, detail)``. Fingerprints, resolves MBIDs via
    AcoustID, then tries each MBID against AcousticBrainz until one yields a key.
    ``mbid_cache`` (optional) memoises AcousticBrainz results across the run."""
    if not api_key:
        return None, "no acoustid api key"
    fp = fingerprint(mp3_path)
    if fp is None:
        return None, "fingerprint unavailable"
    duration, code = fp
    mbids = _acoustid_mbids(api_key, duration, code)
    if not mbids:
        return None, "no acoustid match"
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
