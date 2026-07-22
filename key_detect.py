"""Musical-key detection for karaoke tracks.

Song-sorter's half of KriticalDJ #15: derive an optional musical `key` (plus a
confidence and a provenance tag) per song so an external player can offer a
pre-song pitch reference. CDG carries no pitch data and `index.json` has no key,
so the analysis is done here, offline, and emitted into the library index.

This module is split into two layers:

* A **pure core** (this is the part that carries the correctness weight):
  `pcp_to_key` correlates a 12-bin pitch-class profile against the 24 rotated
  Krumhansl-Schmuckler key templates; `normalize_key` parses whatever a TKEY
  tag or an online source hands us into a canonical `"A minor"` form;
  `combine_key_signals` fuses the manual / tag / offline / online signals into a
  single answer with an honest confidence. All stdlib-only, so it is unit-tested
  without the heavy audio stack.

* An **optional IO layer** (`detect_key_offline`) that decodes the MP3 with
  librosa to produce the pitch-class profile. The import is guarded: if librosa
  (and an MP3 decode backend) isn't installed, `HAVE_LIBROSA` is False and the
  caller falls back to tags / online corroboration. Online lookup lives in
  `key_online.py`.

Canonical key spelling uses sharps for the five black keys
(`C#, D#, F#, G#, A#`); an external player is free to re-spell (e.g. `Bb`).
"""

from __future__ import annotations

import math
import re

# --- optional heavy dependency, guarded ------------------------------------
try:
    import librosa  # noqa: F401  (numpy comes with it)
    import numpy as _np
    HAVE_LIBROSA = True
except Exception:  # pragma: no cover - exercised only where librosa is absent
    HAVE_LIBROSA = False


# Twelve pitch classes, index 0 == C, matching librosa's chroma bin order.
KEY_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Krumhansl-Kessler (1982) key profiles, tonic-relative (index 0 == tonic).
_KS_MAJOR = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
_KS_MINOR = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]

# Confidence is the margin between the best and second-best key correlation,
# scaled so a margin of this size (or more) reads as fully confident. The
# second-best key is usually the relative or parallel key, whose tonic differs
# from the winner's -- exactly the distinction a reference *tone* depends on --
# so a small margin genuinely means "don't trust the tonic", which is the
# behaviour we want ("a wrong key is worse than none").
MARGIN_FULL = 0.12

# song-sorter's emit floor. Auto/online results below this are never written to
# the index (obvious noise). Manual and tag sources always emit. An external
# player is expected to apply its own, stricter, display gate on top using the
# emitted `key_confidence`.
EMIT_FLOOR = 0.50


def _pearson(x: list[float], y: list[float]) -> float:
    """Pearson correlation of two equal-length vectors; 0.0 if either is flat."""
    n = len(x)
    mx = sum(x) / n
    my = sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def pcp_to_key(pcp: list[float]) -> dict:
    """Estimate the key of a 12-bin pitch-class profile (index 0 == C).

    Returns ``{"key", "confidence", "correlations", "runner_up"}`` where ``key``
    is a canonical ``"<Tonic> <major|minor>"`` string, ``confidence`` is in
    [0, 1], and the extras expose the underlying correlations for tests/tuning.
    A flat or empty profile yields confidence 0.0 (and an arbitrary key that the
    confidence gate will discard)."""
    if len(pcp) != 12:
        raise ValueError(f"pitch-class profile must have 12 bins, got {len(pcp)}")

    scored: list[tuple[float, str]] = []
    for tonic in range(12):
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            rotated = [profile[(i - tonic) % 12] for i in range(12)]
            r = _pearson(pcp, rotated)
            scored.append((r, f"{KEY_NAMES[tonic]} {mode}"))

    scored.sort(key=lambda kv: kv[0], reverse=True)
    best_r, best_key = scored[0]
    second_r, second_key = scored[1]
    margin = best_r - second_r
    confidence = max(0.0, min(1.0, margin / MARGIN_FULL))
    return {
        "key": best_key,
        "confidence": round(confidence, 3),
        "correlations": round(best_r, 4),
        "runner_up": second_key,
    }


# --- key-string parsing -----------------------------------------------------

_NOTE_BASE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_CAMELOT_MINOR = {  # Camelot number -> tonic pitch class, minor ('A') ring
    1: 8, 2: 3, 3: 10, 4: 5, 5: 0, 6: 7, 7: 2, 8: 9, 9: 4, 10: 11, 11: 6, 12: 1,
}
_CAMELOT_MAJOR = {  # Camelot number -> tonic pitch class, major ('B') ring
    1: 11, 2: 6, 3: 1, 4: 8, 5: 3, 6: 10, 7: 5, 8: 0, 9: 7, 10: 2, 11: 9, 12: 4,
}
_UNKNOWN_KEY_TOKENS = {"", "o", "off", "none", "n/a", "na", "-", "unknown", "?"}


def normalize_key(raw: str | None) -> str | None:
    """Parse a key string into canonical ``"<Tonic> <major|minor>"``.

    Accepts the shapes that turn up in TKEY tags and online sources:
    ``"Am"``, ``"A minor"``, ``"Amin"``, ``"F#m"``, ``"Gbm"`` (flats),
    ``"C"``/``"Cmaj"``/``"CM"`` (major), ``"Bb"``, and Camelot codes like
    ``"8A"``/``"8B"``. Enharmonic flats fold onto the sharp spelling. Returns
    None for empty, atonal, or unparseable values."""
    if not raw:
        return None
    s = raw.strip()
    if s.lower() in _UNKNOWN_KEY_TOKENS:
        return None

    # Camelot / Open-Key: 1-12 followed by A (minor) or B (major).
    m = re.fullmatch(r"(\d{1,2})\s*([ABab])", s)
    if m:
        num = int(m.group(1))
        if 1 <= num <= 12:
            if m.group(2) in "Aa":
                return f"{KEY_NAMES[_CAMELOT_MINOR[num]]} minor"
            return f"{KEY_NAMES[_CAMELOT_MAJOR[num]]} major"
        return None

    m = re.fullmatch(r"([A-Ga-g])\s*([#b♯♭]?)\s*(.*)", s)
    if not m:
        return None
    pc = _NOTE_BASE[m.group(1).upper()]
    acc = m.group(2)
    if acc in ("#", "♯"):
        pc = (pc + 1) % 12
    elif acc in ("b", "♭"):
        pc = (pc - 1) % 12

    mode_raw = m.group(3).strip()
    # Preserve case just long enough to tell "CM" (major) from "Cm" (minor).
    if mode_raw in ("m", "min"):
        mode = "minor"
    elif mode_raw in ("M", "maj", ""):
        mode = "major"
    else:
        low = mode_raw.lower()
        if low.startswith("min") or low == "m":
            mode = "minor"
        elif low.startswith("maj") or low == "major":
            mode = "major"
        else:
            return None
    return f"{KEY_NAMES[pc]} {mode}"


def to_camelot(key: str | None) -> str | None:
    """Canonical key -> Camelot code (e.g. ``"A minor"`` -> ``"8A"``); None if
    the key can't be parsed. Handy for an external DJ tool's harmonic mixing."""
    norm = normalize_key(key)
    if norm is None:
        return None
    name, mode = norm.rsplit(" ", 1)
    pc = KEY_NAMES.index(name)
    table = _CAMELOT_MINOR if mode == "minor" else _CAMELOT_MAJOR
    letter = "A" if mode == "minor" else "B"
    for num, tonic in table.items():
        if tonic == pc:
            return f"{num}{letter}"
    return None


def keys_agree(a: str | None, b: str | None) -> bool:
    """True when two keys name the same tonic *and* mode. Deliberately strict:
    the relative/parallel/fifth neighbours a detector confuses share notes but
    have a different tonic, so they must not count as agreement for a tone."""
    na, nb = normalize_key(a), normalize_key(b)
    return na is not None and na == nb


# --- signal fusion ----------------------------------------------------------

def combine_key_signals(
    *,
    override: str | None = None,
    tag: str | None = None,
    offline: tuple[str, float] | None = None,
    online: str | None = None,
) -> dict:
    """Fuse the available key signals into one answer.

    Precedence: manual override > TKEY tag > offline audio detection > online.
    Online is *advisory only* -- it reports the original master's key, which can
    differ from a transposed karaoke rip -- so it only boosts confidence when it
    agrees with the offline tonic, or fills in when there is no local signal at
    all (clearly flagged, never overriding a confident local read).

    Returns ``{"key", "confidence", "source", "detail"}``; ``key`` is None and
    ``source`` is ``"none"`` when nothing is known.
    """
    override = normalize_key(override)
    if override is not None:
        return {"key": override, "confidence": 1.0, "source": "manual",
                "detail": ""}

    tag = normalize_key(tag)
    online = normalize_key(online)
    off_key = off_conf = None
    if offline is not None:
        off_key, off_conf = normalize_key(offline[0]), float(offline[1])

    if tag is not None:
        # The tag describes this actual file and was set deliberately; trust it,
        # and lift it when the audio (or online) independently agrees.
        conf = 0.90
        detail = "id3 TKEY"
        if keys_agree(tag, off_key) or keys_agree(tag, online):
            conf = 0.97
            detail = "id3 TKEY (corroborated)"
        return {"key": tag, "confidence": conf, "source": "tag", "detail": detail}

    if off_key is not None:
        conf = off_conf
        detail = ""
        if online is not None:
            if keys_agree(off_key, online):
                conf = min(1.0, off_conf + 0.20)
                detail = "offline+online agree"
            else:
                detail = f"online disagrees ({online})"
        return {"key": off_key, "confidence": round(conf, 3), "source": "auto",
                "detail": detail}

    if online is not None:
        # No local signal (librosa missing / decode failed). Original-master key
        # only -- honest, modest confidence, flagged so nobody mistakes it for a
        # verified read of this rip.
        return {"key": online, "confidence": 0.50, "source": "online",
                "detail": "original master key; unverified vs rip"}

    return {"key": None, "confidence": 0.0, "source": "none", "detail": ""}


def should_emit(source: str, confidence: float) -> bool:
    """Whether a fused result is solid enough to write into the library index.
    Manual/tag always emit; auto/online must clear the emit floor."""
    if source in ("manual", "tag"):
        return True
    if source in ("auto", "online"):
        return confidence >= EMIT_FLOOR
    return False


# --- optional offline audio detection --------------------------------------

# Analyse the first verse only: skip a lead-in, take a window long enough to be
# harmonically stable but short enough to miss the final-chorus key change that
# trips whole-track estimates.
_OFFSET_SECONDS = 10.0
_WINDOW_SECONDS = 45.0


def detect_key_offline(audio_path: str) -> tuple[str, float] | None:
    """Estimate (key, confidence) from a decodable audio file, or None.

    Returns None when librosa isn't installed or the file can't be decoded, so
    callers degrade gracefully to tag/online signals. Requires an MP3 decode
    backend (modern soundfile/libsndfile, or audioread + ffmpeg on PATH)."""
    if not HAVE_LIBROSA:
        return None
    try:
        y, sr = librosa.load(audio_path, mono=True, offset=_OFFSET_SECONDS,
                             duration=_WINDOW_SECONDS)
        if y is None or len(y) < sr:  # under a second decoded -> nothing usable
            # Retry from the very start in case the track is shorter than the offset.
            y, sr = librosa.load(audio_path, mono=True, duration=_WINDOW_SECONDS)
        if y is None or len(y) < sr:
            return None
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        pcp = [float(v) for v in _np.mean(chroma, axis=1)]
    except Exception:
        return None
    result = pcp_to_key(pcp)
    return result["key"], result["confidence"]
