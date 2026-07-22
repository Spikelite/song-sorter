"""Extract details from karaoke tracks (CDG, MP3, ZIP)."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import contextmanager
from sys import exc_info
import zipfile_deflate64 as zipfile   # adds Deflate64 (method 9); supersets stdlib zipfile
import zlib
from io import BytesIO
from pathlib import Path
from tqdm import tqdm

from mutagen.mp3 import MP3
from mutagen.easyid3 import EasyID3

# TKEY (initial musical key) isn't in EasyID3's default whitelist; register it so
# it can be read alongside the other tags. Harmless if a mutagen version already
# knows it.
try:
    EasyID3.RegisterTextKey("initialkey", "TKEY")
except Exception:  # pragma: no cover - depends on mutagen version internals
    pass


def _compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data, hex-encoded."""
    return hashlib.sha256(data).hexdigest()

def _read_member(zf, name):
    """Read a zip member, returning (data, crc_ok).

    On a CRC-32 mismatch the bytes still decompressed fine, so we re-read
    with the checksum disabled and flag the result as unverified rather
    than dropping the track."""
    try:
        return zf.read(name), True
    except zipfile.BadZipFile:
        f = zf.open(name)
        f._expected_crc = None   # private CPython attr; disables end-of-stream CRC check
        return f.read(), False

def _mp3_info(data: bytes) -> dict[str, str]:
    """Extract length and quality from MP3 bytes. Returns empty dict on failure."""
    try:
        audio = MP3(BytesIO(data))
        return {
            "length_seconds": str(round(audio.info.length, 2)),
            "bitrate_bps": str(audio.info.bitrate),
            "sample_rate_hz": str(audio.info.sample_rate),
            "channels": str(audio.info.channels),
        }
    except Exception:
        return {}


# Sentinel for a tag we looked for but didn't find. Keeps the metadata
# schema deterministic: every tag_* key is ALWAYS present on every track.
TAG_NOT_FOUND = "<not-found>"

# Our metadata key -> EasyID3 field name.
_TAG_FIELDS = {
    "tag_artist": "artist",
    "tag_title": "title",
    "tag_album": "album",
    "tag_year": "date",
    "tag_genre": "genre",
    "tag_key": "initialkey",   # ID3 TKEY -- musical key, near-never populated but free to read
}


def _mp3_tags(data: bytes) -> dict[str, str]:
    """Read ID3 tags from MP3 bytes.

    Always returns all five tag_* keys. Any tag that is absent — or an MP3
    that can't be parsed / has no ID3 header — yields TAG_NOT_FOUND for that
    key, so the output schema is identical for every track."""
    out = {key: TAG_NOT_FOUND for key in _TAG_FIELDS}
    try:
        audio = MP3(BytesIO(data), ID3=EasyID3)
    except Exception:
        return out  # unreadable MP3 / no ID3 header — all keys stay NOT_FOUND
    if audio.tags is None:
        return out
    for out_key, easy_key in _TAG_FIELDS.items():
        values = audio.tags.get(easy_key)
        if values:
            value = str(values[0]).strip()
            if value:
                out[out_key] = value
    return out


def _details_from_pair(
    mp3_data: bytes,
    *,
    cdg_data: bytes | None = None,
    cdg_size: int | None = None,
    cdg_crc: int | None = None,
    mp3_size: int | None = None,
) -> dict[str, str]:
    """Build details dict from an MP3 plus a CDG fingerprint.

    The CDG is identified by a CRC-32. The zip central directory supplies
    that for free (pass cdg_crc + cdg_size, no decompression needed). For
    loose .cdg files we pass cdg_data and compute the same CRC-32 from the
    bytes, so a given CDG fingerprints identically whether zipped or loose."""
    if cdg_crc is None and cdg_data is not None:
        cdg_crc = zlib.crc32(cdg_data)
    if cdg_size is None and cdg_data is not None:
        cdg_size = len(cdg_data)

    out: dict[str, str] = {
        "mp3_hash": _compute_hash(mp3_data),
        "mp3_size": str(len(mp3_data) if mp3_size is None else mp3_size),
        "cdg_hash": format(cdg_crc, "08x") if cdg_crc is not None else "",
        "cdg_size": str(cdg_size if cdg_size is not None else 0),
    }

    mp3_info = _mp3_info(mp3_data)
    out.update(mp3_info)
    out.update(_mp3_tags(mp3_data))   # always adds the five tag_* keys

    return out


def track_details(path: str | Path) -> dict[str, str]:
    """
    Extract hash, size, length, and quality from a karaoke track.

    - CDG: finds paired MP3 in same directory, extracts details from both.
    - MP3: finds paired CDG in same directory, extracts details from both.
    - ZIP: inspects archive for CDG/MP3 members, extracts same details.

    Returns dict with keys: cdg_hash, cdg_size, mp3_hash, mp3_size,
    length_seconds, bitrate_bps, sample_rate_hz, channels.
    Missing keys indicate unavailable data.
    """
    p = Path(path)
    if not p.exists():
        return {}

    suffix = p.suffix.lower()

    if suffix == ".cdg":
        mp3_path = p.with_suffix(".mp3")
        if not mp3_path.exists():
            return {}
        with open(p, "rb") as f:
            cdg_data = f.read()
        with open(mp3_path, "rb") as f:
            mp3_data = f.read()
        return _details_from_pair(mp3_data, cdg_data=cdg_data)

    if suffix == ".mp3":
        cdg_path = p.with_suffix(".cdg")
        if not cdg_path.exists():
            return {}
        with open(cdg_path, "rb") as f:
            cdg_data = f.read()
        with open(p, "rb") as f:
            mp3_data = f.read()
        return _details_from_pair(mp3_data, cdg_data=cdg_data)

    if suffix == ".zip":
        try:
            with zipfile.ZipFile(p, "r") as zf:
                stems: dict[str, str] = {}
                for n in zf.namelist():
                    if n.endswith("/"):
                        continue
                    name_lower = Path(n).name.lower()
                    if name_lower.endswith(".cdg"):
                        stems["cdg"] = n
                    elif name_lower.endswith(".mp3"):
                        stems["mp3"] = n

                if len(stems) < 2:
                    return {}

                cdg_member = stems["cdg"]
                mp3_member = stems["mp3"]
                mp3_data, mp3_ok = _read_member(zf, mp3_member)
                # CDG: take size + CRC-32 straight from the zip directory.
                # No need to read or decompress the CDG member at all.
                cdg_info = zf.getinfo(cdg_member)
                return _details_from_pair(
                    mp3_data,
                    cdg_size=cdg_info.file_size,
                    cdg_crc=cdg_info.CRC,
                )
        except NotImplementedError as e:
            # Unsupported compression method (something even deflate64 can't handle)
            tqdm.write(f"unsupported compression {p} :: {e}")
            return {"error": f"unsupported compression: {e}"}

        except zipfile.BadZipFile as e:
            # Structural damage: bad central directory / truncated archive.
            # (Per-member CRC failures are salvaged in _read_member, not here.)
            tqdm.write(f"corrupt archive {p} :: {e}")
            return {"error": f"bad zip: {e}"}

        except zlib.error as e:
            # Compressed stream can't be inflated — the "Error -3" family.
            tqdm.write(f"corrupt data {p} :: {e}")
            return {"error": f"decompress failed: {e}"}

        except Exception as e:
            # Anything unforeseen — keep a catch-all so nothing slips through.
            tqdm.write(f"failed to read {p} :: {e}")
            return {"error": str(e)}

    return {}


def _zip_mp3_member(zf) -> str | None:
    """Name of the first .mp3 member in an open zip, or None."""
    for n in zf.namelist():
        if not n.endswith("/") and Path(n).name.lower().endswith(".mp3"):
            return n
    return None


@contextmanager
def audio_file(base_path: str, file_types: list[str]):
    """Yield a real filesystem path to a track's MP3, or None if there isn't one.

    Key detection (librosa decode, fpcalc fingerprint) needs a genuine file
    path, not bytes. A loose ``.mp3`` is handed back directly; an MP3 packed in a
    ``.zip`` is extracted to a temp file that is deleted on exit. ``base_path``
    is the store's path with any extension; ``file_types`` its recorded types.

    Any failure to obtain the audio yields None rather than raising, so callers
    degrade gracefully; a caller's own error still propagates untouched."""
    p = Path(base_path)
    if "mp3" in file_types:
        mp3 = p.with_suffix(".mp3")
        yield mp3 if mp3.exists() else None
        return
    if "zip" not in file_types:
        yield None
        return

    # Extract the zip's MP3 member to a temp file up front (best-effort). All
    # setup errors are resolved BEFORE the single yield, so the yield sits in a
    # bare try/finally -- an exception thrown back in by the caller's `with`
    # body then propagates cleanly instead of triggering a second yield.
    tmp_path = None   # a file that now exists on disk and must be cleaned up
    ok = False
    try:
        data = None
        with zipfile.ZipFile(p.with_suffix(".zip"), "r") as zf:
            member = _zip_mp3_member(zf)
            if member is not None:
                data, _ = _read_member(zf, member)
        if data is not None:
            fd, tmp = tempfile.mkstemp(suffix=".mp3")
            tmp_path = tmp
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            ok = True
    except Exception:
        ok = False
    try:
        yield Path(tmp_path) if ok else None
    finally:
        if tmp_path is not None:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def read_key_tag(mp3_path: str | Path) -> str | None:
    """Read the ID3 TKEY (initial key) from an MP3 path; None if absent/unreadable."""
    try:
        audio = MP3(str(mp3_path), ID3=EasyID3)
    except Exception:
        return None
    if audio.tags is None:
        return None
    values = audio.tags.get("initialkey")
    if values:
        value = str(values[0]).strip()
        return value or None
    return None
