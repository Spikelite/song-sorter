"""Extract details from karaoke tracks (CDG, MP3, ZIP)."""

import hashlib
from sys import exc_info
import zipfile
from io import BytesIO
from pathlib import Path

from mutagen.mp3 import MP3


def _compute_hash(data: bytes) -> str:
    """Compute SHA-256 hash of data, hex-encoded."""
    return hashlib.sha256(data).hexdigest()


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


def _details_from_pair(
    cdg_data: bytes,
    mp3_data: bytes,
    cdg_size: int | None = None,
    mp3_size: int | None = None,
) -> dict[str, str]:
    """Build details dict from CDG and MP3 data."""
    out: dict[str, str] = {
        "mp3_hash": _compute_hash(mp3_data),
        "mp3_size": str(len(mp3_data) if mp3_size is None else mp3_size),
        "cdg_hash": _compute_hash(cdg_data),
        "cdg_size": str(len(cdg_data) if cdg_size is None else cdg_size),
    }

    mp3_info = _mp3_info(mp3_data)
    out.update(mp3_info)

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
        return _details_from_pair(cdg_data, mp3_data)

    if suffix == ".mp3":
        cdg_path = p.with_suffix(".cdg")
        if not cdg_path.exists():
            return {}
        with open(cdg_path, "rb") as f:
            cdg_data = f.read()
        with open(p, "rb") as f:
            mp3_data = f.read()
        return _details_from_pair(cdg_data, mp3_data)

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
                mp3_data = zf.read(mp3_member)
                cdg_data = zf.read(cdg_member)
                # cdg_size = zf.getinfo(cdg_member).file_size
                # mp3_size = zf.getinfo(mp3_member).file_size
                return _details_from_pair(
                    cdg_data, mp3_data,
                    # cdg_size=cdg_size, mp3_size=mp3_size
                )
        except Exception as e:
            print(f"\nfailed to read {p} :: {e}")
            return { "error": str(e)}

    return {}