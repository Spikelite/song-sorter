"""Source-level regression guards for modules the test env cannot import.

`main.py` pulls in questionary/rapidfuzz and `track_inspect.py` pulls in
mutagen/zipfile-deflate64 (which needs a C compiler), so neither can be
imported here. `test_docs.py` already reads `main.py` as text for the same
reason; these guards do the same for defects that would otherwise silently
come back.
"""

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MAIN = (ROOT / "main.py").read_text(encoding="utf-8")
INSPECT = (ROOT / "track_inspect.py").read_text(encoding="utf-8")


# --- #22: fuzzy-merge tie-breaks must be deterministic ----------------------

def test_fuzz_does_not_tie_break_on_node_hash() -> None:
    """IndexNode defines no __hash__, so hash(node) is address-derived and
    differs every run: using it to break a count tie let two Fuzz passes merge
    the same pair in opposite directions."""
    offenders = re.findall(r"hash\(\s*(?:anode|snode|[a-z_]*nodes?\[)[^)]*\)", MAIN)
    assert not offenders, f"identity-based tie-break reintroduced: {offenders}"


def test_fuzz_tie_breaks_on_the_node_key() -> None:
    assert "this_count == other_count and artist < other_artist" in MAIN
    assert "this_count == other_count and song < other_song" in MAIN


# --- #23: the CRC-verified flag must not be discarded ----------------------

def test_read_member_crc_flag_is_threaded_not_dropped() -> None:
    """_read_member returns (data, crc_ok) and its docstring promises to flag
    an unverified read. The zip path used to bind that flag and never use it,
    so a corrupt MP3 looked identical to a good one."""
    assert "mp3_data, mp3_crc_ok = _read_member(" in INSPECT
    assert "mp3_crc_ok=mp3_crc_ok" in INSPECT
    assert 'out["mp3_crc_failed"] = "1"' in INSPECT
    # the old discarded-binding form must not return
    assert "mp3_ok" not in INSPECT


def test_intact_tracks_gain_no_crc_marker() -> None:
    """The marker is written only on the salvage path, so healthy tracks keep
    exactly the metadata schema they had."""
    assert "if not mp3_crc_ok:" in INSPECT
