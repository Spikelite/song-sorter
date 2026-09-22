"""Tests for track_inspect: metadata extraction from CDG/MP3/ZIP tracks.

Uses real synthesized MPEG frames and real zip archives (see conftest), so
mutagen's parser and the zip handling are exercised for real rather than mocked.
"""

from __future__ import annotations

import zipfile
import zlib
from pathlib import Path

import pytest

from conftest import mp3_bytes, write_mp3, write_zip

ti = pytest.importorskip("track_inspect")

CDG = b"\x09" * 96 * 4   # arbitrary CDG packets; only size/CRC are read


# --- small pure helpers ----------------------------------------------------

def test_compute_hash_is_sha256_hex() -> None:
    import hashlib
    assert ti._compute_hash(b"abc") == hashlib.sha256(b"abc").hexdigest()


def test_mp3_info_reads_real_frames() -> None:
    info = ti._mp3_info(mp3_bytes(60))
    assert info["bitrate_bps"] == "128000"
    assert info["sample_rate_hz"] == "44100"
    assert info["channels"] == "2"
    assert float(info["length_seconds"]) > 1.0


def test_mp3_info_on_garbage_is_empty() -> None:
    assert ti._mp3_info(b"not an mp3 at all") == {}


def test_mp3_tags_without_id3_header_are_all_not_found() -> None:
    out = ti._mp3_tags(mp3_bytes(40))
    assert set(out) == set(ti._TAG_FIELDS)
    assert all(v == ti.TAG_NOT_FOUND for v in out.values())


def test_mp3_tags_on_unparseable_bytes_keep_schema() -> None:
    out = ti._mp3_tags(b"\x00" * 10)
    assert set(out) == set(ti._TAG_FIELDS)
    assert all(v == ti.TAG_NOT_FOUND for v in out.values())


def test_mp3_tags_reads_values_and_tkey(tmp_path) -> None:
    p = write_mp3(tmp_path / "t.mp3", artist="Tom Jones", title="Delilah",
                  album="Greatest", date="1968", genre="Pop", initialkey="Am")
    out = ti._mp3_tags(p.read_bytes())
    assert out["tag_artist"] == "Tom Jones"
    assert out["tag_title"] == "Delilah"
    assert out["tag_album"] == "Greatest"
    assert out["tag_year"] == "1968"
    assert out["tag_genre"] == "Pop"
    assert out["tag_key"] == "Am"


def test_mp3_tags_blank_value_stays_not_found(tmp_path) -> None:
    p = write_mp3(tmp_path / "t.mp3", artist="   ", title="Real")
    out = ti._mp3_tags(p.read_bytes())
    assert out["tag_artist"] == ti.TAG_NOT_FOUND     # whitespace-only is absent
    assert out["tag_title"] == "Real"


def test_mp3_tags_parsed_file_with_no_tag_block(monkeypatch) -> None:
    """A parsed file whose .tags is None keeps the full NOT_FOUND schema."""
    class Stub:
        tags = None
    monkeypatch.setattr(ti, "MP3", lambda *a, **k: Stub())
    out = ti._mp3_tags(b"x")
    assert all(v == ti.TAG_NOT_FOUND for v in out.values())


# --- _details_from_pair ----------------------------------------------------

def test_details_from_pair_computes_crc_from_loose_cdg_bytes() -> None:
    d = ti._details_from_pair(mp3_bytes(40), cdg_data=CDG)
    assert d["cdg_hash"] == format(zlib.crc32(CDG), "08x")
    assert d["cdg_size"] == str(len(CDG))
    assert d["mp3_size"] == str(len(mp3_bytes(40)))
    assert "mp3_crc_failed" not in d            # intact read gains no marker


def test_details_from_pair_uses_supplied_crc_and_sizes() -> None:
    d = ti._details_from_pair(mp3_bytes(40), cdg_size=123, cdg_crc=0xDEADBEEF,
                              mp3_size=999)
    assert d["cdg_hash"] == "deadbeef"
    assert d["cdg_size"] == "123"
    assert d["mp3_size"] == "999"


def test_details_from_pair_without_any_cdg() -> None:
    d = ti._details_from_pair(mp3_bytes(40))
    assert d["cdg_hash"] == "" and d["cdg_size"] == "0"


def test_details_from_pair_flags_unverified_mp3() -> None:
    """#23: the salvage path must be visible in the metadata."""
    d = ti._details_from_pair(mp3_bytes(40), mp3_crc_ok=False)
    assert d["mp3_crc_failed"] == "1"


# --- track_details: loose files --------------------------------------------

def test_track_details_missing_path_is_empty(tmp_path) -> None:
    assert ti.track_details(tmp_path / "nope.cdg") == {}


def test_track_details_from_cdg_finds_paired_mp3(tmp_path) -> None:
    (tmp_path / "A - S.cdg").write_bytes(CDG)
    write_mp3(tmp_path / "A - S.mp3")
    d = ti.track_details(tmp_path / "A - S.cdg")
    assert d["cdg_hash"] == format(zlib.crc32(CDG), "08x")
    assert d["bitrate_bps"] == "128000"


def test_track_details_from_cdg_without_mp3_is_empty(tmp_path) -> None:
    (tmp_path / "A - S.cdg").write_bytes(CDG)
    assert ti.track_details(tmp_path / "A - S.cdg") == {}


def test_track_details_from_mp3_finds_paired_cdg(tmp_path) -> None:
    (tmp_path / "A - S.cdg").write_bytes(CDG)
    write_mp3(tmp_path / "A - S.mp3")
    d = ti.track_details(tmp_path / "A - S.mp3")
    assert d["cdg_size"] == str(len(CDG))


def test_track_details_from_mp3_without_cdg_is_empty(tmp_path) -> None:
    write_mp3(tmp_path / "A - S.mp3")
    assert ti.track_details(tmp_path / "A - S.mp3") == {}


def test_track_details_unknown_suffix_is_empty(tmp_path) -> None:
    (tmp_path / "x.txt").write_text("hi")
    assert ti.track_details(tmp_path / "x.txt") == {}


def test_loose_and_zipped_cdg_fingerprint_identically(tmp_path) -> None:
    """The documented promise: a CDG hashes the same whether zipped or loose."""
    (tmp_path / "A - S.cdg").write_bytes(CDG)
    write_mp3(tmp_path / "A - S.mp3")
    loose = ti.track_details(tmp_path / "A - S.cdg")
    z = write_zip(tmp_path / "B - S.zip",
                  {"B - S.cdg": CDG, "B - S.mp3": mp3_bytes(60)})
    zipped = ti.track_details(z)
    assert loose["cdg_hash"] == zipped["cdg_hash"]


# --- track_details: zips ---------------------------------------------------

def test_track_details_zip_reads_members(tmp_path) -> None:
    z = write_zip(tmp_path / "A - S.zip",
                  {"sub/": b"", "A - S.CDG": CDG, "A - S.MP3": mp3_bytes(60)})
    d = ti.track_details(z)
    assert d["cdg_hash"] == format(zlib.crc32(CDG), "08x")
    assert d["bitrate_bps"] == "128000"
    assert "mp3_crc_failed" not in d


def test_track_details_zip_missing_a_member_is_empty(tmp_path) -> None:
    z = write_zip(tmp_path / "A - S.zip", {"A - S.cdg": CDG})
    assert ti.track_details(z) == {}


def _corrupt_member(zip_path: Path, marker: bytes) -> None:
    """Flip one byte of a stored member so its CRC-32 no longer matches."""
    raw = bytearray(zip_path.read_bytes())
    at = raw.index(marker)
    raw[at] ^= 0xFF
    zip_path.write_bytes(bytes(raw))


def test_crc_failed_mp3_is_salvaged_and_flagged(tmp_path) -> None:
    """#23 end to end: a genuinely corrupt stored MP3 is kept (salvage path)
    AND flagged, instead of looking identical to an intact copy."""
    marker = b"CRC-MARKER-7f3a"
    z = write_zip(tmp_path / "A - S.zip",
                  {"A - S.cdg": CDG, "A - S.mp3": mp3_bytes(60, marker)})
    _corrupt_member(z, marker)
    with zipfile.ZipFile(z) as zf:                       # really is CRC-bad
        with pytest.raises(zipfile.BadZipFile):
            zf.read("A - S.mp3")
    d = ti.track_details(z)
    assert d["mp3_crc_failed"] == "1"
    assert d["bitrate_bps"] == "128000"                  # still usable


def test_read_member_reports_crc_ok_for_intact_member(tmp_path) -> None:
    z = write_zip(tmp_path / "a.zip", {"a.mp3": b"hello"})
    with zipfile.ZipFile(z) as zf:
        assert ti._read_member(zf, "a.mp3") == (b"hello", True)


@pytest.mark.parametrize("exc, prefix", [
    (NotImplementedError("method 99"), "unsupported compression"),
    (zipfile.BadZipFile("bad dir"), "bad zip"),
    (zlib.error("Error -3"), "decompress failed"),
    (RuntimeError("surprise"), "surprise"),
])
def test_track_details_zip_errors_become_error_entries(tmp_path, monkeypatch,
                                                       exc, prefix) -> None:
    z = write_zip(tmp_path / "A - S.zip", {"A - S.cdg": CDG, "A - S.mp3": mp3_bytes()})

    def boom(*a, **k):
        raise exc
    monkeypatch.setattr(ti.zipfile, "ZipFile", boom)
    out = ti.track_details(z)
    assert set(out) == {"error"}
    assert out["error"].startswith(prefix)


# --- _zip_mp3_member -------------------------------------------------------

def test_zip_mp3_member_skips_directories_and_finds_mp3(tmp_path) -> None:
    z = write_zip(tmp_path / "a.zip",
                  {"x.mp3/": b"", "a.cdg": CDG, "d/Song.MP3": b"m"})
    with zipfile.ZipFile(z) as zf:
        assert ti._zip_mp3_member(zf) == "d/Song.MP3"


def test_zip_mp3_member_none_when_absent(tmp_path) -> None:
    z = write_zip(tmp_path / "a.zip", {"a.cdg": CDG})
    with zipfile.ZipFile(z) as zf:
        assert ti._zip_mp3_member(zf) is None


# --- audio_file ------------------------------------------------------------

def test_audio_file_loose_mp3(tmp_path) -> None:
    mp3 = write_mp3(tmp_path / "A - S.mp3")
    with ti.audio_file(str(tmp_path / "A - S.cdg"), ["mp3", "cdg"]) as p:
        assert p == mp3


def test_audio_file_loose_mp3_missing(tmp_path) -> None:
    with ti.audio_file(str(tmp_path / "gone.cdg"), ["mp3"]) as p:
        assert p is None


def test_audio_file_zip_extracts_and_cleans_up(tmp_path) -> None:
    data = mp3_bytes(30)
    z = write_zip(tmp_path / "A - S.zip", {"A - S.cdg": CDG, "A - S.mp3": data})
    with ti.audio_file(str(z), ["zip"]) as p:
        assert p is not None and p.exists()
        assert p.read_bytes() == data
        extracted = p
    assert not extracted.exists()                        # temp file removed


def test_audio_file_zip_without_mp3_yields_none(tmp_path) -> None:
    z = write_zip(tmp_path / "A - S.zip", {"A - S.cdg": CDG})
    with ti.audio_file(str(z), ["zip"]) as p:
        assert p is None


def test_audio_file_unreadable_zip_yields_none(tmp_path) -> None:
    (tmp_path / "bad.zip").write_bytes(b"this is not a zip")
    with ti.audio_file(str(tmp_path / "bad.zip"), ["zip"]) as p:
        assert p is None


def test_audio_file_no_playable_type(tmp_path) -> None:
    with ti.audio_file(str(tmp_path / "x.cdg"), ["cdg"]) as p:
        assert p is None


def test_audio_file_caller_error_propagates_and_temp_is_removed(tmp_path) -> None:
    """A caller exception must surface untouched (no second yield) and the
    extracted temp file must still be cleaned up."""
    z = write_zip(tmp_path / "A - S.zip", {"A - S.cdg": CDG, "A - S.mp3": mp3_bytes()})
    seen = {}
    with pytest.raises(ValueError, match="caller"):
        with ti.audio_file(str(z), ["zip"]) as p:
            seen["p"] = p
            raise ValueError("caller")
    assert not seen["p"].exists()


def test_audio_file_temp_cleanup_tolerates_already_deleted(tmp_path) -> None:
    z = write_zip(tmp_path / "A - S.zip", {"A - S.cdg": CDG, "A - S.mp3": mp3_bytes()})
    with ti.audio_file(str(z), ["zip"]) as p:
        p.unlink()                                       # gone before cleanup
    # reaching here without an exception is the assertion


# --- read_key_tag ----------------------------------------------------------

def test_read_key_tag_present(tmp_path) -> None:
    assert ti.read_key_tag(write_mp3(tmp_path / "t.mp3", initialkey="F#m")) == "F#m"


def test_read_key_tag_absent(tmp_path) -> None:
    assert ti.read_key_tag(write_mp3(tmp_path / "t.mp3", artist="x")) is None


def test_read_key_tag_no_id3_header(tmp_path) -> None:
    assert ti.read_key_tag(write_mp3(tmp_path / "t.mp3")) is None


def test_read_key_tag_unreadable_file(tmp_path) -> None:
    (tmp_path / "t.mp3").write_bytes(b"\x00" * 8)
    assert ti.read_key_tag(tmp_path / "t.mp3") is None


def test_read_key_tag_blank_value(tmp_path) -> None:
    assert ti.read_key_tag(write_mp3(tmp_path / "t.mp3", initialkey="  ")) is None
