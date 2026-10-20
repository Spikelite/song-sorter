"""main.py: config/alias/override loading, Search, Detail and Refresh."""

from __future__ import annotations

import json
import os

import pytest

from conftest import CDG_BYTES, make_track, mp3_bytes, write_zip


@pytest.fixture
def m(mainmod):
    return mainmod


# --- config / aliases / overrides ------------------------------------------

def test_config_roundtrip_and_missing(m) -> None:
    assert m._load_config() == {}
    m._save_config({"output_path": "/out", "version_limit": 2})
    assert m._load_config() == {"output_path": "/out", "version_limit": 2}


def test_config_unreadable_is_empty(m) -> None:
    m._CONFIG_PATH.parent.mkdir(parents=True)
    m._CONFIG_PATH.write_text("{not json", encoding="utf-8")
    assert m._load_config() == {}


def test_aliases_drop_blank_and_identity_entries(m) -> None:
    m._ARTIST_ALIASES_PATH.parent.mkdir(parents=True)
    m._ARTIST_ALIASES_PATH.write_text(json.dumps({"aliases": {
        "Jones, Tom": "Tom Jones", "Same": "Same", "Blank": ""}}), encoding="utf-8")
    assert m._load_aliases() == {"Jones, Tom": "Tom Jones"}


def test_aliases_missing_or_unreadable(m) -> None:
    assert m._load_aliases() == {}
    m._ARTIST_ALIASES_PATH.parent.mkdir(parents=True)
    m._ARTIST_ALIASES_PATH.write_text("[", encoding="utf-8")
    assert m._load_aliases() == {}


def test_key_overrides_accept_both_shapes_and_normalise_keys(m) -> None:
    m._KEY_OVERRIDES_PATH.parent.mkdir(parents=True)
    m._KEY_OVERRIDES_PATH.write_text(json.dumps({"overrides": {
        "  Adele - Hello ": "Fm", "Blank - X": ""}}), encoding="utf-8")
    assert m._load_key_overrides() == {"adele - hello": "Fm"}
    m._KEY_OVERRIDES_PATH.write_text(json.dumps({"A - B": "8A"}), encoding="utf-8")
    assert m._load_key_overrides() == {"a - b": "8A"}         # bare mapping


def test_key_overrides_missing_or_unreadable(m) -> None:
    assert m._load_key_overrides() == {}
    m._KEY_OVERRIDES_PATH.parent.mkdir(parents=True)
    m._KEY_OVERRIDES_PATH.write_text("nope", encoding="utf-8")
    assert m._load_key_overrides() == {}


def test_default_scan_dir(m, store) -> None:
    assert m._default_scan_dir(store) == "."
    store.add(make_track(path=""))
    assert m._default_scan_dir(store) == "."             # blank paths skipped
    store.add(make_track(path=os.path.join("lib", "SC", "a.zip")))
    assert m._default_scan_dir(store) == os.path.join("lib", "SC")


# --- import_path -----------------------------------------------------------

def test_import_path_valid_dir(m, prompts, tmp_path) -> None:
    prompts.queue(str(tmp_path))
    assert m.import_path(".") == tmp_path


def test_import_path_cancelled(m, prompts) -> None:
    prompts.queue(None)
    assert m.import_path() is None


def test_import_path_rejects_non_directory(m, prompts, tmp_path, monkeypatch, capsys) -> None:
    pauses = []
    monkeypatch.setattr("builtins.input", lambda msg="": pauses.append(msg))
    prompts.queue(str(tmp_path / "missing"))
    assert m.import_path() is None
    assert "Not a valid directory" in capsys.readouterr().out
    assert pauses                                         # waited for Enter


# --- add_tracks (Search) ---------------------------------------------------

def test_add_tracks_parses_types_aliases_and_reversed_folders(m, store, tmp_path) -> None:
    lib = tmp_path / "lib"
    (lib / "song-artist").mkdir(parents=True)
    (lib / "SC-1 - Jones, Tom - Delilah.zip").write_bytes(b"z")
    (lib / "Abba - Waterloo.cdg").write_bytes(b"c")
    (lib / "Abba - Waterloo.mp3").write_bytes(b"m")
    (lib / "Queen - Loner.cdg").write_bytes(b"c")          # no paired mp3
    (lib / "song-artist" / "SC-2 - Hello - Adele.zip").write_bytes(b"z")
    (lib / "notes.txt").write_text("ignored")
    m._ARTIST_ALIASES_PATH.parent.mkdir(parents=True)
    m._ARTIST_ALIASES_PATH.write_text(
        json.dumps({"aliases": {"Jones, Tom": "Tom Jones"}}), encoding="utf-8")

    m.add_tracks(store, lib)
    by_song = {t.song: t for t in store.all()}
    assert len(by_song) == 4
    assert by_song["Delilah"].artist == "Tom Jones"         # alias applied
    assert by_song["Delilah"].file_types == ["zip"]
    assert by_song["Waterloo"].file_types == ["mp3", "cdg"]
    assert by_song["Loner"].file_types == ["cdg"]
    assert by_song["Hello"].artist == "Adele"               # song-artist folder


def test_add_tracks_is_additive_and_keeps_metadata(m, store, tmp_path, capsys) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "SC-1 - Abba - Waterloo.zip").write_bytes(b"z")
    m.add_tracks(store, lib)
    t = store.all()[0]
    t.metadata["mp3_hash"] = "keep-me"
    t.artist = "Curated"
    m.add_tracks(store, lib)
    assert store.all()[0].metadata["mp3_hash"] == "keep-me"
    assert store.all()[0].artist == "Curated"
    assert "skipped 1 already present" in capsys.readouterr().out


def test_add_tracks_blank_artist_becomes_unknown(m, store, tmp_path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "JustATitle.zip").write_bytes(b"z")
    m.add_tracks(store, lib)
    assert store.all()[0].artist == "Unknown"


# --- add_details (Detail) --------------------------------------------------

def _zip_track(store, lib, name):
    z = write_zip(lib / f"{name}.zip", {f"{name}.cdg": CDG_BYTES, f"{name}.mp3": mp3_bytes()})
    store.add(make_track(path=str(z.resolve()), artist="A", song=name))
    return z


def test_add_details_extracts_and_is_incremental(m, store, tmp_path, capsys) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    _zip_track(store, lib, "one")
    _zip_track(store, lib, "two")
    (lib / "stray.zip").write_bytes(b"not tracked")         # not in the store
    m.add_details(store, lib, workers=2)
    for t in store.all():
        assert t.metadata["bitrate_bps"] == "128000"
        assert t.metadata["src_size"] and t.metadata["src_mtime"]
    assert m._CACHE_PATH.exists()                           # final checkpoint
    capsys.readouterr()
    m.add_details(store, lib, workers=2)                    # nothing changed
    assert "Added 0 track(s), skipped 2 unchanged" in capsys.readouterr().out


def test_add_details_checkpoints_periodically(m, store, tmp_path, monkeypatch) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    _zip_track(store, lib, "one")
    saves = []
    real_save = store.save
    monkeypatch.setattr(store, "save", lambda p: (saves.append(p), real_save(p)))
    m.add_details(store, lib, workers=1, checkpoint_seconds=0)
    assert len(saves) >= 2                                  # mid-run + final


def test_add_details_survives_a_failing_extractor(m, store, tmp_path, monkeypatch, capsys) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    bad = _zip_track(store, lib, "bad")
    _zip_track(store, lib, "good")
    real = m.track_details

    def flaky(p):
        if p.name == bad.name:
            raise RuntimeError("disk read error")
        return real(p)
    monkeypatch.setattr(m, "track_details", flaky)
    m.add_details(store, lib, workers=2)
    by = {t.song: t for t in store.all()}
    assert "bitrate_bps" in by["good"].metadata
    assert "bitrate_bps" not in by["bad"].metadata
    out = capsys.readouterr()
    assert "detail failed" in out.out + out.err      # tqdm.write -> stdout


class _Vanishing:
    """A path the walk saw but which is gone by the time it is stat'd."""

    def __init__(self, resolved):
        self._resolved = resolved

    def __lt__(self, other):
        return False

    def is_file(self):
        return True

    def resolve(self):
        return self._resolved

    def stat(self):
        raise FileNotFoundError("vanished")


class _FakeRoot:
    def __init__(self, *paths):
        self._paths = paths

    def rglob(self, pattern):
        return iter(self._paths)


def test_add_details_skips_a_file_that_vanishes_mid_walk(m, store, tmp_path) -> None:
    store.add(make_track(path="/lib/gone.zip"))
    m.add_details(store, _FakeRoot(_Vanishing("/lib/gone.zip")), workers=1)
    assert "src_size" not in store.all()[0].metadata


# --- _format_metadata ------------------------------------------------------

def test_format_metadata_summarises_sizes_and_hashes(m) -> None:
    out = m._format_metadata({"length_seconds": "201.5", "bitrate_bps": "192000",
                              "mp3_size": "4800000", "cdg_size": "9600",
                              "mp3_hash": "abcdef123", "cdg_hash": "0011223344"})
    assert out == "sec 201.5, bitrate 192 m.sz 4800k c.sz 9k m# abcdef c# 001122"


def test_format_metadata_defaults(m) -> None:
    assert m._format_metadata({}) == "sec , bitrate 0 m.sz 0k c.sz 0k m#  c# "


# --- refresh_names (Refresh) -----------------------------------------------

def _known(store, artist, n=6):
    for i in range(n):
        store.add(make_track(path=f"/known/{artist}{i}", artist=artist, song=f"S{i}"))


def test_refresh_resolves_orientation_from_known_artists(m, store, tmp_path, capsys) -> None:
    _known(store, "Abba")
    lib = tmp_path / "lib"
    lib.mkdir()
    fwd = lib / "Abba - Waterloo.zip"
    rev = lib / "Mamma_Mia - Abba.zip"                    # underscore -> space
    amb = lib / "Foo - Bar.zip"                           # neither side known
    for p in (fwd, rev, amb):
        p.write_bytes(b"z")
        store.add(make_track(path=str(p.resolve()), artist="?", song="?"))
    m.refresh_names(store, lib)
    got = {t.path: (t.artist, t.song) for t in store.all()}
    assert got[str(fwd.resolve())] == ("Abba", "Waterloo")
    assert got[str(rev.resolve())] == ("Abba", "Mamma Mia")
    assert got[str(amb.resolve())] == ("?", "?")          # left alone
    assert "Modified 2 tracks" in capsys.readouterr().out


def test_refresh_reports_known_mapping_without_a_track(m, store, tmp_path, capsys) -> None:
    _known(store, "Abba")
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "Abba - Waterloo.zip").write_bytes(b"z")        # not in the store
    m.refresh_names(store, lib)
    assert "Found new mapping, but no track" in capsys.readouterr().out


def test_refresh_recovers_compact_stems_only_for_unknown_tracks(m, store, tmp_path) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    unk = lib / "DIS61201-13-MARY POPPINS-I LOVE TO LAUGH.cdg"
    cur = lib / "DIS61201-14-MARY POPPINS-FEED THE BIRDS.cdg"
    for p in (unk, cur):
        p.write_bytes(b"c")
    store.add(make_track(path=str(unk.resolve()), artist="Unknown", song="whole stem"))
    store.add(make_track(path=str(cur.resolve()), artist="Curated", song="Keep"))
    m.refresh_names(store, lib)
    got = {t.path: (t.artist, t.song) for t in store.all()}
    assert got[str(unk.resolve())] == ("MARY POPPINS", "I LOVE TO LAUGH")
    assert got[str(cur.resolve())] == ("Curated", "Keep")  # never clobbered


def test_refresh_skips_three_part_and_unparseable_stems(m, store, tmp_path, capsys) -> None:
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "SC-1 - A - B.zip").write_bytes(b"z")           # 3 parts: left alone
    (lib / "plain.zip").write_bytes(b"z")                  # unparseable
    (lib / "readme.txt").write_text("x")
    (lib / "sub").mkdir()
    m.refresh_names(store, lib)
    out = capsys.readouterr().out
    assert "Skip unexpected: plain" in out
    assert "SC-1" not in out


def test_refresh_prints_progress_every_tenth_change(m, store, tmp_path, capsys) -> None:
    _known(store, "Abba")
    lib = tmp_path / "lib"
    lib.mkdir()
    for i in range(10):
        p = lib / f"Abba - Song{i}.zip"
        p.write_bytes(b"z")
        store.add(make_track(path=str(p.resolve()), artist="?", song="?"))
    m.refresh_names(store, lib)
    assert "." in capsys.readouterr().out
