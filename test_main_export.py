"""main.py: Final-final export, index.json, the songbook, List and Stats.

Final-final writes and PRUNES real files, so it is tested against a real temp
output tree: layout, incrementality, version selection, index contents, and
that the pruner touches only stale files in its own depth-3 layout.
"""

from __future__ import annotations

import json
import os

import pytest

from conftest import make_track


@pytest.fixture
def m(mainmod):
    return mainmod


def _src(tmp_path, rel, data=b"media", artist="Abba", song="Waterloo",
         types=("zip",), **md):
    """A real source file on disk plus its store record."""
    p = tmp_path / "src" / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    for ext in types:
        p.with_suffix(f".{ext}").write_bytes(data)
    md.setdefault("mp3_size", len(data))
    return make_track(str(p), artist, song, file_types=types, **md)


def _export(m, prompts, out, versions="1", prune=None):
    answers = [str(out), versions]
    if prune is not None:
        answers.append(prune)
    prompts.queue(*answers)


def _index(out):
    return json.loads((out / "index.json").read_text(encoding="utf-8"))["songs"]


# --- _best_track / _ranked_tracks -------------------------------------------

def test_best_and_ranked_agree_on_the_largest_copy(m) -> None:
    a = make_track("/a", mp3_size=10)
    b = make_track("/b", mp3_size=30)
    c = make_track("/c")                       # no size
    d = make_track("/d", mp3_size="junk")      # unparseable
    assert m._best_track([a, b, c, d]) is b
    assert m._ranked_tracks([a, b, c, d])[0] is b
    assert m._ranked_tracks([c, d]) == [c, d]  # stable when all are 0


def test_ranked_tracks_tolerate_non_string_sizes(m) -> None:
    t = make_track("/a")
    t.metadata["mp3_size"] = ["not", "a", "size"]
    assert m._ranked_tracks([t]) == [t]


# --- tracks_to_keep: prompts ------------------------------------------------

@pytest.mark.parametrize("answers", [[None], ["   "], ["/out", None]])
def test_cancelling_either_prompt_writes_nothing(m, store, prompts, tmp_path, answers) -> None:
    store.add(_src(tmp_path, "a.zip"))
    prompts.queue(*[a.replace("/out", str(tmp_path / "out")) if isinstance(a, str) else a
                    for a in answers])
    m.tracks_to_keep(store)
    assert not (tmp_path / "out" / "index.json").exists()


def test_output_dir_and_version_limit_are_remembered(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "a.zip"))
    out = tmp_path / "out"
    _export(m, prompts, out, versions="not-a-number")     # falls back to 1
    m.tracks_to_keep(store)
    cfg = m._load_config()
    assert cfg["output_path"] == str(out)
    assert cfg["version_limit"] == 1


def test_a_corrupt_saved_version_limit_still_prompts_with_a_default(m, store, prompts, tmp_path) -> None:
    m._save_config({"version_limit": "garbage"})
    store.add(_src(tmp_path, "a.zip"))
    from conftest import DEFAULT
    prompts.queue(str(tmp_path / "out"), DEFAULT)
    m.tracks_to_keep(store)
    assert m._load_config()["version_limit"] == 1


# --- tracks_to_keep: layout and index ---------------------------------------

def test_exports_best_copy_into_the_letter_artist_layout(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "SC/Abba - Waterloo.zip", b"best!!", length_seconds="165.4",
                   key="A minor", key_confidence="0.9", key_source="auto", key_camelot="8A"))
    store.add(_src(tmp_path, "SF/Abba - Waterloo.zip", b"lesser"[:3]))
    store.add(_src(tmp_path, "X/Unknown - Thing.zip", artist="Unknown", song="Thing"))
    store.add(_src(tmp_path, "X/10cc - Rubber.zip", artist="10cc", song="Rubber"))
    store.add(_src(tmp_path, "X/ACDC - TNT.zip", artist="AC/DC", song="TNT"))
    out = tmp_path / "out"
    _export(m, prompts, out)
    m.tracks_to_keep(store)

    assert (out / "a" / "abba" / "Abba - Waterloo.zip").read_bytes() == b"best!!"
    assert (out / "#" / "10cc" / "10cc - Rubber.zip").exists()        # non-letter
    assert (out / "a" / "ac-dc" / "ACDC - TNT.zip").exists()          # safe_folder
    assert not any(p.name.startswith("Unknown") for p in out.rglob("*.zip"))

    songs = {e["title"]: e for e in _index(out)}
    assert set(songs) == {"Waterloo", "Rubber", "TNT"}
    w = songs["Waterloo"]
    assert w["path"] == "a/abba/Abba - Waterloo.zip"
    assert w["duration"] == 165
    assert (w["key"], w["key_source"], w["key_camelot"]) == ("A minor", "auto", "8A")
    assert "versions" not in w
    assert (out / "songbook.html").exists()
    assert "Songbook refreshed" in prompts.output


def test_rerun_is_incremental_and_counts_missing_sources(m, store, prompts, tmp_path) -> None:
    t = _src(tmp_path, "SC/Abba - Waterloo.zip")
    store.add(t)
    out = tmp_path / "out"
    _export(m, prompts, out)
    m.tracks_to_keep(store)
    assert "copied 1" in prompts.output
    prompts.printed.clear()
    gone = _src(tmp_path, "SC/Blur - Song 2.zip", artist="Blur", song="Song 2")
    store.add(gone)
    os.remove(gone.path)                                   # source vanished
    _export(m, prompts, out)
    m.tracks_to_keep(store)
    assert "copied 0, skipped 1 unchanged, 1 source file(s) missing" in prompts.output


def test_versions_are_exported_with_labels_and_their_own_keys(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "SC Disc/Abba - Waterloo.zip", b"x" * 9, key="A minor",
                   key_confidence="0.9", key_source="auto"))
    store.add(_src(tmp_path, "SF Disc/Abba - Waterloo (SF).zip", b"x" * 6,
                   length_seconds="bad", key="B minor", key_confidence="0.8",
                   key_source="auto"))
    store.add(_src(tmp_path, "Loose/Abba - Waterloo (cdg).cdg", b"x" * 3,
                   types=("cdg",)))                        # unplayable alternate
    store.add(_src(tmp_path, "Dup/Abba - Waterloo.zip", b"x" * 2))   # same stem as best
    out = tmp_path / "out"
    _export(m, prompts, out, versions="4")
    m.tracks_to_keep(store)
    (entry,) = _index(out)
    assert entry["key"] == "A minor"
    assert entry["versions"] == [{
        "path": "a/abba/Abba - Waterloo (SF).zip", "label": "SF Disc",
        "duration": 0, "key": "B minor", "key_confidence": 0.8, "key_source": "auto",
    }]
    assert (out / "a" / "abba" / "Abba - Waterloo (SF).zip").exists()


def test_version_label_falls_back_when_there_is_no_folder(m, store, prompts, tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "Abba - Waterloo.zip").write_bytes(b"x" * 9)
    (tmp_path / "Abba - Waterloo 2.zip").write_bytes(b"x" * 5)
    store.add(make_track("Abba - Waterloo.zip", "Abba", "Waterloo", mp3_size=9))
    store.add(make_track("Abba - Waterloo 2.zip", "Abba", "Waterloo", mp3_size=5))
    out = tmp_path / "out"
    _export(m, prompts, out, versions="2")
    m.tracks_to_keep(store)
    assert _index(out)[0]["versions"][0]["label"] == "Alternate"


def test_a_best_copy_with_no_playable_file_gets_no_index_entry(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "Loose/Abba - Waterloo.cdg", types=("cdg",)))
    out = tmp_path / "out"
    _export(m, prompts, out)
    m.tracks_to_keep(store)
    assert _index(out) == []
    assert (out / "a" / "abba" / "Abba - Waterloo.cdg").exists()     # still exported


# --- tracks_to_keep: pruning ------------------------------------------------

def _stale_tree(out):
    (out / "z" / "old artist").mkdir(parents=True)
    (out / "z" / "old artist" / "gone.zip").write_bytes(b"old")
    (out / "notes.txt").parent.mkdir(parents=True, exist_ok=True)
    (out / "notes.txt").write_text("depth 1: never pruned")
    deep = out / "a" / "b" / "c"
    deep.mkdir(parents=True)
    (deep / "deep.zip").write_bytes(b"depth 4: never pruned")


def test_prune_removes_only_stale_depth_three_files(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "SC/Abba - Waterloo.zip"))
    out = tmp_path / "out"
    _stale_tree(out)
    _export(m, prompts, out, prune=True)
    m.tracks_to_keep(store)
    assert not (out / "z").exists()                         # file and empty dirs gone
    assert (out / "notes.txt").exists()
    assert (out / "a" / "b" / "c" / "deep.zip").exists()
    assert (out / "a" / "abba" / "Abba - Waterloo.zip").exists()
    assert "Pruned 1 stale file(s)" in prompts.output


def test_prune_can_be_declined(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "SC/Abba - Waterloo.zip"))
    out = tmp_path / "out"
    _stale_tree(out)
    _export(m, prompts, out, prune=False)
    m.tracks_to_keep(store)
    assert (out / "z" / "old artist" / "gone.zip").exists()
    assert "Left 1 stale file(s) in place" in prompts.output


@pytest.mark.xfail(strict=True, raises=IndexError,
                   reason="GH #27: Final-final indexes ranked[0] with no guard, so an "
                          "empty store raises IndexError instead of exporting nothing")
def test_final_final_on_an_empty_library_exports_nothing(m, store, prompts, tmp_path) -> None:
    out = tmp_path / "out"
    _export(m, prompts, out)
    m.tracks_to_keep(store)
    assert _index(out) == []


@pytest.mark.xfail(strict=True, reason="GH #28: two different songs by one artist whose "
                   "best copies share a filename stem collide on one export path")
def test_songs_sharing_a_filename_stem_are_both_exported(m, store, prompts, tmp_path) -> None:
    store.add(_src(tmp_path, "Disc1/Track 01.zip", b"first", song="Waterloo"))
    store.add(_src(tmp_path, "Disc2/Track 01.zip", b"second", song="SOS"))
    out = tmp_path / "out"
    _export(m, prompts, out)
    m.tracks_to_keep(store)
    paths = [e["path"] for e in _index(out)]
    assert len(set(paths)) == 2, f"two songs share one exported file: {paths}"


# --- songbook ---------------------------------------------------------------

def test_songbook_entries_dedupe_and_describe_the_best_copy(m, store) -> None:
    store.add(make_track("/1", "Abba", "Waterloo [SF Karaoke]", mp3_size=9,
                         length_seconds="225.0", bitrate_bps="192000",
                         tag_year="Recorded 1974", tag_album="A" * 60))
    store.add(make_track("/2", "ABBA", "Waterloo (wbgv)", mp3_size=1))
    store.add(make_track("/3", "Abba", "SOS", length_seconds="x", bitrate_bps="y",
                         tag_album="<not-found>"))
    store.add(make_track("/4", "Unknown", "Hidden"))
    store.add(make_track("/5", "[SF Karaoke]", "Only Noise"))     # cleans to nothing
    rows = m._songbook_entries(store)
    assert [r[:2] for r in rows] == [("Abba", "SOS"), ("Abba", "Waterloo")]
    details = dict(((a, s), d) for a, s, d in rows)
    assert details[("Abba", "Waterloo")] == \
        "Length: 3:45 @ 192 kbps · 1974 · " + "A" * 48
    assert details[("Abba", "SOS")] == ""


def test_songbook_details_with_only_a_bitrate(m, store) -> None:
    store.add(make_track("/1", "Abba", "SOS", bitrate_bps="128000"))
    assert m._songbook_entries(store)[0][2] == "128 kbps"


def test_build_songbook_escapes_title_and_embedded_data(m, store, tmp_path) -> None:
    store.add(make_track("/1", "AC</script>DC", "T\tN\tT"))
    book, n = m.build_songbook(store, tmp_path / "b" / "songbook.html", name="Tom & <Co>")
    html = book.read_text(encoding="utf-8")
    assert n == 1
    assert "Tom &amp; &lt;Co&gt;'s Karaoke Songbook" in html
    assert "</script>DC" not in html                      # cannot close the block
    assert "<\\/script>" in html
    assert "T N T" in html                                # tabs are the row separator


def test_build_songbook_default_title(m, store, tmp_path) -> None:
    book, n = m.build_songbook(store, tmp_path / "s.html")
    assert n == 0 and "<title>Karaoke Songbook</title>" in book.read_text(encoding="utf-8")


def test_songbook_command_remembers_name_and_path(m, store, prompts, tmp_path) -> None:
    out = tmp_path / "book.html"
    prompts.queue("  Spike  ", str(out))
    m.songbook(store)
    assert out.exists()
    cfg = m._load_config()
    assert (cfg["songbook_name"], cfg["songbook_path"]) == ("Spike", str(out))


def test_songbook_default_path_prefers_saved_then_output_then_cache(m, store, prompts, tmp_path) -> None:
    seen = []

    def grab(kind, message, kwargs):
        seen.append(kwargs["default"])
        return None                                        # cancel after peeking
    prompts.queue("", grab)
    m.songbook(store)                                      # nothing saved yet
    m._save_config({"output_path": str(tmp_path / "out")})
    prompts.queue("", grab)
    m.songbook(store)
    m._save_config({"output_path": "x", "songbook_path": "/saved/book.html"})
    prompts.queue("", grab)
    m.songbook(store)
    assert seen == [str(m._CACHE_PATH.parent / "songbook.html"),
                    str(tmp_path / "out" / "songbook.html"),
                    "/saved/book.html"]


@pytest.mark.parametrize("answers", [[None], ["name", None], ["name", "  "]])
def test_songbook_cancel(m, store, prompts, tmp_path, answers) -> None:
    prompts.queue(*answers)
    m.songbook(store)
    assert "songbook_path" not in m._load_config()


# --- List / Stats -----------------------------------------------------------

def test_report_track_count(m, store, prompts) -> None:
    store.add(make_track("/1", "Abba", "Waterloo"))
    store.add(make_track("/2", "ABBA", "Waterloo [SF Karaoke]"))   # same song
    store.add(make_track("/3", "Unknown", "X"))
    m.report_track_count(store)
    assert "Distinct count: 1 / 3" in prompts.output


def test_stats_on_an_empty_library(m, store, prompts) -> None:
    m.library_stats(store)
    assert "Library is empty" in prompts.output


def test_stats_full_report(m, store, prompts) -> None:
    store.add(make_track("/1", "Abba", "Waterloo", file_types=("mp3", "cdg"),
                         length_seconds="200", mp3_size="3000000", cdg_size="1000000",
                         bitrate_bps="128000", tag_year="1974"))
    store.add(make_track("/2", "ABBA", "Waterloo", length_seconds="100",
                         bitrate_bps="192000", tag_year="c.1983"))
    store.add(make_track("/3", "Blur", "Song 2"))
    store.add(make_track("/4", "Unknown", "X", mp3_size="junk"))
    m.library_stats(store)
    out = prompts.output
    assert "Track files:            4" in out
    assert "Distinct songs:         2" in out
    assert "Unknown-artist files:   1" in out
    assert "Duplicate copies:       1" in out
    assert "Avg song length:        2m 30s" in out
    assert "Avg MP3 bitrate:        160 kbps" in out
    assert "1. Abba  (1)" in out or "1. ABBA  (1)" in out
    assert "2  Abba - Waterloo" in out
    assert "1970s  1" in out and "1980s  1" in out
    assert "—" not in out                         # no em dashes in output


def test_stats_without_any_mp3_details(m, store, prompts) -> None:
    store.add(make_track("/1", "Abba", "Waterloo"))
    m.library_stats(store)
    assert "Avg song length" not in prompts.output
    assert "Avg MP3 bitrate" not in prompts.output
    assert "Songs by decade" not in prompts.output
