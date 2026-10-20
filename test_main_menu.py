"""main.py: the interactive menu's dispatch table, and the main() entry point.

A mis-wired menu entry (Fuzz calling fuzz_song, say) is a real bug class that
no function-level test catches, so every entry is driven and the function it
reaches is asserted. main() is tested for its one critical safety property: a
cache that exists but will not load must never be overwritten.
"""

from __future__ import annotations

import json

import pytest

from conftest import make_track

# menu entry -> (reached via Advanced submenu?, functions it must call, in order)
DISPATCH = {
    "browse": (False, ["browse"]),
    "artist": (False, ["browse_artist"]),
    "song": (False, ["browse_song"]),
    "final-final": (False, ["tracks_to_keep"]),
    "songbook": (False, ["songbook"]),
    "list": (False, ["report_track_count"]),
    "stats": (False, ["library_stats"]),
    "review": (False, ["review_mode"]),
    "tag-review": (False, ["auto_ok_from_tags"]),
    "tag-swap": (False, ["swap_from_tags"]),
    "musicbrainz": (False, ["musicbrainz_lookup"]),
    "restitch": (False, ["restitch_titles"]),
    "apply-resolutions": (False, ["apply_resolutions"]),
    "key-detect": (False, ["detect_keys"]),
    "unify-artists": (False, ["unify_artists"]),
    "fixup": (False, ["browse_fixup"]),
    "fix-artist": (False, ["fix_artist"]),
    "fix-unknown": (False, ["fix_unknown"]),
    # the documented order: Clean -> Trailing-article -> Tag-fill -> Uncomma ->
    # Ungroup -> Fuzz -> Fuzz_song
    "all-clean": (False, ["clean", "trailing_article", "fill_artist_from_tags",
                          "standard_artist", "ungroup_artist", "fuzz_artist", "fuzz_song"]),
    "clean": (True, ["clean"]),
    "tag-fill": (True, ["fill_artist_from_tags"]),
    "unswap": (True, ["find_swapped"]),
    "uncomma": (True, ["standard_artist"]),
    "trailing-article": (True, ["trailing_article"]),
    "ungroup": (True, ["ungroup_artist"]),
    "fuzz": (True, ["fuzz_artist"]),
    "fuzz_song": (True, ["fuzz_song"]),
}
PATH_ENTRIES = {"search": "add_tracks", "detail": "add_details", "refresh": "refresh_names"}

ALL_TARGETS = sorted({f for _adv, fs in DISPATCH.values() for f in fs}
                     | set(PATH_ENTRIES.values()))


@pytest.fixture
def m(mainmod):
    return mainmod


@pytest.fixture
def calls(m, monkeypatch):
    seen: list[str] = []
    for name in ALL_TARGETS:
        monkeypatch.setattr(m, name, (lambda n: lambda *a, **k: seen.append(n))(name))
    return seen


@pytest.mark.parametrize("entry", sorted(DISPATCH))
def test_every_menu_entry_reaches_its_function(m, store, prompts, calls, entry) -> None:
    advanced, want = DISPATCH[entry]
    if advanced:
        prompts.queue("__advanced__", entry, "exit")
    else:
        prompts.queue(entry, "exit")
    m.run_interactive(store)
    assert calls == want


@pytest.mark.parametrize("entry, target", sorted(PATH_ENTRIES.items()))
def test_path_entries_prompt_then_dispatch(m, store, prompts, calls, monkeypatch,
                                           entry, target, tmp_path) -> None:
    seen_default = []
    monkeypatch.setattr(m, "import_path",
                        lambda default=".": (seen_default.append(default), tmp_path)[1])
    store.add(make_track(str(tmp_path / "SC" / "a.zip")))
    prompts.queue(entry, "exit")
    m.run_interactive(store)
    assert calls == [target]
    assert seen_default == [str(tmp_path / "SC")]         # pre-filled from the store


@pytest.mark.parametrize("entry", sorted(PATH_ENTRIES))
def test_path_entries_do_nothing_when_the_prompt_is_cancelled(m, store, prompts, calls,
                                                              monkeypatch, entry) -> None:
    monkeypatch.setattr(m, "import_path", lambda default=".": None)
    prompts.queue(entry, "exit")
    m.run_interactive(store)
    assert calls == []


@pytest.mark.parametrize("answer", ["__back__", None])
def test_advanced_back_or_cancel_returns_to_the_main_menu(m, store, prompts, calls, answer) -> None:
    prompts.queue("__advanced__", answer, "exit")
    m.run_interactive(store)
    assert calls == []


def test_cancelling_the_main_menu_exits(m, store, prompts, calls) -> None:
    prompts.queue(None)
    m.run_interactive(store)
    assert calls == []


def test_online_entries_are_labelled(m, store, prompts, calls) -> None:
    titles = []

    def grab(kind, message, kwargs):
        titles.extend(str(getattr(c, "title", c)) for c in kwargs["choices"])
        return "exit"
    prompts.queue(grab)
    m.run_interactive(store)
    for online in ("Musicbrainz (online)", "Restitch (online)", "Key-detect (online)"):
        assert online in titles
    assert "Browse" in titles and "Browse (online)" not in titles


def test_every_option_is_on_the_menu_exactly_once(m, store, prompts, calls) -> None:
    """The layout guard inside run_interactive asserts options == placed; this
    checks the complementary property from the outside: nothing appears twice."""
    import questionary
    values: list = []

    def collect(then):
        def choose(kind, message, kwargs):
            values.extend(c.value for c in kwargs["choices"]
                          if not isinstance(c, questionary.Separator))
            return then
        return choose
    prompts.queue(collect("__advanced__"), collect("__back__"), "exit")
    m.run_interactive(store)
    entries = [v for v in values if v not in ("__advanced__", "__back__", "exit")]
    assert len(entries) == len(set(entries))
    assert set(entries) == set(DISPATCH) | set(PATH_ENTRIES)


# --- main() ------------------------------------------------------------------

def test_main_loads_runs_and_saves(m, monkeypatch, capsys) -> None:
    from track import TrackStore
    seed = TrackStore()
    seed.add(make_track("/a.zip", "Abba", "Waterloo"))
    seed.save(m._CACHE_PATH)
    ran = []
    monkeypatch.setattr(m, "run_interactive", lambda store: ran.append(len(store.all())))
    m.main()
    assert ran == [1]
    assert json.loads(m._CACHE_PATH.read_text(encoding="utf-8"))["tracks"][0]["artist"] == "Abba"
    assert "Goodbye." in capsys.readouterr().out


def test_main_saves_even_when_the_menu_crashes(m, monkeypatch) -> None:
    def boom(store):
        store.add(make_track("/new.zip", "New", "Work"))
        raise RuntimeError("menu crashed")
    monkeypatch.setattr(m, "run_interactive", boom)
    with pytest.raises(RuntimeError):
        m.main()
    saved = json.loads(m._CACHE_PATH.read_text(encoding="utf-8"))["tracks"]
    assert [t["path"] for t in saved] == ["/new.zip"]      # progress kept


@pytest.mark.parametrize("corrupt", ['{"version": 99, "tracks": []}', '{"version": 1, "tr'])
def test_main_refuses_to_overwrite_a_cache_it_cannot_load(m, monkeypatch, capsys, corrupt) -> None:
    """A future-version or truncated cache must be left byte-for-byte intact:
    continuing with an empty store would save nothing over the whole library."""
    m._CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    m._CACHE_PATH.write_text(corrupt, encoding="utf-8")
    monkeypatch.setattr(m, "run_interactive",
                        lambda store: pytest.fail("must not run on an unloadable cache"))
    m.main()
    assert m._CACHE_PATH.read_text(encoding="utf-8") == corrupt
    assert "cannot be loaded" in capsys.readouterr().out
