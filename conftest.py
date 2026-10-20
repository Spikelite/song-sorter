"""Shared test fixtures.

* Every test runs with the network blocked. A real HTTP request or socket
  connection fails loudly, so no test can quietly depend on, or hammer,
  MusicBrainz / AcousticBrainz. Tests that exercise network code inject fakes.
* ``prompts`` scripts questionary. Queue the answers a flow will be asked for;
  the fake hands them back in order and fails on a prompt the test did not
  expect, on a ``select`` answer the real menu would never offer, or on answers
  left unused (which means the flow took a different path than the test
  believes).
* ``mainmod`` imports main.py with every state path redirected into tmp_path,
  so no test can read or clobber the real .cache/song-sorter store. It skips
  where main.py's dependencies are missing (e.g. a Windows box without the C
  compiler zipfile-deflate64 needs); CI installs everything and runs it all.
"""

from __future__ import annotations

import socket
import urllib.request

import pytest

# Sentinel answer: reply with whatever default the prompt itself offers.
DEFAULT = object()


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("test attempted a real network call")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


class _Question:
    def __init__(self, owner, kind, message, args, kwargs):
        self._owner, self._kind, self._message = owner, kind, message
        self._args, self._kwargs = args, kwargs

    def ask(self):
        return self._owner._answer(self._kind, self._message, self._args, self._kwargs)


class ScriptedPrompts:
    KINDS = ("select", "confirm", "text", "path")

    def __init__(self, monkeypatch, questionary):
        self._q = questionary
        self._queue: list = []
        self.asked: list[tuple[str, str]] = []
        self.printed: list[str] = []
        for kind in self.KINDS:
            monkeypatch.setattr(questionary, kind, self._maker(kind))
        monkeypatch.setattr(questionary, "print",
                            lambda *a, **k: self.printed.append(" ".join(map(str, a))))

    def _maker(self, kind):
        def make(message="", *args, **kwargs):
            return _Question(self, kind, message, args, kwargs)
        return make

    def queue(self, *answers) -> "ScriptedPrompts":
        self._queue.extend(answers)
        return self

    def _allowed(self, args, kwargs) -> list | None:
        choices = kwargs.get("choices", args[0] if args else None)
        if choices is None:
            return None
        return [getattr(c, "value", c) for c in choices
                if not isinstance(c, self._q.Separator)]

    def _answer(self, kind, message, args, kwargs):
        self.asked.append((kind, message))
        if not self._queue:
            raise AssertionError(f"unscripted {kind} prompt: {message!r}")
        ans = self._queue.pop(0)
        if ans is DEFAULT:
            return kwargs.get("default")
        if callable(ans):
            return ans(kind, message, kwargs)
        if kind == "select" and ans is not None:
            allowed = self._allowed(args, kwargs)
            if allowed is not None:
                assert ans in allowed, (
                    f"{ans!r} is not offered by {message!r}; choices were {allowed!r}")
        return ans

    @property
    def output(self) -> str:
        return "\n".join(self.printed)

    def assert_done(self) -> None:
        assert not self._queue, f"unused scripted answers: {self._queue!r}"


def pick(fragment: str):
    """Scripted answer for a ``select``: choose the option whose visible title
    contains `fragment`, and return that option's REAL value. Menus often use
    live objects (IndexNode, Track) as values, which a test cannot construct in
    advance; picking by title is also how a person actually drives the menu."""
    def choose(kind, message, kwargs):
        for c in kwargs.get("choices", []):
            title = getattr(c, "title", c)
            if fragment in str(title):
                return getattr(c, "value", c)
        titles = [str(getattr(c, "title", c)) for c in kwargs.get("choices", [])]
        raise AssertionError(f"no choice containing {fragment!r} in {message!r}: {titles}")
    return choose


@pytest.fixture
def prompts(monkeypatch):
    questionary = pytest.importorskip("questionary")
    sp = ScriptedPrompts(monkeypatch, questionary)
    yield sp
    sp.assert_done()


@pytest.fixture
def mainmod(monkeypatch, tmp_path):
    main = pytest.importorskip("main")
    state = tmp_path / "state"
    for attr, name in (("_CACHE_PATH", "cache.json"),
                       ("_REVIEW_STATE_PATH", "review-state.json"),
                       ("_CONFIG_PATH", "config.json"),
                       ("_RESOLUTIONS_PATH", "resolutions.json"),
                       ("_ARTIST_ALIASES_PATH", "artist-aliases.json"),
                       ("_KEY_OVERRIDES_PATH", "key-overrides.json")):
        monkeypatch.setattr(main, attr, state / name)
    monkeypatch.setattr(main, "_MB_MIN_INTERVAL", 0.0)   # never sleep in tests
    return main


def make_track(path="/lib/A - S.zip", artist="A", song="S",
               file_types=("zip",), **metadata):
    from track import Track
    return Track(path=path, file_types=list(file_types), artist=artist,
                 song=song, metadata={k: str(v) for k, v in metadata.items()})


@pytest.fixture
def store():
    from track import TrackStore
    return TrackStore()


# --- synthetic media -------------------------------------------------------
#
# Real (silent) MPEG-1 Layer III frames rather than mocks, so mutagen's actual
# parser is exercised. 128 kbps / 44.1 kHz frames are 417 bytes: a 4-byte
# header plus 413 payload bytes.

_MP3_HEADER = bytes([0xFF, 0xFB, 0x90, 0x64])
CDG_BYTES = b"	" * 96 * 4   # arbitrary CDG packets; only size/CRC are ever read
_MP3_FRAME = 417


def mp3_bytes(frames: int = 60, marker: bytes = b"") -> bytes:
    """`frames` silent MP3 frames. `marker` is embedded in the first frame's
    payload so a test can locate these exact bytes inside a stored zip."""
    payload = bytearray(_MP3_FRAME - len(_MP3_HEADER))
    payload[: len(marker)] = marker
    first = _MP3_HEADER + bytes(payload)
    rest = (_MP3_HEADER + bytes(_MP3_FRAME - len(_MP3_HEADER))) * (frames - 1)
    return first + rest


def write_mp3(path, frames: int = 60, marker: bytes = b"", **tags):
    """Write an MP3, optionally with EasyID3 tags (e.g. artist=, initialkey=)."""
    from pathlib import Path
    path = Path(path)
    path.write_bytes(mp3_bytes(frames, marker))
    if tags:
        import importlib
        # imported for its side effect: it registers the 'initialkey' (TKEY) mapping
        importlib.import_module("track_inspect")
        from mutagen.easyid3 import EasyID3
        t = EasyID3()
        for k, v in tags.items():
            t[k] = v
        t.save(str(path))
    return path


def write_zip(path, members: dict, compression=None):
    """Zip `members` ({name: bytes}); ZIP_STORED by default so tests can reach
    into the archive and corrupt specific bytes."""
    import zipfile
    comp = zipfile.ZIP_STORED if compression is None else compression
    with zipfile.ZipFile(path, "w", compression=comp) as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path
