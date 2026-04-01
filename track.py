"""Track model and TrackStore for in-memory storage with disk persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Track:
    """A karaoke track"""

    path: str
    file_types: list[str]
    artist: str
    song: str
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dict."""
        return {
            "path": self.path,
            "file_types": self.file_types,
            "artist": self.artist,
            "song": self.song,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Track:
        """Deserialize from dict (e.g. from JSON)."""
        return cls(
            path=data["path"],
            file_types=list(data.get("file_types", [])),
            artist=data.get("artist", ""),
            song=data.get("song", ""),
            metadata=dict(data.get("metadata", {})),
        )


class TrackStore:
    """In-memory store for tracks with load/save to disk."""

    _VERSION = 1

    def __init__(self) -> None:
        self._by_path: dict[str, Track] = {}

    def add(self, track: Track) -> None:
        """Add a track."""
        self._by_path[track.path] = track

    def all(self) -> list[Track]:
        """Return all tracks."""
        return list(self._by_path.values())

    def get(self, path: Path) -> Track | None:
        return self._by_path.get(path)

    def load(self, cache_path: str | Path) -> None:
        """Load store from disk. Replaces in-memory contents."""
        p = Path(cache_path)
        if not p.exists():
            self._by_path = {}
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version", 1)
        if version != self._VERSION:
            raise ValueError(f"Unsupported cache version {version}")
        track_list = [Track.from_dict(t) for t in data.get("tracks", [])]
        track_dict = {t.path: t for t in track_list}
        self._by_path = track_dict

    def save(self, path: str | Path) -> None:
        """Write store to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._VERSION,
            "tracks": [t.to_dict() for t in self.all()],
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
