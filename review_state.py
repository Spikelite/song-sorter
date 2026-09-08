"""Persistent review state for track review mode."""

from __future__ import annotations

import json
import os
from pathlib import Path


class ReviewState:
    """Tracks per-path review status (ok, swap, edit) with disk persistence."""

    _VERSION = 1

    def __init__(self) -> None:
        self._by_path: dict[str, str] = {}

    def get(self, path: str) -> str | None:
        """Return status for path, or None if not reviewed."""
        return self._by_path.get(path)

    def set(self, path: str, status: str) -> None:
        """Set status for path."""
        self._by_path[path] = status

    def load(self, path: str | Path) -> None:
        """Load state from disk. Replaces in-memory contents."""
        p = Path(path)
        if not p.exists():
            self._by_path = {}
            return
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        version = data.get("version", 1)
        if version != self._VERSION:
            raise ValueError(f"Unsupported review-state version {version}")
        self._by_path = dict(data.get("by_path", {}))

    def save(self, path: str | Path) -> None:
        """Write state to disk atomically (temp file + rename).

        Same protection TrackStore.save has, and for a stronger reason: this
        file holds manual review decisions, which cannot be recomputed from the
        media. It is rewritten every 60s during a Musicbrainz run and after
        nearly every keystroke in Review, so an interrupt during a plain write
        would truncate it and lose the lot."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._VERSION,
            "by_path": self._by_path,
        }
        tmp = p.parent / (p.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, p)
