"""Persistent review state for track review mode."""

from __future__ import annotations

import json
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
        """Write state to disk."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": self._VERSION,
            "by_path": self._by_path,
        }
        with open(p, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
