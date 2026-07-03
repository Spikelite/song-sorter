from __future__ import annotations 

import re
from track import TrackStore, Track

from pathlib import Path

class IndexNode:

    def __init__(self):
        self.nodes: dict[str, IndexNode] = {}
        self.tracks: list[Track] = []
        self._count: int | None = None
    
    def add(self, parts: list[str], t: Track):
        if len(parts) <= 1:
            self.tracks.append(t)
        else:
            if parts[0] not in self.nodes:
                self.nodes[parts[0]] = IndexNode()
            self.nodes[parts[0]].add(parts[1:], t)

    def is_leaf(self) -> bool:
        # true iff empty
        return not self.nodes

    def list_nodes(self) -> dict[str, IndexNode]:
        keys = sorted(self.nodes.keys())

        return {k: self.nodes[k] for k in keys}

    def list_tracks(self) -> list[Track]:
        return sorted(self.tracks, key=lambda t: f"{t.artist} - {t.song}")

    def count(self) -> int:
        if self._count is not None:
            return self._count

        total = 0
        for n in self.nodes.values():
            total += n.count()
        
        self._count = total + len(self.tracks)
        return self._count


class TrackIndex:
    """ Support browsing the tracks """

    def __init__(self, store: TrackStore) -> None:
        # go through each track, and split its path
        self.root = IndexNode()

        for t in store.all():
            _path = Path(t.path)
            self.root.add(_path.parts, t)

    def get_root(self) -> IndexNode:
        return self.root


def clean_artist(artist: str) -> str:
    art = artist.lower()
    art = art.replace("_", " ")
    like_and = [" with ", " and ", " + ", " feat. ", " f. ", " ft. ", " ft ", " featuring ", " feat "]
    for n in like_and:
        art = art.replace(n, " & ")
    art = art.replace("'", "")
    art = art.replace(".", " ")
    art = art.replace("  ", " ")
    art = art.removeprefix("the ")
    art = art.removesuffix(", the")
    art = art.strip()
    return art

def clean_song(song: str) -> str:
    song = song.lower()
    # "[sc karaoke]"-style brand tags must not distinguish songs: left in, one
    # title splits into several "distinct" songs, duplicating Final-final
    # exports. Clean strips them from the data too; this keeps grouping right
    # even for not-yet-cleaned batches.
    song = re.sub(r"\[[^\]]*karaoke[^\]]*\]", " ", song)
    song = song.replace("-", "")
    song = song.replace(".", "")
    song = song.replace("&", "and")
    song = song.replace("in' ", "ing ")
    song = song.replace("(duet)", "")
    song = song.replace("(solo)", "")
    song = song.replace("(gospel)", "")
    song = song.replace("(instrumental version)", "")
    song = song.removeprefix("a ")
    song = song.removeprefix("the ")
    song = song.removesuffix(", the")
    song = song.removesuffix(" the")
    song = song.removesuffix(", a")
    song = song.replace(",", "")

    # remove any parenthetical blocks
    song = re.sub(r'\s*\([^)]*\)\s*', ' ', song)

    song = re.sub(r"\s{2,}", " ", song)  # runs of spaces must not split groups
    song = song.strip()
    return song

class ArtistIndex:
    """ Browse tracks by artist """

    @staticmethod
    def from_store(store: TrackStore):
        return ArtistIndex(store.all())
    
    def __init__(self, store: list[Track]) -> None:
        # give each track as artist - song - path
        self._root = IndexNode()

        for t in store:
            art = clean_artist(t.artist)
            song = clean_song(t.song)

            # for easier browsing, group by first letter
            prefix = art[0] if art else ""
            if re.match("[^a-z]", prefix):
                prefix = "#"

            _parts = [prefix, art, song, t.path]

            self._root.add(_parts, t)

    def get_root(self) -> IndexNode:
        return self._root

    def single_artists(self, max_songs: int = 3) -> list[Track]:
        """ return a artists with few tracks """
        result = []
        for letter, top_nodes in self._root.list_nodes().items():
            for artist, anode in top_nodes.list_nodes().items():
                if anode.count() > max_songs:
                    continue
                for song, snode in anode.list_nodes().items():
                    result.append(snode.list_tracks()[0])
        return result

    def count_artists(self, low_bound: int = 1) -> list[str]:
        """ return artists with > 1 song """
        result = []

        alphabet = self._root.list_nodes()
        for letter, top_nodes in alphabet.items():
            if letter == "":
                continue
            for artist, anode in top_nodes.list_nodes().items():
                if anode.count() > low_bound:
                    result.append(artist)
        
        return result

class SongIndex:
    """ Browse tracks by song """

    @staticmethod
    def from_store(store: TrackStore):
        return SongIndex(store.all())
    
    def __init__(self, store: list[Track]) -> None:
        # give each track as artist - song - path
        self._root = IndexNode()

        for t in store:
            artist = t.artist
            if artist.lower() in ["unknown", ""]:
                continue

            song = clean_song(t.song)

            # for easier browsing, group by first letter
            prefix = song[0] if song else ""
            if re.match("[^a-z]", prefix):
                prefix = "#"

            _parts = [prefix, song, t.artist, t.path]

            self._root.add(_parts, t)

    def get_root(self) -> IndexNode:
        return self._root
