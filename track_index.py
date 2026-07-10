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

def iter_tracks(node: IndexNode):
    """Every track under a node, depth-first."""
    stack = [node]
    while stack:
        n = stack.pop()
        if n.is_leaf():
            yield from n.list_tracks()
        else:
            stack.extend(n.list_nodes().values())


def majority_raw(node: IndexNode, field: str) -> str:
    """The most common RAW spelling of `field` among a node's tracks.

    Merge/cleanup passes match on clean keys but must DISPLAY a real spelling:
    writing the clean key back (lowercased, punctuation-stripped) is how the
    library filled up with 'wind & fire earth'-style display names."""
    counts: dict[str, int] = {}
    for t in iter_tracks(node):
        v = getattr(t, field, "")
        if v:
            counts[v] = counts.get(v, 0) + 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda kv: kv[1])[0]


def safe_folder(name: str) -> str:
    """Folder-safe artist name for the export tree: path separators and other
    filename-hostile characters become '-', so 'ac/dc' cannot nest an extra
    directory level (which the export pruner's depth==3 filter never sees)."""
    return re.sub(r'[\\/:*?"<>|]', "-", name).strip() or "_"


def _echo_tokens(s: str) -> frozenset:
    """Order-insensitive word set, so 'Isaak, Chris' matches 'Chris Isaak'."""
    s = re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())
    return frozenset(w for w in s.split() if w)


def strip_artist_echo(artist: str, song: str) -> str | None:
    """Remove an artist echo from a song title, or None if there is none.

    Source filenames like 'Isaak, Chris-Wicked Game' sometimes survive into
    the song field, so the display chain reads 'Chris Isaak - Isaak,
    Chris-Wicked Game' (issue #2). A dash-separated leading or trailing
    segment is stripped ONLY when its word set equals the artist's exactly
    (order-insensitive, so comma forms match) -- titles that merely contain
    the artist's name ("My Home's In Alabama") never lose words."""
    at = _echo_tokens(artist)
    if not at or not song:
        return None
    dashes = [m.start() for m in re.finditer("-", song)]
    for i in dashes:  # leading echo: grow the left side dash by dash
        left, right = song[:i].strip(), song[i + 1:].strip()
        if left and right and _echo_tokens(left) == at:
            return right
    for i in reversed(dashes):  # trailing echo: grow the right side
        left, right = song[:i].strip(), song[i + 1:].strip()
        if left and right and _echo_tokens(right) == at:
            return left
    return None


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv"}


def uncomma_artist(artist: str) -> str | None:
    """'Last, First' -> 'First Last' for PERSON names; None when unsafe.

    Learned the hard way (the 'wind & fire earth' incident): band names keep
    their commas ('Earth, Wind & Fire'), multi-artist credits ('Andrews,
    Michael Ft. Gary Jules') aren't one person, and generational suffixes
    relocate ('Davis, Sammy Jr.' -> 'Sammy Davis Jr.'). Callers must treat
    None as "leave it alone" -- never fall back to a blind swap."""
    if artist.count(",") != 1:
        return None
    last, _, first = artist.partition(",")
    last, first = last.strip(), first.strip()
    if not last or not first:
        return None
    suffix = ""
    head = first.split()
    if len(head) > 1 and head[-1].rstrip(".").lower() in _NAME_SUFFIXES:
        suffix = " " + head[-1]
        first = " ".join(head[:-1])
    joined = f" {last.lower()} {first.lower()} "
    if "&" in joined or "+" in joined:
        return None
    for w in (" and ", " ft ", " ft. ", " feat ", " feat. ", " featuring ",
              " with ", " the "):
        if w in joined:
            return None
    if len(first.split()) > 3 or len(last.split()) > 3:
        return None
    return f"{first} {last}{suffix}"


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
