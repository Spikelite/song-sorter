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


# Catalog-id stems like 'FLY-03-06' / 'SFKK-21-00' heading a filename.
_STEM_CATALOG_RE = re.compile(r"^[A-Za-z]{1,6}-?\d[\w-]*$")
# Stricter shape for a catalog id ANYWHERE in a stem: letter prefix then
# digits/dashes ONLY ('SF 193-16', 'SFKK-21-00', 'EZH-31'), never trailing
# words. Titles like 'Old 67' (no dash, few digits) must not qualify.
_CATALOG_SEG_RE = re.compile(r"^([A-Za-z]{1,6})[- ]?\d[\d\- ]*$")


def is_catalog_segment(seg: str, relaxed: bool = False) -> bool:
    """True when a stem segment is a disc-catalog id like 'SF 193-16'.

    Deliberately strict because titles can look catalog-ish: requires the
    letters-then-digits shape AND a dash ('SF 193-16', 'EZH-31') or 5+
    digits -- so 'Old 67', 'U2', 'Blink 182' and real titles like LeVert's
    'ABC 123' never qualify on shape alone.

    relaxed=True additionally accepts 3+ digits behind a short (<=4 letter)
    prefix ('SF 003'). That form is IDENTICAL in shape to real titles
    ('ABC 123'), so callers may only use it where surrounding context
    disambiguates -- e.g. the '... - SF 003 - 03' trailing pair, where the
    bare track number after it is the tell."""
    t = (seg or "").strip()
    if len(t.replace(" ", "")) < 4:
        return False
    m = _CATALOG_SEG_RE.fullmatch(t)
    if not m:
        return False
    digits = sum(ch.isdigit() for ch in t)
    if "-" in t or digits >= 5:
        return True
    return relaxed and digits >= 3 and len(m.group(1)) <= 4


def _trim_trailing_catalog(parts: list[str]) -> list[str]:
    """Drop trailing catalog markers from stem segments: a plain
    '... - SF 193-16', or the split pair '... - SF 003 - 03' where the disc
    writes the catalog id and the bare track number as separate segments.
    The pair form may use the relaxed catalog shape (the track number after
    it is the disambiguating context); a single trailing segment must match
    strictly, so titles like 'ABC 123' are never eaten."""
    if (len(parts) >= 3 and re.fullmatch(r"\d{1,3}", parts[-1])
            and is_catalog_segment(parts[-2], relaxed=True)):
        return parts[:-2]
    if len(parts) >= 2 and is_catalog_segment(parts[-1]):
        return parts[:-1]
    return parts


# Compact catalog stems with no spaces around the dashes, e.g.
# 'DIS61201-13-MARY POPPINS-I LOVE TO LAUGH' (see parse_artist_song).
_COMPACT_CATALOG_RE = re.compile(r"^[A-Za-z]{1,6}\d{1,6}-\d{1,3}-(.+)$")


def parse_artist_song(stem: str) -> tuple[str, str]:
    """Best-effort (artist, song) from a filename stem.

    Handles the naming orders seen across disc series:
      - '(source) - Artist - Song'   (catalog first -- the common case)
      - 'Artist - Song - SF 193-16'  (catalog LAST: Sunfly Main Series zips;
        the old parser read these as artist='Song', song='SF 193-16' and
        dropped the real artist entirely)
      - 'Artist - Song - SF 003 - 03' (catalog last, split across TWO
        segments: id then bare track number)
      - 'CATALOG-TRACK-ARTIST-SONG'  (compact, no spaced dashes)
    Returns ('', title) when no artist can be inferred."""
    if " - " in stem:
        parts = [p.strip() for p in stem.split(" - ") if p.strip()]
        if parts and not is_catalog_segment(parts[0]):
            trimmed = _trim_trailing_catalog(parts)
            if len(trimmed) != len(parts) and trimmed:
                if len(trimmed) == 1:
                    return "", trimmed[0]
                return trimmed[0], " - ".join(trimmed[1:])
        if len(parts) > 2:
            # (source) - artist - song
            return parts[1], parts[2]
        if len(parts) == 2:
            # (source) - song
            return "", parts[1]
    m = _COMPACT_CATALOG_RE.match(stem)
    if m:
        rest = m.group(1).strip()
        if "-" in rest:
            # first dash splits artist from song; the song itself may keep
            # later dashes ('ZIP-A-DEE-DOO-DAH')
            artist, song = rest.split("-", 1)
            return artist.strip(), song.strip()
        return "", rest  # catalog+track prefix stripped, no artist present
    return "", stem.strip() or "Unknown"


def split_stem(stem: str) -> list[str]:
    """Filename stem -> ' - '-separated segments, catalog ids dropped
    (leading, trailing, or the trailing id+track-number pair -- disc series
    differ on where they put them)."""
    parts = [p.strip() for p in stem.split(" - ") if p.strip()]
    if parts and _STEM_CATALOG_RE.fullmatch(parts[0].replace(" ", "")):
        parts = parts[1:]
    return _trim_trailing_catalog(parts)


def rejoin_artist(parts: list[str], known: set) -> tuple[str, list[str]] | None:
    """Reassemble an artist whose name was dash-split across stem segments.

    'FLY-03-06 - Belinda - Carlisle - Heaven Is A Pla - On Earth' parses as
    artist 'Belinda', song 'Carlisle' -- but joining leading segments and
    checking against the known-artist set recovers ('Belinda Carlisle',
    ['Heaven Is A Pla', 'On Earth']). Longest join wins; None when no join of
    2+ leading segments is a known artist (a 1-segment match is the normal
    parse, not a rejoin)."""
    for k in range(len(parts) - 1, 1, -1):
        cand = " ".join(parts[:k])
        if clean_artist(cand) in known:
            return cand, parts[k:]
    return None


def _frag_norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", (s or "").lower())).strip()


def fragments_match_title(fragments: list[str], title: str) -> bool:
    """True when the stem fragments fit `title` as an elided rendering.

    These disc filenames cut characters out mid-title and mark the cut with a
    dash ('Bel - E Again' == 'Bel[iev]e Again'), and may also truncate the
    end. So: the first fragment must be a PREFIX of the title and every later
    fragment must appear IN ORDER after the previous one; the title may
    continue past the last fragment. All comparisons are lowercase and
    punctuation-insensitive."""
    frags = [_frag_norm(f) for f in fragments]
    frags = [f for f in frags if f]
    t = _frag_norm(title)
    if not frags or not t:
        return False
    if not t.startswith(frags[0]):
        return False
    pos = len(frags[0])
    for f in frags[1:]:
        i = t.find(f, pos)
        if i < 0:
            return False
        pos = i + len(f)
    return True


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
