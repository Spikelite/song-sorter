"""song-sorter: Interactive CLI for organizing karaoke track libraries."""

from __future__ import annotations

import json
import re
import shutil
import socket
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable

import questionary
from questionary import Choice

import rapidfuzz

from tqdm import tqdm

from track import Track, TrackStore
from track_index import ArtistIndex, SongIndex, TrackIndex, IndexNode, clean_artist, clean_song
from track_inspect import track_details
from review_state import ReviewState


_CACHE_PATH = Path(__file__).parent / ".cache" / "song-sorter" / "cache.json"
_REVIEW_STATE_PATH = _CACHE_PATH.parent / "review-state.json"


def _parse_artist_song(stem: str) -> tuple[str, str]:
    """Parse 'Artist - Song' or 'Song' from filename stem."""
    if " - " in stem:
        parts = stem.split(" - ")
        if len(parts) > 2:
            # (source) - artist - song
            return parts[1].strip(), parts[2].strip()
        if len(parts) == 2:
            # (source) - song
            return "", parts[1].strip()
    return "", stem.strip() or "Unknown"


def _uncomma_artist(artist: str) -> str:
    if ',' in artist:
        split = artist.split(",")
        swapped = f"{split[1]} {split[0]}".strip()
        return swapped
    return None


def _default_scan_dir(store: TrackStore) -> str:
    """A sensible default for the path prompt: the directory of the first
    stored track (a previously-scanned location), or '.' if the store is empty.
    With tracks from several roots, this is simply the first one found."""
    for t in store.all():
        if t.path:
            return str(Path(t.path).parent)
    return "."


def import_path(default: str = ".") -> Path | None:
    """Prompt for a path, and verify it exists.  Used to load a folder.

    `default` pre-fills the prompt; callers pass a previously-scanned directory
    so re-runs don't start at '.'."""
    path_str = questionary.path(
        "Enter path to search:",
        default=default,
    ).ask()

    if path_str is None:
        return None

    root = Path(path_str)
    if not root.is_dir():
        print(f"Not a valid directory: {path_str}")
        input("\nPress Enter to continue...")
        return None

    return root
 
def add_tracks(store: TrackStore, root: Path) -> None:
    """Walk root for .zip and .cdg files and add NEW ones to the track store.

    Tracks already in the store are left untouched, so re-running Search is
    additive: it picks up new files without wiping the metadata (hashes, tags,
    detail, review provenance) of files already present. Use Refresh to
    re-parse names on existing tracks."""
    added = 0
    skipped = 0

    files = [p for p in root.rglob("*") if p.is_file()]
    for p in tqdm(files, desc="Scanning", unit="file"):
        suffix = p.suffix.lower()
        if suffix not in (".zip", ".cdg"):
            continue

        tpath = str(p.resolve())
        if store.get(tpath) is not None:
            skipped += 1  # already known -- never overwrite its metadata
            continue

        stem = p.stem
        artist, song = _parse_artist_song(stem)
        if "song-artist" in p.parts:
            # some parts of the path are reversed
            artist, song = song, artist

        if suffix == ".zip":
            file_types = ["zip"]
        else:  # .cdg
            file_types = ["mp3", "cdg"] if p.with_suffix(".mp3").exists() else ["cdg"]

        store.add(Track(
            path=tpath,
            file_types=file_types,
            artist=artist or "Unknown",
            song=song,
        ))
        added += 1

    print(f"\nAdded {added} new track(s), skipped {skipped} already present.")

def add_details(store: TrackStore, root: Path, workers: int = 4,
                checkpoint_seconds: int = 300) -> None:
    """ Modifies track metadata, to include size and mp3 metadata.

    Delta cache: a track is skipped when its source file is unchanged since
    it was last detailed (same size + mtime), so a re-run only touches new
    or modified files. The store is checkpointed (atomically) every
    checkpoint_seconds and once at the end, so even a hard kill resumes from
    the last checkpoint. Delete the cache to force a full re-detail (e.g.
    after changing how details are extracted).

    Detailing runs in a thread pool: track_details is pure, and its heavy
    work (file I/O, decompression, hashing) releases the GIL. On a single
    spinning disk a small worker count (2-4) overlaps CPU with I/O; more
    just causes seek thrashing. Tune `workers` to benchmark your disk."""
    added = 0
    skipped = 0

    # Phase 1 (main thread): decide what actually needs detailing.
    # Sorted for seek-friendly ordering on a single spinning disk.
    files = sorted(p for p in root.rglob("*") if p.is_file())
    todo = []  # (path, track, sig_mtime, sig_size)
    for p in tqdm(files, desc="Scanning", unit="file"):
        track = store.get(str(p.resolve()))
        if track is None:
            continue
        try:
            st = p.stat()
        except OSError:
            continue
        sig_mtime = str(int(st.st_mtime))
        sig_size = str(st.st_size)

        # Skip if already detailed and the source file hasn't changed.
        if (track.metadata.get("src_mtime") == sig_mtime
                and track.metadata.get("src_size") == sig_size):
            skipped += 1
            continue
        todo.append((p, track, sig_mtime, sig_size))

    # Phase 2: detail in parallel; apply results in the main thread so the
    # store stays single-writer (no locking needed). Sorted submission order
    # keeps the few in-flight reads spatially close on the disk.
    last_save = time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(track_details, p): (track, sig_mtime, sig_size)
                   for (p, track, sig_mtime, sig_size) in todo}
        for fut in tqdm(as_completed(futures), total=len(futures),
                        desc="Details", unit="file"):
            track, sig_mtime, sig_size = futures[fut]
            try:
                details = fut.result()
            except Exception as e:
                tqdm.write(f"detail failed {track.path} :: {e}")
                continue
            details["src_mtime"] = sig_mtime
            details["src_size"] = sig_size
            track.metadata.update(details)
            added += 1

            # Periodic atomic checkpoint: a hard kill mid-run then only loses
            # the last <checkpoint_seconds of work instead of everything.
            if time.monotonic() - last_save >= checkpoint_seconds:
                store.save(_CACHE_PATH)
                last_save = time.monotonic()

    store.save(_CACHE_PATH)  # final checkpoint once detailing completes
    print(f"\nAdded {added} track(s), skipped {skipped} unchanged.")

def _format_metadata(metadata: dict[str, str]) -> str:
    
    length = metadata.get("length_seconds", "")
    bitrate = int(int(metadata.get("bitrate_bps", "0"))/1000)
    # seems like always 44k, and 2 channels
    sample_hz = metadata.get("sample_rate_hz", "")
    channels = metadata.get("channels", "")

    track_summary = f"sec {length}, bitrate {bitrate}"

    mp3_size = int(int(metadata.get("mp3_size", "0"))/1000)
    cdg_size = int(int(metadata.get("cdg_size", "0"))/1000)
    mp3_hash = metadata.get("mp3_hash", "")[:6]
    cdg_hash = metadata.get("cdg_hash", "")[:6]

    file_summary = f" m.sz {mp3_size}k c.sz {cdg_size}k m# {mp3_hash} c# {cdg_hash}"

    return track_summary + file_summary

def refresh_names(store: TrackStore, root: Path) -> None:
    """ Used for a repeat walk - recheck name parsing. """
    added = 0
    possible = 0

    artists = ArtistIndex.from_store(store)
    likely_artists = set(artists.count_artists(low_bound=5))

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        if p.suffix.lower() not in [".zip" , ".cdg"]:
            continue

        stem = p.stem
        stem = stem.replace("_", " ")
        parts = stem.split(" - ")

        if len(parts) != 2:
            # only making changes to {??} - {song} tracks
            if len(parts) != 3:
                # But length != 2 or 3 is extra strange
                print(f"Skip unexpected: {stem}")
            continue

        possible += 1
        part1 = parts[0]
        part2 = parts[1]

        p1_artist = clean_artist(part1)
        p1_prefix = p1_artist.split('&', 1)[0]
        p1_prefix = _uncomma_artist(p1_prefix) or p1_prefix

        is_p1_artist = p1_artist in likely_artists or p1_prefix in likely_artists

        p2_artist = clean_artist(part2)
        p2_prefix = p2_artist.split('&', 1)[0]
        p2_prefix = _uncomma_artist(p2_prefix) or p2_prefix

        is_p2_artist = p2_artist in likely_artists or p2_prefix in likely_artists

        if is_p1_artist and not is_p2_artist:
            artist = part1
            song = part2
        elif is_p2_artist and not is_p1_artist:
            artist = part2
            song = part1
        else:
            continue

        track = store.get(str(p.resolve()))
        if track is None:
            print(f"Found new mapping, but no track. {stem}")
            continue

        track.artist = artist
        track.song = song
        added += 1
    
        if added % 10 == 0:
            print(".", end="", flush=True)

    print(f"\nModified {added} tracks of {possible} possible.")

def browse(store: TrackStore) -> None:
    """ interactively walk the track listing by path. """
    index = TrackIndex(store)

    node = index.get_root()
    up_list = []

    while True:
        choices = [
            Choice("Exit", value="exit", shortcut_key="0"),
        ]

        if up_list:
            choices.append(
                Choice("..", value="up", shortcut_key="1"),
            )

        if node.is_leaf():
            for t in node.list_tracks():
                choices.append(
                    Choice(f"{t.artist} - {t.song}", value="track")
                )
        else:
            for path, n in node.list_nodes().items():
                choices.append(
                    Choice(f"{path}  #[{n.count()}]", value=n)
                )

        result = questionary.select(
            "Browse:",
            choices=choices,
        ).ask()

        if result is None:
            break  # User cancelled (Ctrl+C)
        if result == "exit":
            break
        if isinstance(result, IndexNode):
            up_list.append(node)
            node = result
        elif result == "up":
            node = up_list.pop()

def browse_artist(store: TrackStore) -> None:
    """ interactively browse the track list by artist name """
    index = ArtistIndex.from_store(store)

    node = index.get_root()
    up_list = []

    while True:
        choices = [
            Choice("Exit", value="exit", shortcut_key="0"),
        ]

        if up_list:
            choices.append(
                Choice("..", value="up", shortcut_key="1"),
            )

        if node.is_leaf():
            for t in node.list_tracks():
                md_string = _format_metadata(t.metadata)
                stem = Path(t.path).stem
                choices.append(
                    Choice(f"{stem} :: {md_string}", value="track")
                )
        else:
            for path, n in node.list_nodes().items():
                choices.append(
                    Choice(f"{path}  #[{n.count()}]", value=n)
                )

        result = questionary.select(
            "Browse Artist:",
            choices=choices,
        ).ask()

        if result is None:
            break  # User cancelled (Ctrl+C)
        if result == "exit":
            break
        if isinstance(result, IndexNode):
            up_list.append(node)
            node = result
        elif result == "up":
            node = up_list.pop()

def browse_song(store: TrackStore) -> None:
    """ interactively browse the track list by song name """
    index = SongIndex.from_store(store)

    node = index.get_root()
    up_list = []

    while True:
        choices = [
            Choice("Exit", value="exit", shortcut_key="0"),
        ]

        if up_list:
            choices.append(
                Choice("..", value="up", shortcut_key="1"),
            )

        if node.is_leaf():
            for t in node.list_tracks():
                md_string = _format_metadata(t.metadata)
                stem = Path(t.path).stem
                choices.append(
                    Choice(f"{stem} :: {md_string}", value="track")
                )
        else:
            for path, n in node.list_nodes().items():
                choices.append(
                    Choice(f"{path}  #[{n.count()}]", value=n)
                )

        result = questionary.select(
            "Browse Song:",
            choices=choices,
        ).ask()

        if result is None:
            break  # User cancelled (Ctrl+C)
        if result == "exit":
            break
        if isinstance(result, IndexNode):
            up_list.append(node)
            node = result
        elif result == "up":
            node = up_list.pop()


def _tracks(node: IndexNode) -> Iterable[Track]:
    nodes = [node]

    for n in nodes:
        if n.is_leaf():
            for t in n.list_tracks():
                yield t
        else:
            for nn in n.list_nodes().values():
                nodes.append(nn)

def _bulk_edit_artist(store: TrackStore, node: IndexNode) -> None:
    """Edit all tracks for an artist (interactive)."""

    all_tracks = list(_tracks(node))
    first_artist = all_tracks[0].artist

    questionary.print(f"Editing [{first_artist}]")
    choices = [
        Choice("Exit", value="exit", shortcut_key="0"),
        Choice("Edit", value="edit"),
        Choice("Ungroup-Uncomma", value="uncomma"),
    ]

    result = questionary.select(
        "Edit how:",
        choices=choices,
    ).ask()

    if result is None or result == "exit":
        return  # User cancelled (Ctrl+C)
 
    if result == "edit":
        text = questionary.text("Edit", default=first_artist).ask()
        if text:
            for t in all_tracks:
                print(f"Changing {t.artist} to {text}")
                t.artist = text
                store.add(t)
    elif result == "uncomma":
        feature = None
        first_artist = clean_artist(first_artist)
        if '&' in first_artist:
            prefix = first_artist.split("&", 1)
            feature = prefix[1]
            prefix = prefix[0]
        else:
            prefix = first_artist

        prefix = _uncomma_artist(prefix) or prefix

        for t in all_tracks:
            print(f"Changing {t.artist} to {prefix}")
            t.artist = prefix
            if feature:
                t.metadata["feature"] = feature
            store.add(t)

def _edit_track_details(store: TrackStore, track: Track) -> None:
    """Edit a single track's details (interactive)."""
    track_stem = Path(track.path).stem
    questionary.print(f"Editing [{track_stem}] -> '{track.artist} - {track.song}'")
    choices = [
        Choice("Exit", value="exit", shortcut_key="0"),
        Choice("Swap", value="swap"),
        Choice("Edit Artist", value="artist"),
        Choice("Edit Song", value="song"),
        Choice("Unset Artist", value="unset-artist"),
    ]

    result = questionary.select(
        "Edit how:",
        choices=choices,
    ).ask()

    if result is None:
        return  # User cancelled (Ctrl+C)
    if result == "exit":
        return

    if result == "swap":
        track.artist, track.song = track.song, track.artist
        store.add(track)
    elif result == "unset-artist":
        track.artist = ""
        store.add(track)
    elif result == "artist":
        new_artist = questionary.text("New Artist:", default=track.artist).ask()
        if new_artist:
            track.artist = new_artist
            store.add(track)
    elif result == "song":
        new_song = questionary.text("New Song:", default=track.song).ask()
        if new_song:
            track.song = new_song
            store.add(track)

def browse_fixup(store: TrackStore) -> None:
    """Browse and fix up artists with 5 or fewer tracks."""
    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    to_review = _tracks_to_review(store, review_state)

    index = ArtistIndex(to_review)

    root = index.get_root()
    node = root
    up_list = []

    while True:
        choices = [
            Choice("Exit", value="exit", shortcut_key="0"),
        ]

        if up_list:
            choices.append(
                Choice("..", value="up", shortcut_key="1"),
            )

        if node.is_leaf():
            for t in node.list_tracks():
                stem = Path(t.path).stem
                choices.append(
                    Choice(f"{t.artist} - {stem}", value=t)
                )
        else:
            for path, n in node.list_nodes().items():
                choices.append(
                    Choice(f"{path}  #[{n.count()}]", value=n)
                )

        result = questionary.select(
            "Browse Artist:",
            choices=choices,
        ).ask()

        if result is None:
            break  # User cancelled (Ctrl+C)
        if result == "exit":
            break
        if isinstance(result, IndexNode):
            up_list.append(node)
            node = result
        elif isinstance(result, Track):
           _edit_track_details(store, result)
        elif result == "up":
            node = up_list.pop()

def fix_artist(store: TrackStore) -> None:
    """Browse all artist for fixup."""
    index = ArtistIndex.from_store(store)

    root = index.get_root()
    node = root
    up_list = []

    while True:
        choices = [
            Choice("Exit", value="exit", shortcut_key="0"),
        ]

        if up_list:
            choices.append(
                Choice("..", value="up", shortcut_key="1"),
            )

        depth = len(up_list)

        if not node.is_leaf():
            for path, n in node.list_nodes().items():
                choices.append(
                    Choice(f"{path}  #[{n.count()}]", value=n)
                )

        result = questionary.select(
            "Browse Artist:",
            choices=choices,
        ).ask()

        if result is None:
            break  # User cancelled (Ctrl+C)
        if result == "exit":
            break
        if isinstance(result, IndexNode):
            if depth >= 1:
                _bulk_edit_artist(store, result)
            else:
                up_list.append(node)
                node = result
        elif result == "up":
            node = up_list.pop()

def fix_unknown(store: TrackStore) -> None:
    """ For tracks where no artist was detected, try alternate extractions of the track path. """

    aindex = ArtistIndex.from_store(store)
    artists = aindex.count_artists(low_bound=5)

    unknown_tracks = [t for t in store.all() if t.artist.lower() in ["unknown", ""] ]
    questionary.print(f"total unknown {len(unknown_tracks)}")

    no_split = 0
    simple_split = 0
    complex_split = 0
    comma_split = 0
    id_split = 0
    space_split = 0

    success = 0

    for t in unknown_tracks:
        song = t.song
        song = song.replace("_", " ")
        if ' - ' in song:
            split = song.split(' - ')
            if len(split) == 2:
                clean_s1 = clean_artist(split[0])
                if clean_s1 in artists:
                    t.artist = split[0]
                    t.song = split[1]
                    store.add(t)
                    success += 1
                else:
                    simple_split += 1
            if len(split) == 3:
                # identifier - artist - song
                clean_s1 = clean_artist(split[1])
                if clean_s1 in artists:
                    t.artist = split[1]
                    t.song = split[2]
                    store.add(t)
                    success += 1
                else:
                    id_split += 1
            if len(split) == 4:
                # identifier - artist - song
                clean_s1 = clean_artist(split[2])
                if clean_s1 in artists:
                    t.artist = split[2]
                    t.song = split[3]
                    store.add(t)
                    success += 1
                else:
                    id_split += 1
            if len(split) > 4:
                print(f"Complex [{len(split)}] {song}")
                complex_split += 1
        elif '-' in song:
            split = song.split('-')
            if len(split) == 2:
                clean_s1 = clean_artist(split[0])
                if clean_s1 in artists:
                    t.artist = split[0]
                    t.song = split[1]
                    store.add(t)
                    success += 1
                else:
                    simple_split += 1
            if len(split) == 3:
                # identifier - artist - song
                clean_s1 = clean_artist(split[1])
                if clean_s1 in artists:
                    t.artist = split[1]
                    t.song = split[2]
                    store.add(t)
                    success += 1
                else:
                    id_split += 1
            if len(split) > 3:
                print(f"Complex [{len(split)}] {split}")
                complex_split += 1
        elif "  " in song:
            split = song.split('  ')
            if len(split) == 2:
                clean_s0 = clean_artist(split[0])
                clean_s1 = clean_artist(split[1])

                if clean_s0 in artists:
                    t.artist = split[0]
                    t.song = split[1]
                    store.add(t)
                    success += 1
                elif clean_s1 in artists:
                    t.artist = split[1]
                    t.song = split[0]
                    store.add(t)
                    success += 1
                else:
                    space_split += 1
            else:
                print(f"Complex space: {song}")
        elif ',' in song:
            split = song.split(',')
            if len(split) == 2:
                clean_s0 = clean_artist(split[0])
                if clean_s0 in artists:
                    t.artist = split[0]
                    t.song = split[1]
                    store.add(t)
                    success += 1
                else:
                    comma_split += 1
            else:
                #  song name with many commas
                complex_split += 1
                #print(f"Complex comma: {song}")
        else:
            split = song.split(" ")
            if clean_artist(split[0]) in artists:
                artist = split[0]
                song = " ".join(split[1:])
                print(f"split 1 {song} -- {artist}")
                
            elif clean_artist(split[-1]) in artists:
                artist = split[-1]
                song = " ".join(split[:-1])
                print(f"split neg-1 {song} -- {artist}")
            # either just a song, or artist[ ]song 
            # print(f"No Split [{song}]")
            no_split += 1

    questionary.print(f"Success {success}, simple {simple_split}, id {id_split}, Complex {complex_split}, Comma {comma_split}. No Split {no_split}")

def _tracks_to_review(store: TrackStore, review_state: ReviewState) -> list[Track]:
    """ Part of manual track review - build a list of unreviewed tracks.
        Focuses on artists with few total tracks (which are often miscategorized) 
    """
    index = ArtistIndex.from_store(store)
    tracks: list[Track] = []
    for letter, top_node in index.get_root().list_nodes().items():
        for artist, artist_node in top_node.list_nodes().items():
            if artist_node.count() > 5:
                continue
            for song, song_node in artist_node.list_nodes().items():
                tracks.extend(song_node.list_tracks())

    tracks = [t for t in tracks if review_state.get(t.path) != "ok"]
    return tracks


def _auto_clean_artist(artist: str) -> tuple[str, str]:
    artist = clean_artist(artist)
    if '&' in artist:
        prefix, feature = artist.split('&', 1)
    else:
        prefix, feature = artist, None

    prefix = clean_artist(prefix)
    prefix = _uncomma_artist(prefix) or prefix
    return prefix, feature


# Catalog-id prefix that sometimes leads an ID3 tag, e.g. "SFDU11-06 - ".
_CATALOG_PREFIX_RE = re.compile(r"^[A-Za-z]{2,}\d+[-\d]*\s*-\s*")
# Karaoke noise that appears inside artist / tag strings.
_ARTIST_NOISE_RE = re.compile(
    r"\(.*?\)|\bw[\s-]?o?b?gv\b|\bwvocals?\b|\bmultiplex\b|\bvr\b", re.IGNORECASE
)


def _artist_tokens(s: str) -> frozenset:
    """Order-insensitive, normalized token set for comparing an artist to a tag.

    Strips a leading catalog id and karaoke noise, and treats &/and/feat as
    joiners, so 'Murray, Pete' == 'Pete Murray' and 'A & B' == 'A And B'."""
    s = _CATALOG_PREFIX_RE.sub("", s or "")
    s = _ARTIST_NOISE_RE.sub(" ", s)
    s = re.sub(r"\b(feat|ft|featuring|and)\b", " ", s, flags=re.IGNORECASE).replace("&", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return frozenset(w for w in s.split() if w)


def auto_ok_from_tags(store: TrackStore) -> None:
    """Mark review-eligible tracks 'ok' when the parsed artist is corroborated
    by the MP3's ID3 tag_artist.

    Non-destructive: only writes the review-state flag (reversible by deleting
    review-state.json). Tracks with no tag, or a tag whose tokens genuinely
    differ, are left untouched for manual review. Matching is order-insensitive
    and ignores catalog prefixes, karaoke suffixes, and &/and/feat variations."""
    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    okayed = 0
    for t in _tracks_to_review(store, review_state):
        tag = t.metadata.get("tag_artist")
        if not tag or tag == "<not-found>":
            continue
        a = _artist_tokens(t.artist)
        g = _artist_tokens(tag)
        if a and a == g:
            review_state.set(t.path, "ok")
            okayed += 1

    review_state.save(_REVIEW_STATE_PATH)
    questionary.print(f"Auto-ok'd {okayed} tracks corroborated by their ID3 tag")


def swap_from_tags(store: TrackStore) -> None:
    """Fix reversed artist/song parses using the ID3 tag as evidence.

    When tag_artist matches the SONG field (not the artist), and that song
    value is a known artist (>=3 tracks) while the current artist is not, the
    parser reversed the two -- swap them and mark reviewed.

    The known-artist check guards against mislabeled tags (e.g. Disney tracks
    whose ID3 'artist' is actually the song title); those are left for manual
    review rather than swapped."""
    index = ArtistIndex.from_store(store)
    known = set(index.count_artists(low_bound=3))

    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    swapped = 0
    for t in _tracks_to_review(store, review_state):
        tag = t.metadata.get("tag_artist")
        if not tag or tag == "<not-found>":
            continue
        tg, ar, so = _artist_tokens(tag), _artist_tokens(t.artist), _artist_tokens(t.song)
        if not tg or not so or tg != so or tg == ar:
            continue
        # Tag points at the song field. Only trust it if that value is a real
        # artist elsewhere and the current artist is not (guards bad tags).
        if clean_artist(t.song) in known and clean_artist(t.artist) not in known:
            t.artist, t.song = t.song, t.artist
            store.add(t)
            review_state.set(t.path, "ok")
            swapped += 1

    review_state.save(_REVIEW_STATE_PATH)
    questionary.print(f"Swapped {swapped} reversed artist/song pairs (corroborated by ID3 tag)")


_MB_USER_AGENT = "song-sorter/1.0 ( https://github.com/Spikelite/song-sorter )"
_MB_URL = "https://musicbrainz.org/ws/2/recording"
_MB_STRONG = 88   # both sims >= this: confident, low-divergence match -> ok
_MB_WEAK = 70     # both sims >= this (but not strong): a record exists, diverges -> flag


def _is_online(host: str = "musicbrainz.org", port: int = 443, timeout: float = 3.0) -> bool:
    """Quick reachability probe so online features self-skip when offline."""
    try:
        socket.create_connection((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def _mb_escape(s: str) -> str:
    # We wrap the value in a Lucene quoted phrase, so just drop " and \.
    return s.replace("\\", " ").replace('"', " ").strip()


def _mb_credit_name(artist_credit) -> str:
    """Flatten a MusicBrainz artist-credit list into a display string."""
    parts = []
    for c in artist_credit or []:
        if isinstance(c, dict):
            parts.append(c.get("name") or c.get("artist", {}).get("name", ""))
            parts.append(c.get("joinphrase", ""))
    return "".join(parts).strip()


def _mb_search(artist: str, title: str) -> list:
    """Query MusicBrainz for recordings matching artist+title.

    Raises on network error so the caller can handle connectivity loss."""
    q = f'artist:"{_mb_escape(artist)}" AND recording:"{_mb_escape(title)}"'
    url = _MB_URL + "?" + urllib.parse.urlencode({"query": q, "fmt": "json", "limit": "5"})
    req = urllib.request.Request(url, headers={"User-Agent": _MB_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp).get("recordings", [])


def _mb_best(recordings: list, our_artist: str, our_title: str) -> tuple:
    """Return the best (artist_sim, title_sim, mb_artist, mb_title) match.

    token_sort_ratio is order-insensitive, so 'Murray, Pete' scores 100 against
    'Pete Murray'."""
    best = (0.0, 0.0, "", "")
    for r in recordings:
        mb_title = r.get("title", "")
        mb_artist = _mb_credit_name(r.get("artist-credit"))
        a = rapidfuzz.fuzz.token_sort_ratio(our_artist, mb_artist)
        t = rapidfuzz.fuzz.token_sort_ratio(our_title, mb_title)
        if a + t > best[0] + best[1]:
            best = (a, t, mb_artist, mb_title)
    return best


def musicbrainz_lookup(store: TrackStore) -> None:
    """(online) Corroborate remaining review tracks against MusicBrainz.

    For each unreviewed track from a thin artist, query MusicBrainz by
    artist+title (and the swapped orientation, to catch reversed parses):
      - confident match (both fields very similar) -> mark ok
      - only the swapped orientation matches -> swap artist/song, mark ok
      - a record exists but diverges a lot from our data -> flag in metadata
        (mb_artist/mb_title) for review; do NOT auto-apply
      - no match -> left for manual review

    Results are cached (metadata['mb_checked']) so it is resumable, and it is
    OFFLINE-SAFE: skips cleanly with no internet and never aborts hard."""
    if not _is_online():
        questionary.print("MusicBrainz lookup needs internet -- skipped (offline).")
        return

    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)
    todo = [t for t in _tracks_to_review(store, review_state)
            if not t.metadata.get("mb_checked")]
    if not todo:
        questionary.print("No un-checked review tracks for MusicBrainz.")
        return

    okd = swapped = flagged = nomatch = 0
    pair_cache: dict[tuple, tuple] = {}
    consecutive_fail = 0
    last_save = time.monotonic()

    try:
        for t in tqdm(todo, desc="MusicBrainz", unit="track"):
            a, s = t.artist.strip(), t.song.strip()
            if not a or not s:
                continue
            key = (a.lower(), s.lower())
            try:
                if key in pair_cache:
                    na, nt, ma, mt, was_swap = pair_cache[key]
                else:
                    res = _mb_search(a, s)
                    time.sleep(1.1)  # MusicBrainz: ~1 req/sec
                    na, nt, ma, mt = _mb_best(res, a, s)
                    was_swap = False
                    if not (na >= _MB_STRONG and nt >= _MB_STRONG):
                        # Try the reversed orientation to catch swapped parses.
                        res2 = _mb_search(s, a)
                        time.sleep(1.1)
                        sa, st_, sma, smt = _mb_best(res2, s, a)
                        if sa >= _MB_STRONG and st_ >= _MB_STRONG:
                            na, nt, ma, mt, was_swap = sa, st_, sma, smt, True
                    pair_cache[key] = (na, nt, ma, mt, was_swap)
                consecutive_fail = 0
            except Exception:
                consecutive_fail += 1
                if consecutive_fail >= 5:
                    tqdm.write("Lost connection to MusicBrainz -- saving progress and stopping.")
                    break
                continue  # transient error: leave un-checked, retry next run

            t.metadata["mb_checked"] = "1"
            if na >= _MB_STRONG and nt >= _MB_STRONG:
                if was_swap:
                    t.artist, t.song = t.song, t.artist
                    store.add(t)
                    swapped += 1
                review_state.set(t.path, "ok")
                t.metadata["mb_match"] = "swap" if was_swap else "ok"
                okd += 1
            elif na >= _MB_WEAK and nt >= _MB_WEAK:
                # A recording exists but diverges -- keep for review, don't apply.
                t.metadata["mb_match"] = "flag"
                t.metadata["mb_artist"] = ma
                t.metadata["mb_title"] = mt
                flagged += 1
            else:
                t.metadata["mb_match"] = "none"
                nomatch += 1

            if time.monotonic() - last_save >= 60:
                store.save(_CACHE_PATH)
                review_state.save(_REVIEW_STATE_PATH)
                last_save = time.monotonic()
    finally:
        store.save(_CACHE_PATH)
        review_state.save(_REVIEW_STATE_PATH)

    questionary.print(
        f"MusicBrainz: ok'd {okd} (incl {swapped} swapped), "
        f"flagged {flagged}, no-match {nomatch}")


def review_mode(store: TrackStore) -> None:
    """Sequential review of tracks from artists with 5 or fewer tracks."""
    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    tracks = _tracks_to_review(store, review_state)
    if not tracks:
        questionary.print("No tracks from artists with 5 or fewer tracks.")
        return

    session_cache: dict[str, str] = {}

    choices = [
        Choice("(exit)", value="exit", shortcut_key="0"),
        Choice("(ok)", value="ok"),
        Choice("(swap)", value="swap"),
        Choice("(edit)", value="edit"),
    ]

    i = 0
    while i < len(tracks):
        track = tracks[i]
        key = f"{track.artist} - {track.song}".lower()

        # Session cache: apply cached choice and advance
        cached = session_cache.get(key)
        if cached is not None:
            result = cached
        else:
            # Present track and prompt
            stem = Path(track.path).stem
            questionary.print(f"[{i + 1}/{len(tracks)}] {stem} -> '{track.artist} - {track.song}'")

            art_clean = clean_artist(track.artist)
            track_choices = None
            if '&' in art_clean or ',' in art_clean:
                prefix, feature = _auto_clean_artist(track.artist)
                track_choices = list(choices)
                track_choices.append(
                    Choice(f"auto [{prefix}]", value="auto-clean")
                )
            result = questionary.select("Edit this track?", choices=track_choices or choices).ask()

        if result is None or result == "exit":
            review_state.save(_REVIEW_STATE_PATH)
            break

        session_cache[key] = result
        if result == "ok":
            review_state.set(track.path, "ok")
            review_state.save(_REVIEW_STATE_PATH)
        elif result == "swap":
            track.artist, track.song = track.song, track.artist
            store.add(track)
        elif result == "auto-clean":
            # ungroup, uncomma, and clean the artist
            prefix, feature = _auto_clean_artist(track.artist)
            track.artist = prefix
            if feature:
                track.metadata["feature"] = feature.strip()
            store.add(track)
        elif result == "edit":
            _edit_track_details(store, track)

        i += 1

def report_track_count(store: TrackStore) -> None:
    """ List a summary of track information """
    all_songs = set(f"{clean_artist(t.artist)} - {clean_song(t.song)}" for t in store.all()
        if t.artist.lower() not in ["unknown", ""])
    questionary.print(f"Distinct count: {len(all_songs)} / {len(store.all())}")

def _best_track(tracks: list[Track]) -> Track:
    """ Pick one track to represent an artist-song pair."""
    
    # Basic guess: return the track with the largest MP3 size in metadata.
    def _mp3_size(t: Track) -> int:
        raw = t.metadata.get("mp3_size")
        if raw is None:
            return 0
        try:
            return int(raw)
        except ValueError:
            return 0

    return max(tracks, key=_mp3_size)

def tracks_to_keep(store: TrackStore) -> None:
    index = ArtistIndex.from_store(store)

    node_set = [index.get_root()]

    output_path = "/tmp/output"

    while (node_set):
        n = node_set.pop()

        if n.is_leaf():
            best_track = _best_track(n.list_tracks())
            stem = Path(best_track.path).stem
            artist = clean_artist(best_track.artist)
            if artist in ["", "unknown"]:
                continue
            for exn in best_track.file_types:
                prefix = artist[0]
                if not re.match("[a-z]", artist[0:1]):
                    prefix = "#"
                to_path = Path(f"{output_path}/{prefix}/{artist}/{stem}.{exn}")
                print(to_path)
                if not to_path.exists():
                    # final step, copy from track.path to (output) /
                    to_path.parent.mkdir(parents=True, exist_ok=True)
                    source_path = Path(best_track.path)
                    shutil.copy2(source_path, to_path)

        else:
            node_set.extend(n.list_nodes().values())


def clean(store: TrackStore) -> None:
    """ Remove frequent karaoke descriptors """
    indicators = [
        "(Wobgv)",
        "(Wbgv)",
        "(Bgv)",
        "wBGV",
        "Wobgv",
        "(Wvocals)",
        "Wvocals",
        "wvocals",
        "Wvocal",
        "wvocal",
        "W-Vocal",
        "W-vocal",
        "w-vocal",
        " w vocal",
        "vocals",
        "Wmusic",
        "wmusic",
        "()",
        "(duet)",
        "(Duet)",
        "(Solo)",
        "(solo)",
        "Music Only",
        "(Instrumental)",
        "(Explicit Implied)",
        "(Explicit)",
        "(Clean)",
        "Christmas-",
    ]
    song_indicator = [
        "( Instrumental )",
        "(Explicit Implied)",
        "(Explicit Version)",
        "(Explicit)",
        "(Clean)",
        "(Clean Version)",
        "(Duet)",
        "(Solo)",
        "(Solo Male)", 
        "(Instrumental version)",
        "(Music Only)",
        "(Wobgv)",
        "W-Vocal",
        "Wvocals",
        "Wvocal",
        "wvocal",
        "Multiplex",
    ]
    modify_count = 0
    for t in store.all():
        for ind in indicators:
            if ind in t.artist:
                modify_count+=1
                t.metadata['style'] = ind
                t.artist = t.artist.replace(ind, "").strip()
                break
        for ind in song_indicator:
            if ind in t.song:
                modify_count+=1
                t.metadata['style'] = ind
                t.song = t.song.replace(ind, "").strip()

    questionary.print(f"Modified {modify_count}")

    # Karaoke track numbers parsed as the artist: filenames like
    # "EZH-31 - 04 - Milkshake" leave artist="04". A bare 1-2 digit value is a
    # disc track position, never a real artist. We deliberately do NOT touch
    # 3+ digit names (911, 411, 112 are real bands), letters-with-digits
    # (Blink-182, Maroon-5), or song titles (hundreds legitimately start with a
    # number). Cleared artists become "Unknown" for later recovery via Fix-unknown.
    cleared = 0
    for t in store.all():
        artist = t.artist.strip()
        if re.fullmatch(r"\d{1,2}", artist):
            # Preserve what we prune: the disc track number, and the catalog id
            # (parts[0] of the original filename, dropped at parse time). Both
            # are useful later (e.g. catalog-number -> real-artist lookup).
            t.metadata["track_no"] = artist
            stem_parts = Path(t.path).stem.split(" - ")
            if len(stem_parts) >= 3:
                t.metadata["catalog_id"] = stem_parts[0].strip()
            t.artist = "Unknown"
            cleared += 1
    questionary.print(f"Cleared {cleared} track-number artists")


# Karaoke descriptor suffixes that appear inside ID3 artist tags, e.g.
# "Zac Brown Band (Wbgv)". Stripped before a tag is used as an artist.
_TAG_SUFFIX_RE = re.compile(
    r"\s*\((?:w[\s-]?o?b?gv|bgv|w[\s-]?vocals?|wvocals?|duet|solo|instrumental|music only)\)\s*",
    re.IGNORECASE,
)


def fill_artist_from_tags(store: TrackStore) -> None:
    """Fill Unknown/empty artists from the MP3's ID3 tag_artist, conservatively.

    Only clean, real-looking tags are applied, and only where there is no
    artist already (a real artist is never overwritten). Ambiguous tags are
    NOT auto-filled -- they are kept and flagged for later manual / automated
    review so no data is lost:
      - bare numbers (e.g. 311, 1975): could be a real band or junk
      - catalog-id-shaped (e.g. PS1254, dsny01)
    Filled artists are marked metadata['artist_from']='tag'; flagged ones get
    metadata['artist_review'] (the candidate) + ['artist_review_reason']."""
    filled = 0
    flagged = 0
    for t in store.all():
        if t.artist.strip().lower() not in ("unknown", ""):
            continue
        raw = t.metadata.get("tag_artist")
        if not raw or raw == "<not-found>":
            continue
        cand = _TAG_SUFFIX_RE.sub("", raw).strip()
        if not cand:
            continue

        reason = None
        if re.fullmatch(r"\d+", cand):
            reason = "numeric"
        elif re.fullmatch(r"[A-Za-z]+\d{2,}", cand):
            reason = "catalog-id"

        if reason:
            # Keep the candidate and flag it; do not auto-fill an ambiguous tag.
            t.metadata["artist_review"] = cand
            t.metadata["artist_review_reason"] = reason
            flagged += 1
        else:
            t.artist = cand
            t.metadata["artist_from"] = "tag"
            filled += 1

    questionary.print(f"Filled {filled} artists from tags; flagged {flagged} for review")


# Prefix is comma-free so this only matches a SINGLE name ending in an article.
_TRAILING_ARTICLE_RE = re.compile(r"^([^,]*\S),\s*(the|a|an)\s*$", re.IGNORECASE)


def _fix_trailing_article(s: str) -> str:
    """'Beatles, The' -> 'The Beatles'. Conservative: only fires for a single
    clean name/title -- it must have no other comma and no dash. That leaves
    mashed artist-song blobs ('Dion, Celine-Power Of Love, The', 'Ace Of
    Base-Sign, The') and multi-comma names ('Earth, Wind & Fire') untouched, at
    the cost of skipping a few real hyphenated names (handled later by MB)."""
    m = _TRAILING_ARTICLE_RE.match(s.strip())
    if m and "-" not in m.group(1):
        return f"{m.group(2).capitalize()} {m.group(1)}"
    return s


def trailing_article(store: TrackStore) -> None:
    """Move trailing articles to the front in artist and song fields, e.g.
    'Models, The' -> 'The Models', 'Whole New World, A' -> 'A Whole New World'.

    Deterministic and safe: only triggers when a field ENDS in ', The/A/An'.
    Best run after Clean, which strips karaoke suffixes that would otherwise
    sit between the name and its trailing article."""
    fixed = 0
    for t in store.all():
        new_a = _fix_trailing_article(t.artist)
        new_s = _fix_trailing_article(t.song)
        if new_a != t.artist or new_s != t.song:
            t.artist, t.song = new_a, new_s
            fixed += 1
    questionary.print(f"Fixed {fixed} trailing articles")


def standard_artist(store: TrackStore) -> None:
    # swapping "artist, name" to "name artist"
    index = ArtistIndex.from_store(store)
    artists = set(index.count_artists())

    swap_count = 0
    for t in store.all():
        clean_name = clean_artist(t.artist)
        if "," in clean_name:
            swapped = _uncomma_artist(clean_name)
            if swapped in artists:
                t.artist = swapped
                swap_count += 1
    questionary.print(f"Swapped {swap_count}")


def fuzz_artist(store: TrackStore) -> None:
    # One letter off like "ac dc -> ac-dc"

    index = ArtistIndex.from_store(store)

    for letter, top_node in index.get_root().list_nodes().items():
        if letter == "":
            continue

        artist_set = set(top_node.list_nodes().keys())
        letter_nodes = top_node.list_nodes()

        for artist, anode in letter_nodes.items():
            if anode.count() > 5:
                # assume popular spellings are correct
                continue

            # n-squared
            for other_artist in artist_set:
                if other_artist == artist:
                    continue
                ratio = rapidfuzz.fuzz.ratio(artist, other_artist)
                if ratio >= 90.0:
                    this_count = anode.count()
                    other_count = letter_nodes[other_artist].count()

                    # break ties consistently. hash?
                    if this_count < other_count or (this_count == other_count and hash(anode) < hash(letter_nodes[other_artist])):
                        questionary.print(f"Fuzz match {ratio:.1f} for {artist}#{this_count} to {other_artist}#{other_count}")
                        # now rename
                        to_rename = [anode]
                        while len(to_rename) > 0:
                            node = to_rename.pop()
                            if node.is_leaf():
                                for t in node.list_tracks():
                                    t.artist = other_artist
                            else:
                                for n in node.list_nodes().values():
                                    to_rename.append(n)
                        break


def fuzz_song(store: TrackStore) -> None:
    """ Fuzz songs for the same artist """
    index = ArtistIndex.from_store(store)

    for letter, top_node in index.get_root().list_nodes().items():
        if letter == "":
            continue

        for artist, anode in top_node.list_nodes().items():
            song_nodes = anode.list_nodes()
            song_set = set(song_nodes.keys())
            
            for song, snode in anode.list_nodes().items():
                # n-squared
                for other_song in song_set:
                    if other_song == song:
                        continue

                    ratio = rapidfuzz.fuzz.ratio(song, other_song)    
                    if ratio >= 85.0:
                        this_count = snode.count()
                        other_count = song_nodes[other_song].count()

                        # break ties consistently. hash?
                        if this_count < other_count or (this_count == other_count and hash(snode) < hash(song_nodes[other_song])):
                            questionary.print(f"Fuzz match {ratio:.1f} for {song}#{this_count} to {other_song}#{other_count}")
                        
                            # now rename
                            to_rename = [snode]
                            while len(to_rename) > 0:
                                 node = to_rename.pop()
                                 if node.is_leaf():
                                     for t in node.list_tracks():
                                         t.song = other_song
                                         store.add(t)
                                 else:
                                     for n in node.list_nodes().values():
                                         to_rename.append(n)
                            break


def find_swapped(store: TrackStore) -> None:
    """ If the song name for a track is actually a popular artist name, swap! """

    index = ArtistIndex.from_store(store)

    likely_artists = set(index.count_artists(low_bound=3))

    questionary.print(f"Looking at {len(likely_artists)} artists")

    single_tracks = index.single_artists()
    questionary.print(f"with {len(single_tracks)} options")

    total_swaps = 0
    swap_folder : dict[str, list[Track]] = {}
    for t in single_tracks:
        song_as_artist = clean_artist(t.song)
        artist_as_artist = clean_artist(t.artist)
        if song_as_artist in likely_artists and artist_as_artist not in likely_artists:
            folder = Path(t.path).parent
            if folder not in swap_folder:
                swap_folder[folder] = []
            swap_folder[folder].append(t)
            total_swaps += 1

    questionary.print(f"Available {total_swaps} in {len(swap_folder)}")
    for k, v in swap_folder.items():
        if len(v) < 3:
            pass
            # print(f"{len(v)} : {k} : \t{v[0].artist}-{v[0].song}")
        else:
            pass
            # print(f"{len(v)} : {k}")

    total_swaps = 0
    for k, v in swap_folder.items():
        if len(v) > 3:
            for trk in v:
                total_swaps += 1
                old_artist = trk.artist
                old_song = trk.song
                trk.song = old_artist
                trk.artist = old_song
    questionary.print(f"Completed Swaps {total_swaps}")


def ungroup_artist(store: TrackStore) -> None:
    """ take out the additional actors from `artist & someone & else` """
    index = ArtistIndex.from_store(store)

    likely_artists = set(index.count_artists())

    questionary.print(f"Looking at {len(likely_artists)} artists")

    single_tracks = index.single_artists(max_songs=9)
    
    maybe_groups = [t for t in single_tracks if '&' in t.artist]
    questionary.print(f"with {len(maybe_groups)} options")

    total_swaps = 0
    for t in maybe_groups:

        prefix, feature = t.artist.split('&', 1)
        prefix = clean_artist(prefix)
        prefix = _uncomma_artist(prefix) or prefix

        if prefix in likely_artists:
            total_swaps += 1
            t.artist = prefix
            t.metadata["feature"] = feature.strip()
    questionary.print(f"Completed Swaps {total_swaps}")


def run_interactive(store: TrackStore) -> None:
    """Run the interactive main menu loop."""
    options = [
        "search",
        "detail",
        "refresh",
        "browse",
        "artist",
        "song",
        "final-final",
        "list",
        "review",
        "tag-review",
        "tag-swap",
        "musicbrainz",
        "fixup",
        "fix-artist",
        "fix-unknown",
        "all-clean",
        "clean",
        "tag-fill",
        "unswap",
        "uncomma",
        "trailing-article",
        "ungroup",
        "fuzz",
        "fuzz_song",
        "exit",
    ]
    online = {"musicbrainz"}  # options that need internet, flagged in the menu
    choices = [
        Choice(o.capitalize() + (" (online)" if o in online else ""), value=o)
        for o in options
    ]

    while True:
        result = questionary.select(
            "Select an option:",
            choices=choices,
        ).ask()

        if result is None:
            break  # User cancelled (Ctrl+C)
        if result == "exit":
            break
        if result == "search":
            path = import_path(_default_scan_dir(store))
            if path is not None:
                add_tracks(store, path)
        elif result == "detail":
            path = import_path(_default_scan_dir(store))
            if path is not None:
                add_details(store, path)
        elif result == "refresh":
            path = import_path(_default_scan_dir(store))
            if path is not None:
                refresh_names(store, path)
        elif result == "browse":
            browse(store)
        elif result == "artist":
            browse_artist(store)
        elif result == "song":
            browse_song(store)
        elif result == "final-final":
            tracks_to_keep(store)
        elif result == "list":
            report_track_count(store)
        elif result == "review":
            review_mode(store)
        elif result == "tag-review":
            auto_ok_from_tags(store)
        elif result == "tag-swap":
            swap_from_tags(store)
        elif result == "musicbrainz":
            musicbrainz_lookup(store)
        elif result == "fixup":
            browse_fixup(store)
        elif result == "fix-artist":
            fix_artist(store)
        elif result == "fix-unknown":
            fix_unknown(store)
        elif result == "all-clean":
            clean(store)
            trailing_article(store)
            fill_artist_from_tags(store)
            # find_swapped(store)
            standard_artist(store)
            ungroup_artist(store)
            fuzz_artist(store)
            fuzz_song(store)
        elif result == "tag-fill":
            fill_artist_from_tags(store)
        elif result == "clean":
            clean(store)
        elif result == "unswap":
            find_swapped(store)
        elif result == "uncomma":
            standard_artist(store)
        elif result == "trailing-article":
            trailing_article(store)
        elif result == "ungroup":
            ungroup_artist(store)
        elif result == "fuzz":
            fuzz_artist(store)
        elif result == "fuzz_song":
            fuzz_song(store)


def main() -> None:
    """Entry point: run interactive mode."""
    store = TrackStore()
    try:
        store.load(_CACHE_PATH)
    except ValueError:
        pass  # Invalid cache version; start with empty store

    try:
        run_interactive(store)
    finally:
        store.save(_CACHE_PATH)
    print("Goodbye.")


if __name__ == "__main__":
    main()
