"""song-sorter: Interactive CLI for organizing karaoke track libraries."""

from __future__ import annotations

import collections
import json
import os
import re
import shutil
import socket
import time
import unicodedata
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
from track_index import (ArtistIndex, SongIndex, TrackIndex, IndexNode,
                         clean_artist, clean_song, fragments_match_title,
                         is_catalog_segment, majority_raw, parse_artist_song,
                         rejoin_artist, safe_folder, split_stem,
                         strip_artist_echo, uncomma_artist)
from track_inspect import track_details, audio_file, read_key_tag
from review_state import ReviewState
import key_detect
import key_online


_CACHE_PATH = Path(__file__).parent / ".cache" / "song-sorter" / "cache.json"
_REVIEW_STATE_PATH = _CACHE_PATH.parent / "review-state.json"
_CONFIG_PATH = _CACHE_PATH.parent / "config.json"
_RESOLUTIONS_PATH = _CACHE_PATH.parent / "resolutions.json"
_ARTIST_ALIASES_PATH = _CACHE_PATH.parent / "artist-aliases.json"
_KEY_OVERRIDES_PATH = _CACHE_PATH.parent / "key-overrides.json"


def _load_config() -> dict:
    """Load persisted settings (e.g. output_path); returns {} if none/invalid."""
    p = Path(_CONFIG_PATH)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (ValueError, OSError):
        return {}


def _save_config(cfg: dict) -> None:
    """Persist settings to the config file."""
    p = Path(_CONFIG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)


def _load_aliases() -> dict:
    """The artist alias map (variant -> canonical) from artist-aliases.json;
    {} when the file is absent or unreadable. Shared by Unify, Uncomma, and
    Search so a curated rename is enforced everywhere new names can enter."""
    try:
        with open(_ARTIST_ALIASES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {}
    return {k: v for k, v in data.get("aliases", {}).items() if v and k != v}


def _load_key_overrides() -> dict:
    """Curated musical-key overrides -- the source of truth for Key-detect.

    key-overrides.json maps ``"<artist> - <song>"`` (matched case-insensitively
    on the track's current fields) to a key string in any form normalize_key
    accepts (``"A minor"``, ``"Am"``, ``"8A"``...). {} when absent/unreadable."""
    try:
        with open(_KEY_OVERRIDES_PATH, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError):
        return {}
    src = data.get("overrides", data) if isinstance(data, dict) else {}
    return {str(k).strip().lower(): v for k, v in src.items() if v}


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
    aliases = _load_aliases()  # curated renames self-heal at the door: a new
    #                            'Jones, Tom' rip lands as 'Tom Jones'

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
        artist, song = parse_artist_song(stem)
        if "song-artist" in p.parts:
            # some parts of the path are reversed
            artist, song = song, artist
        artist = aliases.get(artist, artist)

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
                # Compact catalog stems (no ' - ') the initial scan stored as
                # 'Unknown - <whole stem>': recover them via the shared parser.
                # Only overwrite tracks still marked Unknown so curated names
                # are never clobbered.
                ca, cs = parse_artist_song(p.stem)
                if ca and cs:
                    track = store.get(str(p.resolve()))
                    if track is not None and track.artist.lower() in ("unknown", ""):
                        track.artist = ca
                        track.song = cs
                        added += 1
                        continue
                print(f"Skip unexpected: {stem}")
            continue

        possible += 1
        part1 = parts[0]
        part2 = parts[1]

        p1_artist = clean_artist(part1)
        p1_prefix = p1_artist.split('&', 1)[0]
        p1_prefix = uncomma_artist(p1_prefix) or p1_prefix

        is_p1_artist = p1_artist in likely_artists or p1_prefix in likely_artists

        p2_artist = clean_artist(part2)
        p2_prefix = p2_artist.split('&', 1)[0]
        p2_prefix = uncomma_artist(p2_prefix) or p2_prefix

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
        # raw-preserving: never write clean_artist output as a display name
        prefix, feature = _auto_clean_artist(first_artist)

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
            split = [s.strip() for s in song.split('-')]
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
            split = [s.strip() for s in song.split(',')]
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
            # Space-only stems: a leading/trailing word matching a known
            # artist is too weak to auto-apply ('America The Beautiful' would
            # become America - 'The Beautiful'). Report the candidates so a
            # human can act; never write. This branch previously computed a
            # fix, printed it, and silently discarded it.
            split = song.split(" ")
            if len(split) > 1 and clean_artist(split[0]) in artists:
                print(f"candidate: '{split[0]}' - '{' '.join(split[1:])}'  [{song}]")
            elif len(split) > 1 and clean_artist(split[-1]) in artists:
                print(f"candidate: '{split[-1]}' - '{' '.join(split[:-1])}'  [{song}]")
            no_split += 1

    questionary.print(f"Success {success}, simple {simple_split}, id {id_split}, Complex {complex_split}, Comma {comma_split}. No Split {no_split}")

def _tracks_to_review(store: TrackStore, review_state: ReviewState) -> list[Track]:
    """ Part of manual track review - build a list of unreviewed tracks.
        Focuses on artists with few total tracks (which are often miscategorized)
    """
    index = ArtistIndex.from_store(store)
    # Bypass: artist groups named in config "always_review" enter the queue
    # regardless of size. Garbage buckets (a wrong value shared by many
    # tracks) grow past the thin-artist threshold and would otherwise become
    # permanently invisible to Review. Config entries are matched cleaned, so
    # natural spellings work:  "always_review": ["Some Artist", ...]
    always = {clean_artist(a) for a in _load_config().get("always_review", [])}
    tracks: list[Track] = []
    for letter, top_node in index.get_root().list_nodes().items():
        for artist, artist_node in top_node.list_nodes().items():
            if artist_node.count() > 5 and artist not in always:
                continue
            for song, song_node in artist_node.list_nodes().items():
                tracks.extend(song_node.list_tracks())

    tracks = [t for t in tracks if review_state.get(t.path) != "ok"]
    return tracks


def _auto_clean_artist(artist: str) -> tuple[str, str | None]:
    """Split a credit into (primary artist, feature) and uncomma the primary,
    PRESERVING raw casing -- the old version returned clean_artist output,
    which put lowercased display names into the library."""
    raw = re.sub(r"\s+(?:with|and|feat\.?|f\.|ft\.?|featuring)\s+", " & ",
                 artist, flags=re.IGNORECASE)
    if "&" in raw:
        prefix, feature = raw.split("&", 1)
        prefix, feature = prefix.strip(), feature.strip()
    else:
        prefix, feature = raw.strip(), None
    prefix = uncomma_artist(prefix) or prefix
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
    and ignores catalog prefixes, karaoke suffixes, and &/and/feat variations.

    Swap guard: a track whose SONG field is a known artist while its artist
    field is not is never auto-ok'd here, even if the tag agrees -- on reversed
    rips the ID3 tag is often reversed too, so agreement proves nothing. Those
    are left for Tag-swap / manual review."""
    known = set(ArtistIndex.from_store(store).count_artists(low_bound=3))
    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    okayed = suspects = 0
    for t in _tracks_to_review(store, review_state):
        tag = t.metadata.get("tag_artist")
        if not tag or tag == "<not-found>":
            continue
        a = _artist_tokens(t.artist)
        g = _artist_tokens(tag)
        if a and a == g:
            if (clean_artist(t.artist) not in known
                    and _song_as_known_artist(t.song, known)):
                suspects += 1  # smells swapped; tag agreement is not trusted
                continue
            review_state.set(t.path, "ok")
            okayed += 1

    review_state.save(_REVIEW_STATE_PATH)
    questionary.print(
        f"Auto-ok'd {okayed} tracks corroborated by their ID3 tag"
        + (f"; left {suspects} swap-suspect(s) for Tag-swap/review" if suspects else "")
    )


def _song_as_known_artist(song: str, known: set) -> str | None:
    """If the song field is (or comma-flips to) a known artist, return that
    artist in natural word order; else None. 'Lavigne, Avril' -> 'Avril
    Lavigne' -- clean_artist alone misses comma forms, which is exactly how a
    batch of reversed tracks once evaded Tag-swap.

    The comma flip is tried FIRST: 'Dion, Celine' should resolve to the real
    'Celine Dion' (comma-form evidence) even when a 'Dion, Celine' spelling
    group also exists in the store. Band names whose canonical spelling has a
    comma ('Earth, Wind & Fire') don't flip to a known artist and fall through
    to the direct match."""
    if song.count(",") == 1:
        flipped = uncomma_artist(song)
        if flipped and clean_artist(flipped) in known:
            return flipped
    if clean_artist(song) in known:
        return song
    return None


def swap_from_tags(store: TrackStore) -> None:
    """Fix reversed artist/song parses.

    Two kinds of evidence, each requiring the song-field value to be a known
    artist (>=3 tracks, comma-aware) while the current artist is not (guards
    against mislabeled tags and titles that merely look like artist names):
      - the ID3 tag_artist matches the SONG field, or
      - the song field is a comma name ("Lavigne, Avril") -- real titles
        essentially never look like Last, First of a known artist.
    Scans the WHOLE store, including tracks already marked reviewed, so a
    swapped track that was wrongly ok'd earlier still gets corrected. A tag
    that corroborates the current orientation vetoes tag-based swaps (but not
    comma-form ones -- reversed rips often have reversed tags too)."""
    index = ArtistIndex.from_store(store)
    known = set(index.count_artists(low_bound=3))

    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    swapped = 0
    for t in store.all():
        if t.artist.lower() in ("unknown", "") or not t.song:
            continue
        if "\\" in t.artist:
            continue  # catalog-path artist: Clean's domain, and the title is
            # lost anyway -- swapping would just lock in a garbage song field
        if clean_artist(t.artist) in known:
            continue  # current artist looks legit; don't second-guess it
        flip = _song_as_known_artist(t.song, known)
        if flip is None:
            continue
        tag = t.metadata.get("tag_artist")
        tg = _artist_tokens(tag) if tag and tag != "<not-found>" else frozenset()
        comma_form = flip != t.song  # only true when the comma flip matched
        # A tag agreeing with the current orientation vetoes tag-based swaps --
        # but not comma-form ones: on reversed rips the tag is often reversed
        # too, and no real title looks like "Last, First" of a known artist.
        if not comma_form and tg and tg == _artist_tokens(t.artist):
            continue
        tag_says_song = bool(tg) and tg == _artist_tokens(t.song)
        if not (tag_says_song or comma_form):
            continue
        questionary.print(f"swap: '{t.artist} - {t.song}' -> '{flip} - {t.artist}'")
        t.artist, t.song = flip, t.artist
        store.add(t)
        review_state.set(t.path, "ok")
        swapped += 1

    review_state.save(_REVIEW_STATE_PATH)
    questionary.print(f"Swapped {swapped} reversed artist/song pairs")


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


# Karaoke noise to drop before both querying and scoring: parentheticals,
# bracketed tags ([SF Karaoke]), and common markers -- so '(Mplx)' etc. neither
# breaks the query nor depresses the score against MusicBrainz's clean title.
_MB_NOISE_RE = re.compile(
    r"\(.*?\)|\[.*?\]|\bw[\s-]?o?b?gv\b|\bwvocals?\b|\bmultiplex\b|\bmplx\b|\bvr\b|\bkaraoke\b",
    re.IGNORECASE,
)


def _mb_norm(s: str, uncomma: bool = False) -> str:
    """Normalize a field for a MusicBrainz phrase query: strip catalog/karaoke
    noise and collapse whitespace. For artists, flip a single 'Last, First' or
    'X, The' -> 'First Last' / 'The X' on the PRIMARY name (before the first
    &/feat) so MB can phrase-match multi-artist credits like
    'Chesney, Kenny & Uncle Kracker'. This only shapes the query -- scoring
    always uses the original string, so a mis-flip of a comma-band such as
    'Emerson, Lake & Palmer' can never produce a wrong match."""
    s = _CATALOG_PREFIX_RE.sub("", s or "")
    s = _MB_NOISE_RE.sub(" ", s)
    if uncomma:
        parts = re.split(r"(\s*(?:&|\b(?:and|feat|ft|featuring)\b\.?)\s*)",
                         s, maxsplit=1, flags=re.IGNORECASE)
        head = parts[0]
        if head.count(",") == 1:
            left, right = head.split(",")
            if left.strip() and right.strip():
                head = f"{right.strip()} {left.strip()}"
        s = head + "".join(parts[1:])
    return re.sub(r"\s+", " ", s).strip()


def _mb_sim(a: str, b: str) -> float:
    """Order- and punctuation-insensitive similarity, 0-100. Strips karaoke
    noise first so a '(Mplx)'/'[SF Karaoke]' suffix on our side doesn't lower
    the score against MusicBrainz's clean title."""
    def clean(s):
        s = _MB_NOISE_RE.sub(" ", s)
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
        s = s.replace("'", "")  # drop apostrophes (curly ones already gone) so contractions line up
        s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
        s = re.sub(r"\b(and|feat|ft|featuring)\b", " ", s)  # '&' (already dropped)/'and'/'feat' equivalent
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"^the ", "", s)     # ignore a leading article...
        return re.sub(r" the$", "", s)  # ...or a trailing one ('Four Aces, The')
    return rapidfuzz.fuzz.token_sort_ratio(clean(a), clean(b))


def _mb_search(artist: str, title: str) -> list:
    """Query MusicBrainz for recordings matching artist+title.

    Raises on network error so the caller can handle connectivity loss."""
    qa = _mb_escape(_mb_norm(artist, uncomma=True))
    qt = _mb_escape(_mb_norm(title))
    q = f'artist:"{qa}" AND recording:"{qt}"'
    url = _MB_URL + "?" + urllib.parse.urlencode({"query": q, "fmt": "json", "limit": "5"})
    req = urllib.request.Request(url, headers={"User-Agent": _MB_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp).get("recordings", [])


def _mb_search_title(title: str) -> list:
    """Search MusicBrainz by title only (for the conservative third pass),
    returning more candidates so the caller can match our artist against their
    credits. Raises on network error."""
    q = f'recording:"{_mb_escape(_mb_norm(title))}"'
    url = _MB_URL + "?" + urllib.parse.urlencode({"query": q, "fmt": "json", "limit": "20"})
    req = urllib.request.Request(url, headers={"User-Agent": _MB_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp).get("recordings", [])


def _mb_search_release(title: str, release: str) -> list:
    """Search MusicBrainz by title + release. For soundtrack tracks whose
    'artist' field is really the album/soundtrack name, a hit identifies the
    real performer (self-corroborated by the release match). Raises on error."""
    q = (f'recording:"{_mb_escape(_mb_norm(title))}" '
         f'AND release:"{_mb_escape(_mb_norm(release))}"')
    url = _MB_URL + "?" + urllib.parse.urlencode({"query": q, "fmt": "json", "limit": "10"})
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
        a = _mb_sim(our_artist, mb_artist)
        t = _mb_sim(our_title, mb_title)
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

    recheck = bool(questionary.confirm(
        "Re-check tracks previously scored none/flag (e.g. after matching improvements)?",
        default=False,
    ).ask())

    def _needs_mb(t: Track) -> bool:
        if not t.metadata.get("mb_checked"):
            return True  # never looked up
        return recheck and t.metadata.get("mb_match") in ("none", "flag", "suggest")

    todo = [t for t in _tracks_to_review(store, review_state) if _needs_mb(t)]
    if not todo:
        questionary.print("No tracks to look up on MusicBrainz.")
        return

    verbose = bool(questionary.confirm(
        "Verbose? Print a line per track alongside the progress bar.",
        default=False,
    ).ask())

    okd = swapped = flagged = suggested = nomatch = 0
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
                    na, nt, ma, mt, was_swap, via_title, sug_a, sug_t = pair_cache[key]
                else:
                    res = _mb_search(a, s)
                    time.sleep(1.1)  # MusicBrainz: ~1 req/sec
                    na, nt, ma, mt = _mb_best(res, a, s)
                    was_swap = via_title = False
                    sug_a = sug_t = ""
                    if not (na >= _MB_STRONG and nt >= _MB_STRONG):
                        # Try the reversed orientation to catch swapped parses.
                        res2 = _mb_search(s, a)
                        time.sleep(1.1)
                        sa, st_, sma, smt = _mb_best(res2, s, a)
                        if sa >= _MB_STRONG and st_ >= _MB_STRONG:
                            na, nt, ma, mt, was_swap = sa, st_, sma, smt, True
                    if not (na >= _MB_STRONG and nt >= _MB_STRONG):
                        # Conservative third pass: title-only search, accept only
                        # if our artist AND title both strongly match a returned
                        # recording (rejects same-title/different-artist noise).
                        res3 = _mb_search_title(s)
                        time.sleep(1.1)
                        ta, tt, tma, tmt = _mb_best(res3, a, s)
                        if ta >= _MB_STRONG and tt >= _MB_STRONG:
                            na, nt, ma, mt, via_title = ta, tt, tma, tmt, True
                        else:
                            # Soundtrack fallback: our 'artist' may actually be the
                            # album/soundtrack name. Query title + release; a hit
                            # identifies the real performer, corroborated by the
                            # release match. Recorded as a suggestion, never applied.
                            res4 = _mb_search_release(s, a)
                            time.sleep(1.1)
                            for r in res4:
                                if _mb_sim(s, r.get("title", "")) >= _MB_STRONG:
                                    sug_a = _mb_credit_name(r.get("artist-credit"))
                                    sug_t = r.get("title", "")
                                    break
                    pair_cache[key] = (na, nt, ma, mt, was_swap, via_title, sug_a, sug_t)
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
                kind = "swap" if was_swap else ("title" if via_title else "ok")
                t.metadata["mb_match"] = kind
                okd += 1
                if verbose:
                    tqdm.write(f"{kind:5} {a!r} - {s!r}  ->  mb {ma!r}/{mt!r}  (a={na:.0f} t={nt:.0f})")
            elif na >= _MB_WEAK and nt >= _MB_WEAK:
                # A recording exists but diverges -- keep for review, don't apply.
                t.metadata["mb_match"] = "flag"
                t.metadata["mb_artist"] = ma
                t.metadata["mb_title"] = mt
                flagged += 1
                if verbose:
                    tqdm.write(f"flag  {a!r} - {s!r}  diverges from mb {ma!r}/{mt!r}  (a={na:.0f} t={nt:.0f})")
            elif sug_a:
                # Title mapped to one dominant MB artist we couldn't corroborate
                # -- record the suggestion for review; never auto-applied.
                t.metadata["mb_match"] = "suggest"
                t.metadata["mb_artist"] = sug_a
                t.metadata["mb_title"] = sug_t
                suggested += 1
                if verbose:
                    tqdm.write(f"sug   {a!r} - {s!r}  ->  mb {sug_a!r}/{sug_t!r}  (title only, unconfirmed)")
            else:
                t.metadata["mb_match"] = "none"
                nomatch += 1
                if verbose:
                    tqdm.write(f"none  {a!r} - {s!r}  (no match)")

            if time.monotonic() - last_save >= 60:
                store.save(_CACHE_PATH)
                review_state.save(_REVIEW_STATE_PATH)
                last_save = time.monotonic()
    finally:
        store.save(_CACHE_PATH)
        review_state.save(_REVIEW_STATE_PATH)

    questionary.print(
        f"MusicBrainz: ok'd {okd} (incl {swapped} swapped), "
        f"flagged {flagged}, suggested {suggested}, no-match {nomatch}")


def _norm_eq(a: str, b: str) -> bool:
    """True if two strings match ignoring case, spaces and punctuation."""
    norm = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
    return norm(a) == norm(b)


def _reliable_words(fragments: list[str]) -> list[str]:
    """Query words from elided title fragments that are certainly complete.

    The disc filenames cut characters at the dashes, so a fragment's edge
    words may be partial ('Tur - E Loose' == 'Tur[n M]e Loose'): the first
    fragment's leading word is complete only when the fragment has more
    words, interior words are always complete, and the fragment after a cut
    may start mid-word. Returns up to three longest words (>= 3 chars)."""
    words: list[str] = []
    for i, frag in enumerate(fragments):
        w = frag.split()
        if i == 0:
            words += ([w[0]] + w[1:-1]) if len(w) > 1 else []
        elif i == len(fragments) - 1:
            words += w[1:] if len(w) > 1 else []
        else:
            words += w[1:-1]
    words = [re.sub(r"[^A-Za-z0-9']", "", x) for x in words]
    words = sorted({x for x in words if len(x) >= 3}, key=len, reverse=True)
    return words[:3]


def _mb_search_restitch(artist: str, words: list[str]) -> list:
    """Recording search by artist + known-complete title words (AND'd), for
    titles whose full phrase can't be queried because parts were elided."""
    qa = _mb_escape(_mb_norm(artist, uncomma=True))
    qw = " AND ".join(_mb_escape(w) for w in words)
    q = f'artist:"{qa}" AND recording:({qw})'
    url = _MB_URL + "?" + urllib.parse.urlencode({"query": q, "fmt": "json", "limit": "25"})
    req = urllib.request.Request(url, headers={"User-Agent": _MB_USER_AGENT})
    with urllib.request.urlopen(req, timeout=15) as resp:
        return json.load(resp).get("recordings", [])


def restitch_titles(store: TrackStore) -> None:
    """(online) Repair titles from dash-elided disc filenames.

    Some discs (FLY/SFKK style) write stems like
    'FLY-03-06 - Belinda - Carlisle - Heaven Is A Pla - On Earth': the artist
    is dash-split and the title has characters cut out at each dash. The
    parser took artist 'Belinda', song 'Carlisle', and dropped the rest.

    For unreviewed tracks whose stem has 3+ segments: reassemble the artist
    against the known-artist set, then ask MusicBrainz for recordings by that
    artist containing the fragments' complete words. A candidate title is
    accepted only when every fragment fits IN ORDER from the title's start
    (fragments_match_title) and the artist strongly matches. A single
    unambiguous hit is applied and marked ok; ambiguous or thin evidence is
    recorded as an mb_suggest for one-click acceptance in Review. Resumable
    (metadata['restitch']) and offline-safe."""
    if not _is_online():
        questionary.print("Restitch needs internet -- skipped (offline).")
        return

    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)
    known = set(ArtistIndex.from_store(store).count_artists())

    recheck = bool(questionary.confirm(
        "Re-check tracks previously restitched with no/weak result?",
        default=False,
    ).ask())

    todo = []
    for t in store.all():
        if review_state.get(t.path) == "ok":
            continue
        if t.metadata.get("restitch") and not recheck:
            continue
        parts = split_stem(Path(t.path).stem)
        if len(parts) < 3:
            continue
        joined = rejoin_artist(parts, known)
        if joined:
            artist, fragments = joined
        elif clean_artist(t.artist) in known:
            k = next((k for k in range(1, len(parts))
                      if clean_artist(" ".join(parts[:k])) == clean_artist(t.artist)), None)
            if k is None:
                continue
            artist, fragments = t.artist, parts[k:]
        else:
            continue
        fragments = [f for f in fragments if f]
        if fragments:
            todo.append((t, artist, fragments))

    if not todo:
        questionary.print("No dash-elided stems to restitch.")
        return
    questionary.print(f"{len(todo)} candidate track(s).")

    applied = suggested = none = 0
    consecutive_fail = 0
    last_save = time.monotonic()
    try:
        for t, artist, fragments in tqdm(todo, desc="Restitch", unit="track"):
            words = _reliable_words(fragments)
            if not words:
                t.metadata["restitch"] = "none"
                none += 1
                continue
            try:
                res = _mb_search_restitch(artist, words)
                time.sleep(1.1)
                consecutive_fail = 0
            except Exception:
                consecutive_fail += 1
                if consecutive_fail >= 5:
                    tqdm.write("Lost connection to MusicBrainz -- saving and stopping.")
                    break
                continue

            matches = []
            for r in res:
                mb_artist = _mb_credit_name(r.get("artist-credit"))
                mb_title = r.get("title", "")
                if (_mb_sim(artist, mb_artist) >= _MB_STRONG
                        and fragments_match_title(fragments, mb_title)):
                    matches.append((mb_artist, mb_title))
            distinct = {_norm_eq_key(mt) for _, mt in matches}
            frag_len = sum(len(re.sub(r"[^a-z0-9]", "", f.lower())) for f in fragments)
            if len(distinct) == 1 and frag_len >= 10:
                mb_artist, mb_title = matches[0]
                tqdm.write(f"fix   {t.artist!r} - {t.song!r}  ->  {mb_artist!r} - {mb_title!r}")
                t.artist, t.song = mb_artist, mb_title
                t.metadata["restitch"] = "ok"
                store.add(t)
                review_state.set(t.path, "ok")
                applied += 1
            elif matches:
                mb_artist, mb_title = matches[0]
                t.metadata["restitch"] = "suggest"
                t.metadata["mb_match"] = "suggest"
                t.metadata["mb_artist"] = mb_artist
                t.metadata["mb_title"] = mb_title
                suggested += 1
            else:
                t.metadata["restitch"] = "none"
                none += 1

            if time.monotonic() - last_save >= 60:
                store.save(_CACHE_PATH)
                review_state.save(_REVIEW_STATE_PATH)
                last_save = time.monotonic()
    finally:
        store.save(_CACHE_PATH)
        review_state.save(_REVIEW_STATE_PATH)

    questionary.print(
        f"Restitch: applied {applied}, suggested {suggested} (accept in Review), "
        f"no-match {none}")


def _norm_eq_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def review_mode(store: TrackStore) -> None:
    """Sequential review of tracks from artists with 5 or fewer tracks.

    A track is only marked reviewed ('ok') by choosing (ok) or accepting a
    suggestion. swap/edit/auto-clean apply a change but keep you on the same
    track so you can keep adjusting until it's right; (skip) moves on without
    marking it, so it returns in a later review. Suggestions come from a prior
    MusicBrainz run (mb_artist) or, failing that, the MP3's ID3 tag_artist."""
    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    tracks = _tracks_to_review(store, review_state)
    if not tracks:
        questionary.print("No tracks from artists with 5 or fewer tracks.")
        return

    # For the swapped-track warning: artists with >=3 tracks are "known".
    known_artists = set(ArtistIndex.from_store(store).count_artists(low_bound=3))

    session_cache: dict[str, str] = {}

    base_choices = [
        Choice("(exit)", value="exit", shortcut_key="0"),
        Choice("(ok)", value="ok"),
        Choice("(swap)", value="swap"),
        Choice("(edit)", value="edit"),
        Choice("(skip)", value="skip"),
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

            flip = _song_as_known_artist(track.song, known_artists)
            if flip and clean_artist(track.artist) not in known_artists:
                questionary.print(
                    f"    !! song field '{track.song}' is known artist '{flip}'"
                    " - probably swapped"
                )

            extra = []
            art_clean = clean_artist(track.artist)
            if '&' in art_clean or ',' in art_clean:
                prefix, feature = _auto_clean_artist(track.artist)
                extra.append(Choice(f"auto [{prefix}]", value="auto-clean"))
            mb_artist = track.metadata.get("mb_artist")
            mb_title = track.metadata.get("mb_title", "")
            if mb_artist:
                mb_kind = track.metadata.get("mb_match", "mb")
                questionary.print(f"    MusicBrainz ({mb_kind}): '{mb_artist}' / '{mb_title}'")
                set_a = not _norm_eq(mb_artist, track.artist)
                set_t = bool(mb_title) and not _norm_eq(mb_title, track.song)
                if set_a and set_t:
                    label = f"use MB -> artist '{mb_artist}', song '{mb_title}'"
                elif set_a:
                    label = f"use MB -> artist '{mb_artist}'"
                elif set_t:
                    label = f"use MB -> song '{mb_title}'"
                else:
                    label = None
                if label:
                    extra.append(Choice(label, value="mb-accept"))

            # ID3 tag as an offline fallback suggestion (cleaned of karaoke noise).
            tag = track.metadata.get("tag_artist")
            tag_clean = _mb_norm(tag, uncomma=True) if tag and tag != "<not-found>" else ""
            if tag_clean and not _norm_eq(tag_clean, track.artist):
                questionary.print(f"    ID3 tag: '{tag_clean}'")
                extra.append(Choice(f"use tag -> artist '{tag_clean}'", value="tag-accept"))

            result = questionary.select("Edit this track?", choices=base_choices + extra).ask()

        if result is None or result == "exit":
            review_state.save(_REVIEW_STATE_PATH)
            break

        # Only (ok) and accepting a suggestion mark a track reviewed and advance.
        # swap/edit/auto-clean apply a change but stay on the same track so you
        # can keep adjusting; (skip) moves on without marking it (it returns in a
        # later review).
        advance = False
        if result == "ok":
            review_state.set(track.path, "ok")
            review_state.save(_REVIEW_STATE_PATH)
            session_cache[key] = "ok"
            advance = True
        elif result == "skip":
            session_cache[key] = "skip"
            advance = True
        elif result == "mb-accept":
            mb_artist = track.metadata.get("mb_artist")
            mb_title = track.metadata.get("mb_title", "")
            if mb_artist:
                if not _norm_eq(mb_artist, track.artist):
                    track.artist = mb_artist
                if mb_title and not _norm_eq(mb_title, track.song):
                    track.song = mb_title
                store.add(track)
                review_state.set(track.path, "ok")
                review_state.save(_REVIEW_STATE_PATH)
                advance = True
        elif result == "tag-accept":
            tag = track.metadata.get("tag_artist")
            tag_clean = _mb_norm(tag, uncomma=True) if tag else ""
            if tag_clean:
                track.artist = tag_clean
                store.add(track)
                review_state.set(track.path, "ok")
                review_state.save(_REVIEW_STATE_PATH)
                advance = True
        elif result == "swap":
            track.artist, track.song = track.song, track.artist
            store.add(track)  # stay on this track
        elif result == "auto-clean":
            prefix, feature = _auto_clean_artist(track.artist)
            track.artist = prefix
            if feature:
                track.metadata["feature"] = feature.strip()
            store.add(track)  # stay on this track
        elif result == "edit":
            _edit_track_details(store, track)  # stay on this track

        if advance:
            i += 1


# Offline detection this confident needs no online confirmation -- skip the
# network call (and spare the AcoustID quota) once we're already sure.
_ONLINE_CORROBORATE_BELOW = 0.85


def _apply_key_result(track: Track, result: dict, sig: str) -> None:
    """Write a fused key result onto a track's metadata (or clear it)."""
    track.metadata["key_sig"] = sig
    track.metadata["key_source"] = result["source"]
    track.metadata["key_detail"] = result.get("detail", "")
    if result["key"]:
        track.metadata["key"] = result["key"]
        track.metadata["key_confidence"] = f"{result['confidence']:.3f}"
        cam = key_detect.to_camelot(result["key"])
        if cam:
            track.metadata["key_camelot"] = cam
        else:
            track.metadata.pop("key_camelot", None)
    else:
        for k in ("key", "key_confidence", "key_camelot"):
            track.metadata.pop(k, None)


def detect_keys(store: TrackStore) -> None:
    """(offline + online) Estimate each track's musical key for a pre-song pitch
    reference, writing key / key_confidence / key_source into metadata (which
    Final-final then emits into index.json for players like KriticalDJ).

    Precedence: manual override > ID3 TKEY tag > offline audio detection
    (librosa chromagram + Krumhansl-Schmuckler over the first verse) > online
    corroboration (AcoustID + AcousticBrainz). Online is advisory -- it reports
    the original master's key, not the transposed rip -- so it only lifts a
    weak offline read when it agrees, or fills where there's no local signal.

    Incremental & resumable: a track is skipped when it already carries a key
    computed from the current MP3 (key_sig == mp3_hash), unless you opt to
    re-check weak/none results. Offline detection needs librosa; online needs
    fpcalc + a free AcoustID key -- each degrades gracefully when absent."""
    cfg = _load_config()

    # --- capability + online setup ----------------------------------------
    if not key_detect.HAVE_LIBROSA:
        questionary.print(
            "librosa not installed -- offline audio key detection is unavailable "
            "(install with: pip install -r requirements-key.txt).")
    online_enabled = False
    api_key = ""
    if key_online.HAVE_ACOUSTID:
        if _is_online("api.acoustid.org"):
            api_key = cfg.get("acoustid_api_key", "") or os.environ.get("ACOUSTID_API_KEY", "")
            if not api_key:
                questionary.print(
                    "Online corroboration uses AcoustID + AcousticBrainz. It needs a "
                    "free AcoustID API key: https://acoustid.org/new-application")
                ans = questionary.text("AcoustID API key (blank to skip online):").ask()
                if ans and ans.strip():
                    api_key = ans.strip()
                    cfg["acoustid_api_key"] = api_key
                    _save_config(cfg)
            online_enabled = bool(api_key)
        else:
            questionary.print("AcoustID unreachable -- offline detection only.")
    else:
        questionary.print(
            "pyacoustid/fpcalc not installed -- online corroboration unavailable "
            "(offline only).")
    if online_enabled:
        questionary.print(
            "Online corroboration ON. Note: AcousticBrainz reports the ORIGINAL "
            "master's key, which can differ from a transposed karaoke rip; it is "
            "used only to confirm/fill, never to override a confident local read.")

    if not key_detect.HAVE_LIBROSA and not online_enabled:
        if not questionary.confirm(
                "Only ID3 TKEY tags and manual overrides are available. Continue?",
                default=False).ask():
            return

    recheck = questionary.confirm(
        "Re-check tracks previously scored none/online/low-confidence?",
        default=False).ask()

    overrides = _load_key_overrides()

    # --- build the work list ----------------------------------------------
    def _needs_work(t: Track) -> bool:
        if "mp3" not in t.file_types and "zip" not in t.file_types:
            return False  # no playable audio to analyse
        # A curated override must always win, even over a track already keyed
        # confidently on a previous run -- so never skip an overridden track.
        if f"{t.artist} - {t.song}".strip().lower() in overrides:
            return True
        sig = t.metadata.get("mp3_hash", "")
        done = bool(sig) and t.metadata.get("key_sig") == sig
        if not done:
            return True
        if not recheck:
            return False
        src = t.metadata.get("key_source", "")
        try:
            conf = float(t.metadata.get("key_confidence", "0") or 0)
        except ValueError:
            conf = 0.0
        return src in ("none", "online") or (src == "auto" and conf < key_detect.EMIT_FLOOR)

    todo = [t for t in store.all() if _needs_work(t)]
    if not todo:
        questionary.print("No tracks need key detection.")
        return
    missing_detail = sum(1 for t in todo if not t.metadata.get("mp3_hash"))
    if missing_detail:
        questionary.print(
            f"{missing_detail} track(s) have no mp3_hash -- run Detail first so key "
            "results can be cached/skipped on re-runs.")

    # --- detect ------------------------------------------------------------
    counts: dict[str, int] = collections.Counter()
    mbid_cache: dict[str, str | None] = {}
    last_save = time.monotonic()
    processed = 0
    for track in tqdm(todo, desc="Key-detect", unit="track"):
        sig = track.metadata.get("mp3_hash", "")
        ov = overrides.get(f"{track.artist} - {track.song}".strip().lower())
        tag = offline = online = None
        if ov is None:  # an override needs no file access at all
            with audio_file(track.path, track.file_types) as mp3_path:
                if mp3_path is not None:
                    tag = read_key_tag(mp3_path)
                    offline = key_detect.detect_key_offline(str(mp3_path))
                    if online_enabled and (offline is None
                                           or offline[1] < _ONLINE_CORROBORATE_BELOW):
                        try:
                            okey, _detail = key_online.lookup_online(
                                str(mp3_path), api_key, mbid_cache)
                            online = okey
                        except Exception:
                            online = None
        result = key_detect.combine_key_signals(
            override=ov, tag=tag, offline=offline, online=online)
        _apply_key_result(track, result, sig)
        counts[result["source"]] += 1
        processed += 1
        if time.monotonic() - last_save >= 300:
            store.save(_CACHE_PATH)
            last_save = time.monotonic()

    store.save(_CACHE_PATH)
    emitted = sum(1 for t in store.all()
                  if key_detect.should_emit(t.metadata.get("key_source", "none"),
                                            float(t.metadata.get("key_confidence", "0") or 0)))
    questionary.print(
        f"Key-detect: processed {processed} -- "
        f"manual {counts['manual']}, tag {counts['tag']}, auto {counts['auto']}, "
        f"online {counts['online']}, none {counts['none']}. "
        f"{emitted} song(s) now carry an index-worthy key.")


def apply_resolutions(store: TrackStore) -> None:
    """Apply a curated resolutions file to the store (dry-run first).

    Reads `resolutions.json` next to the cache — shape:
        {"version": 1, "resolutions": {"<track path>": {"artist": ..., "song":
        ..., "why": ...}}}
    For each path present in the store it sets the artist/song and marks the
    track reviewed ('ok'), recording provenance in metadata ('artist_from' =
    'resolutions'). Entries missing 'artist'/'song' leave that field unchanged.
    Paths not in the store are skipped and counted. Prompts for a dry run first
    so the whole change set can be eyeballed before anything is written."""
    p = Path(_RESOLUTIONS_PATH)
    if not p.exists():
        questionary.print(f"No resolutions file at {p}.")
        return
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except (ValueError, OSError) as exc:
        questionary.print(f"Could not read resolutions: {exc}")
        return
    res = data.get("resolutions", {})
    if not res:
        questionary.print("Resolutions file has no entries.")
        return

    dry = questionary.confirm(
        f"{len(res)} resolutions found. Dry run (show changes without writing)?",
        default=True,
    ).ask()
    if dry is None:
        return

    review_state = ReviewState()
    review_state.load(_REVIEW_STATE_PATH)

    applied = missing = unchanged = 0
    for path, r in res.items():
        track = store.get(path)
        if track is None:
            missing += 1
            continue
        new_artist = r.get("artist", track.artist)
        new_song = r.get("song", track.song)
        changed = new_artist != track.artist or new_song != track.song
        if not changed and review_state.get(path) == "ok":
            unchanged += 1
            continue
        questionary.print(
            f"{'[dry] ' if dry else ''}{Path(path).stem}\n"
            f"    '{track.artist} - {track.song}'  ->  '{new_artist} - {new_song}'"
            f"    ({r.get('why', '')})"
        )
        if not dry:
            track.artist = new_artist
            track.song = new_song
            track.metadata["artist_from"] = "resolutions"
            store.add(track)
            review_state.set(path, "ok")
            applied += 1

    if dry:
        questionary.print(
            f"Dry run: {len(res)} entries ({missing} not in this store). "
            "Re-run and answer 'no' to apply."
        )
    else:
        store.save(_CACHE_PATH)
        review_state.save(_REVIEW_STATE_PATH)
        questionary.print(
            f"Applied {applied}; skipped {missing} not-in-store, {unchanged} already-set."
        )


def unify_artists(store: TrackStore) -> None:
    """Bulk-rename artist variants to a canonical spelling from an alias map.

    Reads `artist-aliases.json` next to the cache — shape:
        {"version": 1, "aliases": {"<variant artist>": "<canonical>", ...}}
    For every track whose exact artist string matches a key, rewrites it to the
    canonical form. Prompts for a dry run first so the full rename set can be
    reviewed. Only the artist field changes; review state is left untouched."""
    aliases = _load_aliases()
    if not aliases:
        questionary.print(f"No usable aliases at {_ARTIST_ALIASES_PATH}.")
        return

    dry = questionary.confirm(
        f"{len(aliases)} artist aliases loaded. Dry run (show renames without writing)?",
        default=True,
    ).ask()
    if dry is None:
        return

    per_alias: dict[str, int] = {}
    renamed = 0
    for t in store.all():
        canon = aliases.get(t.artist)
        if canon is None:
            continue
        per_alias[t.artist] = per_alias.get(t.artist, 0) + 1
        renamed += 1
        if not dry:
            t.artist = canon
            store.add(t)

    ordered = sorted(per_alias, key=lambda k: -per_alias[k])
    shown = ordered if len(ordered) <= 80 else ordered[:80]
    for variant in shown:
        questionary.print(
            f"{'[dry] ' if dry else ''}'{variant}' -> '{aliases[variant]}'"
            f"  ({per_alias[variant]} track(s))"
        )
    if len(ordered) > len(shown):
        questionary.print(f"    ... and {len(ordered) - len(shown)} more variant(s)")

    if dry:
        questionary.print(
            f"Dry run: {len(per_alias)}/{len(aliases)} aliases match tracks, "
            f"{renamed} track(s) would be renamed. Re-run and answer 'no' to apply."
        )
    else:
        store.save(_CACHE_PATH)
        questionary.print(
            f"Renamed {renamed} track(s) across {len(per_alias)} variant(s)."
        )


def report_track_count(store: TrackStore) -> None:
    """ List a summary of track information """
    all_songs = set(f"{clean_artist(t.artist)} - {clean_song(t.song)}" for t in store.all()
        if t.artist.lower() not in ["unknown", ""])
    questionary.print(f"Distinct count: {len(all_songs)} / {len(store.all())}")


def library_stats(store: TrackStore) -> None:
    """Print interesting statistics about the library (read-only)."""
    tracks = store.all()
    total = len(tracks)
    if not total:
        questionary.print("Library is empty.")
        return

    def _int(v):
        try:
            return int(float(v))
        except (TypeError, ValueError):
            return 0

    artist_songs: dict[str, set] = collections.defaultdict(set)
    artist_display: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    pair_copies: collections.Counter = collections.Counter()
    pair_display: dict[tuple, str] = {}
    fmt: collections.Counter = collections.Counter()
    decades: collections.Counter = collections.Counter()
    total_seconds = 0.0
    total_bytes = 0
    bitrates: list[int] = []
    detailed = unknown = 0

    for t in tracks:
        a = t.artist
        ca, cs = clean_artist(a), clean_song(t.song)
        is_unknown = a.lower() in ("unknown", "")
        unknown += is_unknown
        for ext in t.file_types:
            fmt[ext] += 1
        md = t.metadata
        total_seconds += _int(md.get("length_seconds"))  # rounded secs, fine for a sum
        total_bytes += _int(md.get("mp3_size")) + _int(md.get("cdg_size"))
        br = _int(md.get("bitrate_bps"))
        if br:
            bitrates.append(br)
        if md.get("length_seconds"):
            detailed += 1
        ym = re.search(r"(19|20)\d\d", md.get("tag_year", "") or "")
        if ym:
            decades[(int(ym.group()) // 10) * 10] += 1
        if not is_unknown and cs:
            artist_songs[ca].add(cs)
            artist_display[ca][a] += 1
            pair_copies[(ca, cs)] += 1
            pair_display.setdefault((ca, cs), f"{a} - {t.song}")

    distinct_pairs = len(pair_copies)
    dupes = sum(c - 1 for c in pair_copies.values())

    def hrs(sec):
        return f"{sec / 3600:,.0f}h ({sec / 86400:.1f} days)"

    def name(counter):
        return counter.most_common(1)[0][0]

    L = questionary.print
    L("=== Library statistics ===")
    L(f"Track files:            {total:,}")
    L(f"Distinct songs:         {distinct_pairs:,}  (artist+song, spelling-normalized)")
    L(f"Distinct artists:       {len(artist_songs):,}")
    L(f"Unknown-artist files:   {unknown:,}")
    L(f"Duplicate copies:       {dupes:,}  ({dupes / total:.0%} of files are extra copies)")
    L(f"Formats:                " + ", ".join(f"{k}={v:,}" for k, v in fmt.most_common()))
    L("")
    L(f"Detailed (have MP3 info): {detailed:,} / {total:,}")
    L(f"Total audio:            {hrs(total_seconds)}")
    L(f"Total MP3+CDG size:     {total_bytes / 1e9:,.1f} GB")
    if detailed:
        L(f"Avg song length:        {int(total_seconds / detailed) // 60}m {int(total_seconds / detailed) % 60}s")
    if bitrates:
        L(f"Avg MP3 bitrate:        {sum(bitrates) // len(bitrates) // 1000} kbps")
    L("")
    L("Top 15 artists by distinct songs:")
    top_artists = sorted(artist_songs, key=lambda k: -len(artist_songs[k]))[:15]
    for i, ca in enumerate(top_artists, 1):
        L(f"  {i:>2}. {name(artist_display[ca])}  —  {len(artist_songs[ca]):,}")
    L("")
    L("Most-duplicated songs (copies in library):")
    for (key, n) in pair_copies.most_common(10):
        if n < 2:
            break
        L(f"  {n:>3}  {pair_display[key]}")
    if decades:
        L("")
        L("Songs by decade (from ID3 year):")
        for d in sorted(decades):
            L(f"  {d}s  {decades[d]:,}")

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


def _ranked_tracks(tracks: list[Track]) -> list[Track]:
    """The copies of one artist-song pair, best-first (largest MP3 wins), so
    the export can keep the best N versions rather than only the single best."""
    def _mp3_size(t: Track) -> int:
        raw = t.metadata.get("mp3_size")
        try:
            return int(raw) if raw is not None else 0
        except (TypeError, ValueError):
            return 0

    return sorted(tracks, key=_mp3_size, reverse=True)

def tracks_to_keep(store: TrackStore) -> None:
    """Export one best copy per artist+song to an output tree, laid out as
    <output>/<first-letter>/<artist>/<name>.<ext> (unknown artists skipped).

    The output directory is prompted and remembered between runs. Files already
    present and unchanged (same size) are skipped, so re-running is safe and
    incremental. Stale files from earlier runs (e.g. after an artist rename) can
    optionally be pruned."""
    cfg = _load_config()
    out = questionary.path(
        "Output directory:",
        default=cfg.get("output_path", "/tmp/output"),
    ).ask()
    if not out or not out.strip():
        return
    out = out.strip()
    cfg["output_path"] = out
    _save_config(cfg)
    output_root = Path(out)

    # How many copies of each song to export (1 = best only, as before). Extra
    # copies become selectable "versions" in the index for players like
    # KriticalDJ. Remembered between runs.
    try:
        default_vl = int(cfg.get("version_limit", 1))
    except (TypeError, ValueError):
        default_vl = 1
    vl_raw = questionary.text(
        "Max versions to export per song (1 = best only):",
        default=str(default_vl),
    ).ask()
    if vl_raw is None:
        return
    try:
        version_limit = max(1, int(vl_raw.strip()))
    except ValueError:
        version_limit = 1
    cfg["version_limit"] = version_limit
    _save_config(cfg)

    # Collect the desired output: dest -> source, one best copy per artist+song.
    expected: dict[Path, Path] = {}
    index_entries: list[dict] = []
    node_set = [ArtistIndex.from_store(store).get_root()]
    while node_set:
        n = node_set.pop()
        if not n.is_leaf():
            node_set.extend(n.list_nodes().values())
            continue
        ranked = _ranked_tracks(n.list_tracks())  # best-first
        best = ranked[0]
        artist = clean_artist(best.artist)
        if artist in ("", "unknown"):
            continue
        prefix = artist[0] if re.match("[a-z]", artist[:1]) else "#"
        # folder-safe: an artist like 'ac/dc' must not nest an extra directory
        # level (the pruner's depth==3 filter would never see those files)
        artist = safe_folder(artist)
        stem = Path(best.path).stem
        for exn in best.file_types:
            src = Path(best.path).with_suffix(f".{exn}")  # per-type source (.mp3/.cdg/.zip)
            expected[output_root / prefix / artist / f"{stem}.{exn}"] = src

        def _playable_ext(t: Track):
            return "zip" if "zip" in t.file_types else \
                ("mp3" if "mp3" in t.file_types else None)

        def _dur(t: Track) -> int:
            try:
                return int(float(t.metadata.get("length_seconds", "") or 0))
            except ValueError:
                return 0

        # Curated index entry for external players (e.g. KriticalDJ): cleaned
        # artist/song strings + duration, pointing at the playable file, so
        # players need not re-derive names from noisy filename stems.
        media_ext = _playable_ext(best)
        if media_ext:
            entry = {
                "path": f"{prefix}/{artist}/{stem}.{media_ext}",
                "artist": best.artist,
                "title": best.song,
                "duration": _dur(best),
            }
            # Optional musical key for a pre-song pitch reference (KriticalDJ
            # #15). Emitted only when confident enough (Key-detect gates auto/
            # online on the emit floor; manual/tag always pass), so a player can
            # trust its presence and light the feature up gradually. Fields are
            # omitted entirely otherwise -- backward compatible with readers
            # that don't know about keys.
            entry.update(key_detect.key_index_fields(best.metadata))
            # Extra copies (up to version_limit total) are exported alongside
            # and listed as selectable versions. Label = the source's folder
            # (usually the karaoke brand/disc) so the KJ can tell them apart.
            versions = []
            for alt in ranked[1:version_limit]:
                astem = Path(alt.path).stem
                aext = _playable_ext(alt)
                if aext is None or astem == stem:  # unplayable or name-collision
                    continue
                for exn in alt.file_types:
                    expected[output_root / prefix / artist / f"{astem}.{exn}"] = \
                        Path(alt.path).with_suffix(f".{exn}")
                # Each alternate publishes its OWN key, never the best copy's:
                # rips from different karaoke brands are often transposed
                # relative to each other, so a shared key would be wrong for
                # whichever copy didn't produce it. An alternate without a
                # confident key of its own simply carries none.
                ventry = {
                    "path": f"{prefix}/{artist}/{astem}.{aext}",
                    "label": Path(alt.path).parent.name or "Alternate",
                    "duration": _dur(alt),
                }
                ventry.update(key_detect.key_index_fields(alt.metadata))
                versions.append(ventry)
            if versions:
                entry["versions"] = versions
            index_entries.append(entry)

    # Copy anything missing or changed; skip files already present and identical.
    copied = skipped = missing = 0
    for dest, src in tqdm(
        expected.items(), total=len(expected), desc="Exporting", unit="file"
    ):
        if not src.exists():
            missing += 1
            continue
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            skipped += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        copied += 1
    msg = f"Output -> {output_root}: copied {copied}, skipped {skipped} unchanged"
    if missing:
        msg += f", {missing} source file(s) missing"
    questionary.print(msg)

    # Optional prune of stale files from earlier runs. Restricted to this tool's
    # own <prefix>/<artist>/<file> layout so unrelated files are never touched.
    if output_root.exists():
        want = {d.resolve() for d in expected}
        stale = [
            p for p in output_root.rglob("*")
            if p.is_file()
            and len(p.relative_to(output_root).parts) == 3
            and p.resolve() not in want
        ]
        if stale and questionary.confirm(
            f"Prune {len(stale)} stale output file(s) no longer in the keep set?",
            default=False,
        ).ask():
            for p in tqdm(stale, desc="Pruning", unit="file"):
                p.unlink()
            for d in sorted(
                (d for d in output_root.rglob("*") if d.is_dir()),
                key=lambda x: len(x.parts), reverse=True,
            ):
                try:
                    d.rmdir()  # drop now-empty dirs
                except OSError:
                    pass
            questionary.print(f"Pruned {len(stale)} stale file(s).")
        elif stale:
            questionary.print(f"Left {len(stale)} stale file(s) in place.")

    # Curated library index for external players (KriticalDJ reads this).
    output_root.mkdir(parents=True, exist_ok=True)
    index_path = output_root / "index.json"
    index_path.write_text(
        json.dumps({"version": 1, "songs": index_entries}, ensure_ascii=False),
        encoding="utf-8")
    questionary.print(f"Library index -> {index_path} ({len(index_entries):,} songs)")

    # Keep the digital songbook in the output tree current with this export.
    book, n = build_songbook(store, output_root / "songbook.html",
                             cfg.get("songbook_name", ""))
    questionary.print(f"Songbook refreshed -> {book} ({n:,} songs)")


# Single-file offline songbook. Everything (styles, script, data) is inlined so
# the generated file works by double-clicking it in any browser, no internet or
# install needed. Raw string: JS escapes like ̀ and "\t" must reach the
# browser verbatim.
_SONGBOOK_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#14161a;--panel:#1e2128;--text:#e8eaed;--dim:#9aa0a6;--accent:#4fc3f7;--line:#22252c}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--text);font-family:"Segoe UI",system-ui,Arial,sans-serif}
header{position:sticky;top:0;background:var(--panel);padding:10px 14px;box-shadow:0 2px 8px rgba(0,0,0,.5);z-index:2}
h1{margin:0 0 8px;font-size:20px}
h1 small{color:var(--dim);font-weight:normal;font-size:13px;margin-left:8px}
#q{width:100%;font-size:22px;padding:12px 14px;border-radius:10px;border:1px solid #333;background:var(--bg);color:var(--text)}
#q:focus{outline:2px solid var(--accent)}
.filters{display:flex;gap:8px;margin-top:8px}
.filters button{font-size:15px;padding:8px 16px;border-radius:999px;border:1px solid #3a3f47;background:var(--bg);color:var(--dim);cursor:pointer}
.filters button.on{background:var(--accent);color:#00222f;border-color:var(--accent);font-weight:600}
.alpha{display:flex;gap:4px;margin-top:8px;flex-wrap:wrap}
.alpha button{flex:1;min-width:30px;font-size:14px;padding:6px 0;border-radius:6px;border:1px solid #3a3f47;background:var(--bg);color:var(--dim);cursor:pointer}
.alpha button.on{background:var(--accent);color:#00222f;border-color:var(--accent);font-weight:700}
#meta{color:var(--dim);font-size:14px;padding:10px 16px}
#list{padding:0 10px 40px}
.artist{font-size:17px;font-weight:700;color:var(--accent);padding:14px 8px 4px;border-bottom:1px solid #2c3038;cursor:pointer}
.song{font-size:18px;padding:10px 8px 10px 24px;border-bottom:1px solid var(--line);cursor:pointer}
.details{font-size:14px;color:var(--dim);padding:2px 8px 12px 24px;border-bottom:1px solid var(--line)}
</style>
</head>
<body>
<header>
<h1>__TITLE__ <small>__COUNT__ songs &middot; __DATE__</small></h1>
<input id="q" type="search" placeholder="Search artist or song&hellip;" autofocus>
<div class="filters" id="filters"><button data-f="all">All</button><button data-f="artist">Artist</button><button data-f="song">Song title</button></div>
<div class="alpha" id="alpha"></div>
</header>
<div id="meta"></div>
<div id="list"></div>
<script id="data" type="application/json">__DATA__</script>
<script>
const RAW=JSON.parse(document.getElementById("data").textContent);
const norm=s=>s.normalize("NFKD").replace(/[\u0300-\u036f]/g,"").toLowerCase().replace(/[^a-z0-9 ]+/g," ").replace(/ +/g," ").trim();
const ITEMS=RAW.map(l=>{const p=l.split("\t");return{a:p[0],s:p[1],m:p[2]||"",na:norm(p[0]),ns:norm(p[1])};});
const MAX=400;
let filter="all",letter="";
const q=document.getElementById("q"),list=document.getElementById("list"),meta=document.getElementById("meta");
const fbar=document.getElementById("filters"),abar=document.getElementById("alpha");
"#ABCDEFGHIJKLMNOPQRSTUVWXYZ".split("").forEach(ch=>{const b=document.createElement("button");b.textContent=ch;b.onclick=()=>{letter=(letter===ch)?"":ch;sync();render();};abar.appendChild(b);});
function sync(){[...abar.children].forEach(b=>b.classList.toggle("on",b.textContent===letter));[...fbar.children].forEach(b=>b.classList.toggle("on",b.dataset.f===filter));}
fbar.addEventListener("click",e=>{const f=e.target.dataset.f;if(!f)return;filter=f;sync();render();});
let timer;q.addEventListener("input",()=>{clearTimeout(timer);timer=setTimeout(render,120);});
const letterOf=na=>{const c=na.charAt(0);return c>="a"&&c<="z"?c.toUpperCase():"#";};
function render(){
 const toks=norm(q.value).split(" ").filter(Boolean);
 let total=0;const out=[];
 for(const it of ITEMS){
  if(letter&&letterOf(it.na)!==letter)continue;
  const hay=filter==="artist"?it.na:filter==="song"?it.ns:it.na+" "+it.ns;
  let ok=true;for(const tk of toks){if(hay.indexOf(tk)<0){ok=false;break;}}
  if(!ok)continue;
  total++;if(out.length<MAX)out.push(it);
 }
 const frag=document.createDocumentFragment();let last=null;
 for(const it of out){
  if(it.a!==last){last=it.a;const h=document.createElement("div");h.className="artist";h.textContent=it.a;
   h.onclick=()=>{q.value=it.a;filter="artist";letter="";sync();render();};frag.appendChild(h);}
  const d=document.createElement("div");d.className="song";d.textContent=it.s;
  d.onclick=()=>{const n=d.nextElementSibling;
   if(n&&n.classList.contains("details")){n.remove();return;}
   document.querySelectorAll(".details").forEach(x=>x.remove());
   const dd=document.createElement("div");dd.className="details";
   dd.textContent=it.m||"No details available.";d.after(dd);};
  frag.appendChild(d);
 }
 list.replaceChildren(frag);
 meta.textContent=total===0?"No matches.":(total>out.length?total.toLocaleString()+" matches - showing first "+out.length+". Keep typing to narrow.":total.toLocaleString()+(total===1?" match.":" matches."));
}
sync();render();
</script>
</body>
</html>"""


# Karaoke-brand noise: "[SC Karaoke]"-style bracket tags and known technical
# parentheticals -- stripped from the store by Clean and from songbook display
# strings. Deliberately narrow so real parentheticals like "(I've Had) The
# Time of My Life" survive.
_KARAOKE_BRACKET_RE = re.compile(r"\s*\[[^\]]*karaoke[^\]]*\]", re.IGNORECASE)
_BOOK_PAREN_RE = re.compile(
    r"\s*\((?:wo?bgv|w/?bgv|no bgv|bgv|wvocals?|vr|multiplex|mplx|"
    r"no backing vocals?|instr\.?|instrumental(?: version)?)\)",
    re.IGNORECASE,
)


def _book_display(s: str) -> str:
    s = _KARAOKE_BRACKET_RE.sub("", s or "")
    s = _BOOK_PAREN_RE.sub("", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def _songbook_entries(store: TrackStore) -> list[tuple[str, str, str]]:
    """Distinct (artist, song, details) rows for the songbook.

    Display strings are stripped of karaoke-brand noise first, then deduped
    with the same normalization List/Stats use (so bracket variants of one
    title collapse together); shows the most-common spelling of each artist
    and title. Details describe the best copy — the same pick Final-final
    exports: duration, bitrate, format, year and album where available."""
    artist_names: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    song_names: dict[tuple, collections.Counter] = collections.defaultdict(collections.Counter)
    pair_tracks: dict[tuple, list[Track]] = collections.defaultdict(list)
    for t in store.all():
        if t.artist.lower() in ("unknown", ""):
            continue
        da, ds = _book_display(t.artist), _book_display(t.song)
        ca, cs = clean_artist(da), clean_song(ds)
        if not ca or not cs:
            continue
        artist_names[ca][da] += 1
        song_names[(ca, cs)][ds] += 1
        pair_tracks[(ca, cs)].append(t)

    def fold(s: str) -> str:  # accent-insensitive sort key
        return unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()

    def details(group: list[Track]) -> str:
        # "Length: 3:47 @ 192 kbps · 2007 · Album" — time @ bitrate, then the rest.
        md = _best_track(group).metadata
        head = ""
        try:
            sec = int(float(md.get("length_seconds", "")))
            head = f"Length: {sec // 60}:{sec % 60:02d}"
        except ValueError:
            pass
        try:
            kbps = f"{int(md.get('bitrate_bps', '')) // 1000} kbps"
            head = f"{head} @ {kbps}" if head else kbps
        except ValueError:
            pass
        parts = [head] if head else []
        ym = re.search(r"(19|20)\d\d", md.get("tag_year", "") or "")
        if ym:
            parts.append(ym.group())
        album = md.get("tag_album", "")
        if album and album != "<not-found>":
            parts.append(album[:48])
        return " · ".join(parts)

    entries = [
        (
            artist_names[key[0]].most_common(1)[0][0],
            names.most_common(1)[0][0],
            details(pair_tracks[key]),
        )
        for key, names in song_names.items()
    ]
    entries.sort(key=lambda e: (fold(e[0]), fold(e[1])))
    return entries


def build_songbook(store: TrackStore, output_file: Path, name: str = "") -> tuple[Path, int]:
    """Generate the single-file offline songbook HTML at output_file.

    `name` personalizes the title ("<name>'s Karaoke Songbook"). Entries are
    embedded as JSON "artist\\tsong\\tdetails" lines; '</' is escaped so no
    title can terminate the script block early."""
    entries = _songbook_entries(store)
    tab = chr(9)
    lines = [
        f"{a.replace(tab, ' ')}\t{s.replace(tab, ' ')}\t{m.replace(tab, ' ')}"
        for a, s, m in entries
    ]
    data = json.dumps(lines, ensure_ascii=False).replace("</", "<\\/")
    title = f"{name}'s Karaoke Songbook" if name else "Karaoke Songbook"
    title = title.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = (
        _SONGBOOK_TEMPLATE
        .replace("__TITLE__", title)
        .replace("__COUNT__", f"{len(entries):,}")
        .replace("__DATE__", time.strftime("%Y-%m-%d"))
        .replace("__DATA__", data)
    )
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(html, encoding="utf-8")
    return output_file, len(entries)


def songbook(store: TrackStore) -> None:
    """Menu command: generate the songbook to a prompted, remembered path,
    with a remembered owner name for the title."""
    cfg = _load_config()
    name = questionary.text(
        "Name for the book title (\"<name>'s Karaoke Songbook\"; blank for plain):",
        default=cfg.get("songbook_name", ""),
    ).ask()
    if name is None:
        return
    name = name.strip()
    default = cfg.get("songbook_path") or (
        str(Path(cfg["output_path"]) / "songbook.html")
        if cfg.get("output_path") else str(_CACHE_PATH.parent / "songbook.html")
    )
    out = questionary.path("Songbook output file:", default=default).ask()
    if not out or not out.strip():
        return
    out = out.strip()
    cfg["songbook_name"] = name
    cfg["songbook_path"] = out
    _save_config(cfg)
    book, n = build_songbook(store, Path(out), name)
    questionary.print(f"Songbook -> {book} ({n:,} songs)")


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

    # "[SC Karaoke]"-style brand tags: the bracket contents vary by disc
    # series, so these are matched by pattern rather than the literal lists
    # above. Left in place, they split one song into several "distinct" titles
    # (with/without the tag), duplicating Final-final exports and inflating
    # counts.
    debranded = 0
    for t in store.all():
        for field in ("artist", "song"):
            val = getattr(t, field)
            m = _KARAOKE_BRACKET_RE.search(val)
            if m:
                t.metadata["style"] = m.group().strip()
                setattr(t, field, re.sub(r"\s{2,}", " ", _KARAOKE_BRACKET_RE.sub("", val)).strip())
                debranded += 1
    questionary.print(f"Stripped {debranded} [.. Karaoke ..] brand tags")

    # Case-insensitive backstop for the paren-noise the literal lists above
    # miss: the song list lacked "(Wbgv)"/"(Bgv)"/"(Mplx)" outright, and
    # literal matching can never cover case variants like "(WOBGV)".
    denoised = 0
    for t in store.all():
        for field in ("artist", "song"):
            val = getattr(t, field)
            m = _BOOK_PAREN_RE.search(val)
            if m:
                t.metadata["style"] = m.group().strip()
                setattr(t, field, re.sub(r"\s{2,}", " ", _BOOK_PAREN_RE.sub("", val)).strip())
                denoised += 1
    questionary.print(f"Stripped {denoised} karaoke parentheticals (case-insensitive)")

    # Bogus artists become "Unknown" so the Unknown pipeline (Tag-fill,
    # Fix-unknown) can reach them:
    #   - bare 1-2 digit values: disc track positions from filenames like
    #     "EZH-31 - 04 - Milkshake" (3+ digits untouched: 911/411/112 are real
    #     bands, as are letters-with-digits like Blink-182)
    #   - catalog-path fragments ("SC\SC-199\SC-199-02")
    #   - empty strings: their shared "" group exceeds the thin-artist
    #     threshold, making them permanently invisible to Review otherwise
    cleared = 0
    for t in store.all():
        artist = t.artist.strip()
        if artist.lower() == "unknown":
            continue
        is_track_no = bool(re.fullmatch(r"\d{1,2}", artist))
        is_catalog_path = "\\" in artist
        if not (is_track_no or is_catalog_path or not artist):
            continue
        if is_track_no:
            # Preserve what we prune: the disc track number, and the catalog id
            # (parts[0] of the original filename, dropped at parse time). Both
            # are useful later (e.g. catalog-number -> real-artist lookup).
            t.metadata["track_no"] = artist
            stem_parts = Path(t.path).stem.split(" - ")
            if len(stem_parts) >= 3:
                t.metadata["catalog_id"] = stem_parts[0].strip()
        elif is_catalog_path:
            t.metadata.setdefault("catalog_id", artist)
        t.artist = "Unknown"
        cleared += 1
    questionary.print(f"Cleared {cleared} track-number/catalog-path/empty artists")

    # Artist echoes in song titles (issue #2): a filename like
    # 'Isaak, Chris-Wicked Game' surviving into the song field makes the
    # display chain read 'Chris Isaak - Isaak, Chris-Wicked Game'. Strip a
    # dash-separated leading/trailing segment only when its words are exactly
    # the artist's (order-insensitive, so comma forms match too).
    echoed = 0
    for t in store.all():
        if t.artist.lower() in ("unknown", ""):
            continue
        fixed = strip_artist_echo(t.artist, t.song)
        if fixed:
            t.song = fixed
            echoed += 1
    questionary.print(f"Stripped {echoed} artist echoes from song titles")

    # Catalog ids parked in the song field: 'Artist - Song - SF 193-16' stems
    # (catalog LAST -- Sunfly Main Series zips) used to parse as
    # artist='Song', song='SF 193-16', dropping the real artist. Re-parse the
    # stem with the catalog-order-aware parser and recover both fields.
    recats = 0
    for t in store.all():
        if not is_catalog_segment(t.song, relaxed=True):
            continue
        # positional guard for the relaxed shape: the catalog-ish song value
        # must literally be one of the stem's trailing segments, so a real
        # title that merely looks catalog-ish (LeVert's 'ABC 123') is never
        # "repaired" away
        stem_parts = [p.strip() for p in Path(t.path).stem.split(" - ") if p.strip()]
        if t.song.strip() not in stem_parts[-2:]:
            continue
        a2, s2 = parse_artist_song(Path(t.path).stem)
        if s2 and s2 != t.song and not is_catalog_segment(s2, relaxed=True):
            t.metadata.setdefault("catalog_id", t.song.strip())
            if a2:
                t.artist = a2
            elif _norm_eq(t.artist, s2):
                # the artist slot was just holding the title; route the track
                # into the Unknown pipeline instead of leaving artist == song
                t.artist = "Unknown"
            t.song = s2
            recats += 1
    questionary.print(f"Re-parsed {recats} catalog-id songs from their filenames")


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
    """Uncomma: rewrite 'Last, First' artists to 'First Last'.

    Only swaps when the safe person-name swap succeeds (uncomma_artist -- band
    names like 'Earth, Wind & Fire' return None and are left alone) AND the
    swapped form is corroborated: either the alias map names a canonical
    target, or the swapped artist already exists in the store. Rewrites the
    RAW artist string, preserving casing -- the old version wrote the
    lowercased clean form back as the display name, which is exactly where
    relics like 'wind & fire earth' came from."""
    index = ArtistIndex.from_store(store)
    artists = set(index.count_artists())
    aliases = _load_aliases()

    swap_count = 0
    for t in store.all():
        raw = t.artist
        target = aliases.get(raw)
        if target is None and "," in raw:
            swapped = uncomma_artist(raw)
            if swapped and clean_artist(swapped) in artists:
                target = swapped
        if target and target != raw:
            t.artist = target
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
                        # display the winner's majority RAW spelling -- the
                        # node key is the lowercased clean form and writing it
                        # back is how lowercase display relics were born
                        target = majority_raw(letter_nodes[other_artist], "artist") or other_artist
                        questionary.print(f"Fuzz match {ratio:.1f} for {artist}#{this_count} to {target}#{other_count}")
                        for t in _tracks(anode):
                            t.artist = target
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
                            # write the winner's majority RAW title, never the
                            # clean key (lowercased, apostrophes stripped,
                            # in'->ing rewritten -- a display-name destroyer)
                            target = majority_raw(song_nodes[other_song], "song") or other_song
                            questionary.print(f"Fuzz match {ratio:.1f} for {song}#{this_count} to {target}#{other_count}")
                            for t in _tracks(snode):
                                t.song = target
                                store.add(t)
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

    # clean form -> most common raw spelling, so the rewrite below can display
    # a real name instead of the lowercased clean key
    raw_of: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for t in store.all():
        raw_of[clean_artist(t.artist)][t.artist] += 1

    total_swaps = 0
    for t in maybe_groups:
        raw_prefix, feature = t.artist.split('&', 1)
        prefix = clean_artist(raw_prefix)
        prefix = uncomma_artist(prefix) or prefix

        if prefix in likely_artists:
            total_swaps += 1
            t.artist = (raw_of[prefix].most_common(1)[0][0]
                        if raw_of.get(prefix) else raw_prefix.strip())
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
        "songbook",
        "list",
        "stats",
        "review",
        "tag-review",
        "tag-swap",
        "musicbrainz",
        "restitch",
        "apply-resolutions",
        "unify-artists",
        "fixup",
        "fix-artist",
        "fix-unknown",
        "key-detect",
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
    # key-detect can use the network (AcoustID/AcousticBrainz) but degrades to
    # fully offline, so it's marked online to flag the optional internet use.
    online = {"musicbrainz", "restitch", "key-detect"}  # need internet, flagged in the menu

    # Menu layout: workflow-ordered sections for the main screen; the granular
    # one-off cleaners live in an Advanced submenu (All-clean chains them). The
    # flat `options` list above stays the source of truth for the docs test.
    sections = [
        ("Build library", ["search", "detail", "refresh"]),
        ("Clean & identify  (run in order)",
         ["all-clean", "tag-swap", "tag-review", "musicbrainz", "restitch",
          "apply-resolutions"]),
        ("Review & fix", ["review", "fixup", "fix-artist", "fix-unknown"]),
        ("Organize", ["unify-artists"]),
        ("Inspect", ["browse", "artist", "song", "list", "stats"]),
        ("Output", ["key-detect", "final-final", "songbook"]),
    ]
    advanced = ["clean", "tag-fill", "unswap", "uncomma", "trailing-article",
                "ungroup", "fuzz", "fuzz_song"]

    def _label(key):
        return key.capitalize() + (" (online)" if key in online else "")

    # Guard against drift: every documented option must be placed in the layout.
    _placed = {k for _, keys in sections for k in keys} | set(advanced) | {"exit"}
    assert _placed == set(options), f"menu layout != options: {_placed ^ set(options)}"

    main_choices = []
    for _title, _keys in sections:
        main_choices.append(questionary.Separator(f"── {_title} ──"))
        main_choices.extend(Choice(_label(k), value=k) for k in _keys)
    main_choices.append(questionary.Separator("── More ──"))
    main_choices.append(Choice("Advanced cleanup ▸", value="__advanced__"))
    main_choices.append(Choice("Exit", value="exit"))

    advanced_choices = [Choice(_label(k), value=k) for k in advanced]
    advanced_choices.append(Choice("(back)", value="__back__"))

    while True:
        result = questionary.select(
            "Select an option:",
            choices=main_choices,
        ).ask()

        if result is None or result == "exit":
            break
        if result == "__advanced__":
            result = questionary.select(
                "Advanced cleanup:", choices=advanced_choices
            ).ask()
            if result is None or result == "__back__":
                continue

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
        elif result == "key-detect":
            detect_keys(store)
        elif result == "final-final":
            tracks_to_keep(store)
        elif result == "songbook":
            songbook(store)
        elif result == "list":
            report_track_count(store)
        elif result == "stats":
            library_stats(store)
        elif result == "review":
            review_mode(store)
        elif result == "tag-review":
            auto_ok_from_tags(store)
        elif result == "tag-swap":
            swap_from_tags(store)
        elif result == "musicbrainz":
            musicbrainz_lookup(store)
        elif result == "restitch":
            restitch_titles(store)
        elif result == "apply-resolutions":
            apply_resolutions(store)
        elif result == "unify-artists":
            unify_artists(store)
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
    except ValueError as exc:
        # An EXISTING cache that will not load must never be silently replaced:
        # continuing with an empty store means the save-on-exit below would
        # overwrite the whole library index with nothing. Refuse and let the
        # user repair/restore it (a missing cache loads empty and is fine).
        print(f"Cache at {_CACHE_PATH} exists but cannot be loaded: {exc}")
        print("Repair or remove it (restore from a backup) and rerun.")
        return

    try:
        run_interactive(store)
    finally:
        store.save(_CACHE_PATH)
    print("Goodbye.")


if __name__ == "__main__":
    main()
