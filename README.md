# Song sorter

A one off tool for cleaning a karaoke library (ZIP / CDG+MP3 pairs), matching up duplicates and writing back one copy per artist+song.

## Setup

Requires **Python 3.9+**.

```text
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```text
python main.py
```

This launches an interactive menu (see **Main Menu Options** below). Everything
operates on a persistent store, so you can stop and resume at any point.

### Typical workflow

1. **Search** a folder to add tracks to the store.
2. **Detail** to extract metadata and ID3 tags (incremental — safe to re-run).
3. **Tag-fill**, **Tag-review**, **Tag-swap** — use the embedded ID3 tags to
   fill missing artists, auto-accept already-correct parses, and fix reversed
   ones. These clear the bulk of the work automatically.
4. **All-clean** to run the automated text-cleanup chain.
5. **Review** / **Fixup** to hand-fix whatever automation couldn't resolve.
6. **Final-final** to write one best copy per artist+song to the output tree.

## Main Menu Options

The interactive menu (`python main.py`) operates on a persistent track store
(`.cache/song-sorter/cache.json`), loaded on start and saved on exit.

### Building the library
- **Search** — Walk a folder for `.zip` and `.cdg` files, parse an
  `artist`/`song` from each filename, and add them to the store. This is how
  tracks first enter the library.
- **Detail** — For tracked files under a folder, extract technical metadata:
  file sizes, an MP3 SHA-256 hash, a CDG CRC-32 fingerprint, MP3
  bitrate/length/sample-rate/channels, and ID3 tags (artist/title/album/
  year/genre). Uses a delta cache — files whose size+mtime are unchanged
  since the last run are skipped, so re-runs are incremental and resumable.
- **Refresh** — Re-parse `artist`/`song` from filenames on a repeat walk.
  Uses the set of known artists (those with ≥5 tracks) to decide which side
  of an `A - B` name is the artist versus the song, and corrects the record.

### Browsing & inspecting
- **Browse** — Walk the library interactively by **file path** (folder tree).
- **Artist** — Browse the library grouped by **artist**.
- **Song** — Browse the library grouped by **song title**.
- **List** — Print a summary: distinct `artist - song` pairs versus total
  track count (ignoring unknown artists).

### Reviewing & manual fixes
- **Review** — Step through tracks from "thin" artists (artists with ≤5
  tracks) one at a time. For each you can mark **ok**, **swap** artist/song,
  **edit** fields, or **auto-clean** the artist. Decisions are saved to a
  persistent review state and reapplied to identical entries within a session.
- **Tag-review** — Automatically mark review tracks **ok** when the parsed
  artist is corroborated by the MP3's ID3 `tag_artist` (matched ignoring word
  order, catalog prefixes, karaoke suffixes, and `&`/`and`/`feat`). Clears the
  large "parse was already right" portion of the queue. Non-destructive — only
  writes the review-state flag.
- **Tag-swap** — Fix reversed parses using the tag as evidence: when
  `tag_artist` matches the **song** field (not the artist) and that value is a
  known artist (≥3 tracks) while the current artist is not, swap artist/song
  and mark reviewed. The known-artist check guards against mislabeled tags
  (e.g. Disney tracks whose ID3 artist is the song title), which are left for
  manual review.
- **Musicbrainz** *(online)* — Last-resort corroboration for tracks with no
  usable tag. Queries MusicBrainz by artist + title (and the reversed
  orientation, to catch swapped parses): a confident match marks the track
  **ok** (swapping if needed); a record that exists but diverges far from our
  data is flagged in metadata (`mb_artist`/`mb_title`) for review rather than
  applied; no match is left for manual review. **Offline-safe** — it skips
  cleanly without internet, is rate-limited and cached/resumable, and is never
  part of an offline batch.
- **Fixup** — Browse only the not-yet-reviewed tracks from thin artists,
  grouped by artist, for editing.
- **Fix-artist** — Browse all artists; drilling into one lets you **bulk
  rename** that artist across every track under it.
- **Fix-unknown** — For tracks with no/unknown artist, attempt to recover an
  artist by splitting the song/path on delimiters and matching the result
  against known artists.

### Automated cleanup
- **All-clean** — Run the full cleanup chain in sequence: Clean → Tag-fill →
  Uncomma → Ungroup → Fuzz → Fuzz_song.
- **Clean** — Strip common karaoke descriptors (e.g. `wvocal`, `(Wobgv)`,
  `(Instrumental)`, `(Duet)`) from artist/song fields, recording what was
  removed in `style` metadata. Also clears bare track-number artists (a 1–2
  digit `artist` like `04`, left by filenames such as `EZH-31 - 04 - Milkshake`)
  to `Unknown`, preserving the number and catalog id in metadata.
- **Tag-fill** — Fill `Unknown`/empty artists from the MP3's ID3 `tag_artist`,
  only for clean real-looking names (additive — never overwrites an existing
  artist). Ambiguous tags (bare numbers, catalog-style IDs) are kept and
  flagged in metadata (`artist_review`) for later review rather than guessed at.
- **Unswap** — Find tracks whose **song** field is actually a known artist
  name (and whose artist field isn't), and swap them — applied only where ≥3
  such tracks in the same folder agree.
- **Uncomma** — Convert `"Last, First"` artist names to `"First Last"` when
  the swapped form matches a known artist.
- **Trailing-article** — Move a trailing article to the front in artist and
  song fields (`"Models, The"` → `"The Models"`, `"Whole New World, A"` →
  `"A Whole New World"`). Deterministic — only fires when a field *ends* in
  `", The/A/An"`, so multi-comma names like `"Earth, Wind & Fire"` are untouched.
- **Ungroup** — For `"Artist & Someone & Else"`, keep the primary artist and
  move the trailing collaborators into a `feature` field.
- **Fuzz** — Merge near-duplicate **artist** spellings (fuzzy ratio ≥ 90),
  folding the rarer spelling into the more common one (artists with >5 tracks
  are assumed canonical and skipped).
- **Fuzz_song** — The same fuzzy merge for **song titles** within one artist
  (ratio ≥ 85).

### Output
- **Final-final** — For each artist/song group, pick the best copy (largest
  MP3) and copy its files into an output tree organized as
  `<output>/<first-letter>/<artist>/<name>.<ext>`, skipping unknown artists.
- **Exit** — Leave the menu (the store is saved on the way out).

## Data & files

State lives under `.cache/song-sorter/` (git-ignored):

- **`cache.json`** — the track store: one record per track with its parsed
  `artist`/`song`, `file_types`, and a `metadata` dict (hashes, sizes, MP3 info,
  ID3 `tag_*` fields, and provenance markers like `artist_from` / `artist_review`).
  Written atomically, and checkpointed during long **Detail** runs.
- **`review-state.json`** — per-track review decisions (e.g. `ok`), kept
  separate from the track data so re-running cleanup never loses review progress.

Delete `cache.json` to force a full rebuild (re-run **Search** then **Detail**).

## Development

Run the tests with [pytest](https://pytest.org):

```text
pip install pytest
pytest
```

`test_docs.py` guards against documentation drift: it fails if a menu option in
`main.py` is missing from the **Main Menu Options** reference above.

## Stack

Python 3 with 
- [questionary](https://github.com/tmbo/questionary) (prompts), 
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) (string similarity), 
- [mutagen](https://mutagen.readthedocs.io/) (MP3 metadata),
- [tqdm](https://tqdm.github.io/) (progress bars).
