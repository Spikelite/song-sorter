# Song sorter

A one off tool for cleaning a karaoke library (ZIP / CDG+MP3 pairs), matching up duplicates and writing back one copy per artist+song.

## Setup

```text
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```text
python main.py
```

The main workflow:
- **search** to scan folders folders
- **detail** adds track metadata

For iterating faster, after those run all the track details are stored in a json blob.

Then clean up matches:
- **all-clean** runs a lot of individual cleanup commands.

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
- **Review** — Step through tracks from "thin" artists (artists with ≤3
  tracks) one at a time. For each you can mark **ok**, **swap** artist/song,
  **edit** fields, or **auto-clean** the artist. Decisions are saved to a
  persistent review state and reapplied to identical entries within a session.
- **Fixup** — Browse only the not-yet-reviewed tracks from thin artists,
  grouped by artist, for editing.
- **Fix-artist** — Browse all artists; drilling into one lets you **bulk
  rename** that artist across every track under it.
- **Fix-unknown** — For tracks with no/unknown artist, attempt to recover an
  artist by splitting the song/path on delimiters and matching the result
  against known artists.

### Automated cleanup
- **All-clean** — Run the full cleanup chain in sequence: Clean → Uncomma →
  Ungroup → Fuzz → Fuzz_song.
- **Clean** — Strip common karaoke descriptors (e.g. `wvocal`, `(Wobgv)`,
  `(Instrumental)`, `(Duet)`) from artist/song fields, recording what was
  removed in `style` metadata.
- **Unswap** — Find tracks whose **song** field is actually a known artist
  name (and whose artist field isn't), and swap them — applied only where ≥3
  such tracks in the same folder agree.
- **Uncomma** — Convert `"Last, First"` artist names to `"First Last"` when
  the swapped form matches a known artist.
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

## Stack

Python 3 with 
- [questionary](https://github.com/tmbo/questionary) (prompts), 
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) (string similarity), 
- [mutagen](https://mutagen.readthedocs.io/) (MP3 metadata),
- [tqdm](https://tqdm.github.io/) (progress bars).
