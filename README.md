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

## Stack

Python 3 with 
- [questionary](https://github.com/tmbo/questionary) (prompts), 
- [rapidfuzz](https://github.com/rapidfuzz/RapidFuzz) (string similarity), 
- [mutagen](https://mutagen.readthedocs.io/) (MP3 metadata),
- [tqdm](https://tqdm.github.io/) (progress bars).
