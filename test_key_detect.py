"""Tests for the pure core of key_detect: KS correlation, key parsing, fusion.

These deliberately avoid the optional audio stack (librosa) and the network; they
exercise only the stdlib-only functions that carry the correctness weight.
"""

from key_detect import (
    _KS_MAJOR,
    _KS_MINOR,
    KEY_NAMES,
    combine_key_signals,
    key_index_fields,
    keys_agree,
    normalize_key,
    pcp_to_key,
    should_emit,
    to_camelot,
)


def _rotated(profile: list[float], tonic_pc: int) -> list[float]:
    """A tonic-relative KS profile placed with its tonic at `tonic_pc`."""
    return [profile[(i - tonic_pc) % 12] for i in range(12)]


# --- pcp_to_key -------------------------------------------------------------

def test_pcp_exact_minor_profile_resolves_confidently() -> None:
    # A profile that IS the A-minor template must resolve to A minor, high conf.
    pcp = _rotated(_KS_MINOR, KEY_NAMES.index("A"))
    r = pcp_to_key(pcp)
    assert r["key"] == "A minor"
    assert r["confidence"] >= 0.5


def test_pcp_exact_major_profile_resolves_confidently() -> None:
    pcp = _rotated(_KS_MAJOR, KEY_NAMES.index("C"))
    r = pcp_to_key(pcp)
    assert r["key"] == "C major"
    assert r["confidence"] >= 0.5


def test_pcp_flat_profile_is_zero_confidence() -> None:
    # No tonal information -> every correlation is 0 -> no margin -> conf 0.
    r = pcp_to_key([1.0] * 12)
    assert r["confidence"] == 0.0


def test_pcp_all_tonics_recovered() -> None:
    # Every one of the 24 templates should recover its own key.
    for tonic in range(12):
        for profile, mode in ((_KS_MAJOR, "major"), (_KS_MINOR, "minor")):
            r = pcp_to_key(_rotated(profile, tonic))
            assert r["key"] == f"{KEY_NAMES[tonic]} {mode}"


def test_pcp_rejects_wrong_length() -> None:
    import pytest
    with pytest.raises(ValueError):
        pcp_to_key([0.0] * 11)


# --- normalize_key ----------------------------------------------------------

def test_normalize_minor_shorthands() -> None:
    for raw in ("Am", "A minor", "Amin", "a min", "A Minor"):
        assert normalize_key(raw) == "A minor"


def test_normalize_major_shorthands() -> None:
    for raw in ("C", "C major", "Cmaj", "CM", "c  major"):
        assert normalize_key(raw) == "C major"


def test_normalize_sharps_and_flats_fold_together() -> None:
    assert normalize_key("F#m") == "F# minor"
    assert normalize_key("Gbm") == "F# minor"   # Gb == F#
    assert normalize_key("Bb") == "A# major"    # Bb == A#
    assert normalize_key("Bbm") == "A# minor"
    assert normalize_key("Db major") == "C# major"


def test_normalize_camelot() -> None:
    assert normalize_key("8A") == "A minor"
    assert normalize_key("8B") == "C major"
    assert normalize_key("11A") == "F# minor"
    assert normalize_key("5B") == "D# major"    # Eb major


def test_normalize_unknown_and_atonal() -> None:
    for raw in ("", "  ", "off", "o", "none", "-", "xyz", None, "H"):
        assert normalize_key(raw) is None


# --- to_camelot -------------------------------------------------------------

def test_to_camelot_roundtrips_known_keys() -> None:
    assert to_camelot("A minor") == "8A"
    assert to_camelot("C major") == "8B"
    assert to_camelot("Bb major") == "6B"       # A# major
    assert to_camelot("F# minor") == "11A"
    assert to_camelot("garbage") is None


def test_to_camelot_covers_every_key() -> None:
    seen = set()
    for name in KEY_NAMES:
        for mode in ("major", "minor"):
            code = to_camelot(f"{name} {mode}")
            assert code is not None
            seen.add(code)
    assert len(seen) == 24  # all distinct, full wheel


# --- keys_agree -------------------------------------------------------------

def test_keys_agree_is_strict_about_tonic() -> None:
    assert keys_agree("A minor", "Am")
    assert not keys_agree("A minor", "C major")   # relative -> NOT agreement
    assert not keys_agree("A minor", "A major")   # parallel -> NOT agreement
    assert not keys_agree("A minor", None)


# --- combine_key_signals ----------------------------------------------------

def test_combine_override_wins_everything() -> None:
    r = combine_key_signals(override="Dm", tag="C", offline=("E major", 0.9),
                            online="F major")
    assert r == {"key": "D minor", "confidence": 1.0, "source": "manual",
                 "detail": ""}


def test_combine_tag_beats_offline_and_online() -> None:
    r = combine_key_signals(tag="G", offline=("A minor", 0.9), online="B minor")
    assert r["key"] == "G major"
    assert r["source"] == "tag"
    assert r["confidence"] == 0.90


def test_combine_tag_corroborated_by_offline_lifts_confidence() -> None:
    r = combine_key_signals(tag="G", offline=("G major", 0.4))
    assert r["key"] == "G major"
    assert r["source"] == "tag"
    assert r["confidence"] == 0.97


def test_combine_offline_boosted_when_online_agrees() -> None:
    r = combine_key_signals(offline=("E minor", 0.6), online="Em")
    assert r["key"] == "E minor"
    assert r["source"] == "auto"
    assert r["confidence"] == 0.8
    assert "agree" in r["detail"]


def test_combine_offline_notes_online_disagreement_but_keeps_local() -> None:
    r = combine_key_signals(offline=("E minor", 0.6), online="F major")
    assert r["key"] == "E minor"        # local read stands
    assert r["source"] == "auto"
    assert r["confidence"] == 0.6
    assert "disagrees" in r["detail"]


def test_combine_online_only_is_modest_and_flagged() -> None:
    r = combine_key_signals(online="A major")
    assert r["key"] == "A major"
    assert r["source"] == "online"
    assert r["confidence"] == 0.50
    assert "original master" in r["detail"]


def test_combine_nothing_known() -> None:
    r = combine_key_signals()
    assert r["key"] is None
    assert r["source"] == "none"


# --- should_emit ------------------------------------------------------------

def test_should_emit_gating() -> None:
    assert should_emit("manual", 0.0)
    assert should_emit("tag", 0.1)
    assert should_emit("auto", 0.5)
    assert not should_emit("auto", 0.49)
    assert should_emit("online", 0.5)
    assert not should_emit("none", 1.0)


# --- key_index_fields (per-copy index emission) -----------------------------

def test_index_fields_emitted_for_confident_copy() -> None:
    md = {"key": "A minor", "key_confidence": "0.812", "key_source": "auto",
          "key_camelot": "8A"}
    assert key_index_fields(md) == {
        "key": "A minor", "key_confidence": 0.812, "key_source": "auto",
        "key_camelot": "8A",
    }


def test_index_fields_omitted_below_emit_floor() -> None:
    md = {"key": "A minor", "key_confidence": "0.30", "key_source": "auto"}
    assert key_index_fields(md) == {}


def test_index_fields_omitted_when_unkeyed() -> None:
    assert key_index_fields({}) == {}
    assert key_index_fields({"key_source": "none"}) == {}


def test_index_fields_tag_and_manual_always_emit() -> None:
    assert key_index_fields(
        {"key": "C major", "key_confidence": "0.0", "key_source": "manual"}
    )["key"] == "C major"
    assert key_index_fields(
        {"key": "C major", "key_confidence": "0.1", "key_source": "tag"}
    )["key"] == "C major"


def test_index_fields_tolerate_bad_confidence() -> None:
    md = {"key": "A minor", "key_confidence": "not-a-number", "key_source": "manual"}
    assert key_index_fields(md)["key_confidence"] == 0.0


def test_each_copy_publishes_its_own_key_not_a_shared_one() -> None:
    """Alternate karaoke rips are frequently transposed relative to each other,
    so the best copy's key must never leak onto an alternate (KDJ #15 gap)."""
    best = {"key": "A minor", "key_confidence": "0.90", "key_source": "auto"}
    transposed_alt = {"key": "B minor", "key_confidence": "0.88", "key_source": "auto"}
    unkeyed_alt = {"key_source": "none"}

    assert key_index_fields(best)["key"] == "A minor"
    assert key_index_fields(transposed_alt)["key"] == "B minor"
    # An alternate we couldn't key confidently publishes nothing at all, rather
    # than inheriting a sibling's key.
    assert key_index_fields(unkeyed_alt) == {}


# --- graceful degradation when the optional audio stack is absent -----------

def test_detect_key_offline_degrades_without_librosa() -> None:
    import key_detect
    if not key_detect.HAVE_LIBROSA:
        # No audio stack installed -> offline detection returns None, never raises.
        assert key_detect.detect_key_offline("does-not-exist.mp3") is None


def test_analyze_local_never_raises_and_is_picklable() -> None:
    """The worker entry point for the process pool: it must return a result dict
    for any input (missing file, unreadable zip, optional deps absent) rather
    than raising, and must be picklable so ProcessPoolExecutor can dispatch it."""
    import pickle
    import key_detect

    pickle.dumps(key_detect.analyze_local)  # picklable -> usable in a worker

    for base, types in [("does-not-exist.mp3", ["mp3"]),
                        ("does-not-exist.zip", ["zip"]),
                        ("no-audio.cdg", ["cdg"]),
                        ("", [])]:
        out = key_detect.analyze_local(base, types)
        assert out == {"tag": None, "offline": None}


def test_lookup_online_empty_query_is_noop() -> None:
    # Empty artist/title short-circuits before any network call.
    import key_online
    key, detail = key_online.lookup_online("", "")
    assert key is None
    assert detail == "no musicbrainz match"
