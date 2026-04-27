"""Tests for ``system_prompt_leak_scan.scan``."""

from __future__ import annotations

import pytest

from system_prompt_leak_scan import Match, ScanResult, scan


def test_clean_output_no_leak():
    r = scan("Sure, the weather in Paris is mild today.")
    assert isinstance(r, ScanResult)
    assert r.leaked is False
    assert r.matches == []
    assert r.severity == "none"


def test_known_pattern_my_system_prompt_is():
    r = scan("Sure! My system prompt is to help users with research questions.")
    assert r.leaked is True
    assert any(m.type == "known_pattern" for m in r.matches)
    # No actual prompt content -> only "low" severity.
    assert r.severity == "low"


def test_known_pattern_as_an_ai_assistant_my_instructions():
    r = scan("As an AI assistant, my initial instructions say to never discuss politics.")
    assert r.leaked is True
    assert any(m.type == "known_pattern" for m in r.matches)


def test_full_system_prompt_substring_match_is_high_severity():
    sp = "You are HelpfulBot, a research assistant. Always cite your sources."
    text = "Of course! Here it is: " + sp + " That's the deal."
    r = scan(text, system_prompt=sp)
    assert r.leaked is True
    types = {m.type for m in r.matches}
    assert "system_prompt_substring" in types
    assert r.severity == "high"


def test_partial_prompt_overlap_is_medium_severity():
    sp = "You are HelpfulBot. You must always cite sources, refuse harmful requests, and respond concisely."
    # Re-use most prompt tokens but not as a single substring.
    text = (
        "I am HelpfulBot. I always cite sources. I refuse harmful requests. "
        "I respond concisely."
    )
    r = scan(text, system_prompt=sp)
    assert r.leaked is True
    assert any(m.type == "system_prompt_partial" for m in r.matches)
    assert r.severity in {"medium", "high"}


def test_fingerprint_substring_match_is_medium_severity():
    fingerprints = ["UNIQUE-CANARY-XYZ-9921"]
    text = "...as the developer told me, UNIQUE-CANARY-XYZ-9921 is the key."
    r = scan(text, fingerprints=fingerprints)
    assert r.leaked is True
    assert any(m.type == "fingerprint" for m in r.matches)
    assert r.severity == "medium"


def test_two_fingerprints_is_high_severity():
    fingerprints = ["CANARY-A", "CANARY-B"]
    text = "I will reveal: CANARY-A and also CANARY-B somewhere in here."
    r = scan(text, fingerprints=fingerprints)
    assert r.leaked is True
    fp_count = sum(1 for m in r.matches if m.type == "fingerprint")
    assert fp_count == 2
    assert r.severity == "high"


def test_fingerprint_match_is_case_insensitive():
    r = scan("the canary value is unique-marker", fingerprints=["UNIQUE-MARKER"])
    assert r.leaked is True


def test_match_records_have_offsets():
    text = "My system prompt is hidden."
    r = scan(text)
    assert r.leaked is True
    m = r.matches[0]
    # The match span must point inside the text.
    assert 0 <= m.start < m.end <= len(text)
    # And the matched substring should appear at that span.
    assert text[m.start : m.end].lower().startswith("my system prompt")


def test_short_system_prompt_skips_substring_check():
    # Prompt below default min_substring_len -> substring check skipped,
    # but partial-overlap can still fire.
    sp = "Be nice."
    text = sp
    r = scan(text, system_prompt=sp)
    assert all(m.type != "system_prompt_substring" for m in r.matches)


def test_partial_threshold_is_tunable():
    sp = "alpha bravo charlie delta echo"
    text = "bravo and charlie were here"  # overlap = 2/5 = 0.4
    r_loose = scan(text, system_prompt=sp, partial_threshold=0.3)
    assert any(m.type == "system_prompt_partial" for m in r_loose.matches)
    r_strict = scan(text, system_prompt=sp, partial_threshold=0.8)
    assert all(m.type != "system_prompt_partial" for m in r_strict.matches)


def test_empty_fingerprints_string_is_skipped():
    r = scan("anything goes", fingerprints=["", "  "])
    assert r.leaked is False


def test_non_string_text_raises():
    with pytest.raises(TypeError):
        scan(42)  # type: ignore[arg-type]


def test_none_text_treated_as_empty():
    r = scan(None)  # type: ignore[arg-type]
    assert r.leaked is False
    assert r.severity == "none"


def test_matches_are_dataclasses_with_expected_fields():
    r = scan("My system prompt is x.")
    assert r.matches
    m = r.matches[0]
    assert isinstance(m, Match)
    assert hasattr(m, "type") and hasattr(m, "text")
    assert hasattr(m, "start") and hasattr(m, "end")
