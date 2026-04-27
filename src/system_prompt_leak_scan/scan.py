"""Detect system-prompt leakage in LLM model output.

Three detection layers, run in order, all contributing to the same
``matches`` list:

1. **Known patterns** -- a built-in regex set covering common giveaway
   phrasings: "As an AI assistant, my system prompt is...", "I am
   instructed to...", "My initial instructions say...", etc.
2. **Configured prompt match** -- if ``system_prompt`` is supplied:
   * full-substring match (case-sensitive) -> ``"system_prompt_substring"``
   * partial overlap above ``partial_threshold`` of the prompt's tokens
     appearing in ``text`` -> ``"system_prompt_partial"``
3. **Fingerprints** -- if ``fingerprints`` is supplied, each unique-to-prompt
   phrase is scanned case-insensitively as a substring.

The output severity is bucketed:

* ``"none"``     -- no matches.
* ``"low"``      -- only "known pattern" matches (the model talked ABOUT
  having a prompt, but didn't reveal its contents).
* ``"medium"``   -- one fingerprint matched OR a partial prompt overlap.
* ``"high"``     -- a full system-prompt substring leaked, or two-or-more
  fingerprints matched (strong evidence of full leakage).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Set

# Known leakage phrasings -- conservative, only patterns that are themselves
# strong evidence the model is referencing its hidden instructions.
_KNOWN_PATTERNS = [
    re.compile(
        r"\b(?:my|the)\s+system\s+prompt\s+(?:is|says|reads|begins\s+with)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bas\s+an\s+ai\s+(?:assistant|model|language\s+model)\s*[,:]?\s*my\s+(?:system\s+prompt|initial\s+instructions|instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:i\s+(?:am|was)\s+(?:told|instructed|programmed)\s+(?:to|that))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bmy\s+(?:initial|original|hidden)\s+instructions?\s+(?:are|is|say|reads?)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bi'?m\s+not\s+(?:supposed|allowed)\s+to\s+(?:reveal|share|disclose)\s+(?:my\s+)?(?:system\s+prompt|instructions)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bdeveloper\s+(?:instructions?|message)\s+(?:says?|is|reads?)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bhere\s+(?:is|are)\s+my\s+(?:system\s+prompt|instructions)\b", re.IGNORECASE),
]

_TOKEN_RE = re.compile(r"\w+")


@dataclass
class Match:
    """A single leakage finding.

    Attributes:
        type: One of ``"known_pattern"``, ``"system_prompt_substring"``,
            ``"system_prompt_partial"``, ``"fingerprint"``.
        text: The matched text (truncated to ~200 chars for log safety).
        start: Character offset in the scanned text.
        end: Character offset in the scanned text.
    """

    type: str
    text: str
    start: int
    end: int


@dataclass
class ScanResult:
    """Structured result returned by :func:`scan`.

    Attributes:
        leaked: ``True`` iff any match was found.
        matches: List of all findings (in detection order).
        severity: ``"none"`` / ``"low"`` / ``"medium"`` / ``"high"``.
    """

    leaked: bool
    matches: List[Match] = field(default_factory=list)
    severity: str = "none"


def scan(
    text: str,
    system_prompt: Optional[str] = None,
    fingerprints: Optional[Sequence[str]] = None,
    *,
    partial_threshold: float = 0.6,
    min_substring_len: int = 30,
) -> ScanResult:
    """Scan ``text`` for system-prompt leakage.

    Args:
        text: Model output to inspect.
        system_prompt: The verbatim system prompt configured for the call.
            Used for substring + partial-overlap detection.
        fingerprints: Iterable of unique-to-prompt phrases that should never
            appear in normal output. Case-insensitive substring match.
        partial_threshold: Minimum fraction of system-prompt tokens that
            must appear (by token-set membership) in ``text`` to fire the
            ``system_prompt_partial`` signal. Default ``0.6``.
        min_substring_len: Skip the substring check if the system prompt is
            shorter than this; very short prompts are noisy. Default 30.

    Returns:
        :class:`ScanResult` describing the matches and a severity bucket.
    """
    if text is None:
        text = ""
    if not isinstance(text, str):
        raise TypeError("scan: text must be a string")

    matches: List[Match] = []

    # 1. Known patterns (model TALKING about its prompt).
    for rx in _KNOWN_PATTERNS:
        for m in rx.finditer(text):
            matches.append(
                Match(
                    type="known_pattern",
                    text=_clip(m.group(0)),
                    start=m.start(),
                    end=m.end(),
                )
            )

    # 2. Configured-prompt comparisons.
    sp_full_hit = False
    sp_partial_hit = False
    if isinstance(system_prompt, str) and system_prompt.strip():
        sp = system_prompt.strip()
        if len(sp) >= min_substring_len and sp in text:
            idx = text.find(sp)
            matches.append(
                Match(
                    type="system_prompt_substring",
                    text=_clip(sp),
                    start=idx,
                    end=idx + len(sp),
                )
            )
            sp_full_hit = True
        # Partial: tokenize the prompt, compare against text token set.
        # Done even when the full substring matched -- both signals together
        # are strictly more informative than just one.
        prompt_tokens = _token_set(sp)
        text_tokens = _token_set(text)
        if prompt_tokens:
            overlap = len(prompt_tokens & text_tokens) / len(prompt_tokens)
            if overlap >= partial_threshold and not sp_full_hit:
                # We don't have a single span, so report the whole text
                # range as the partial-overlap region.
                matches.append(
                    Match(
                        type="system_prompt_partial",
                        text=f"overlap={round(overlap * 100) / 100}",
                        start=0,
                        end=len(text),
                    )
                )
                sp_partial_hit = True

    # 3. Fingerprints (caller-supplied unique phrases).
    fp_hit_count = 0
    if fingerprints:
        text_lower = text.lower()
        seen_spans: Set[int] = set()
        for fp in fingerprints:
            if not isinstance(fp, str) or not fp.strip():
                continue
            needle = fp.strip().lower()
            idx = text_lower.find(needle)
            if idx == -1:
                continue
            if idx in seen_spans:
                continue
            seen_spans.add(idx)
            matches.append(
                Match(
                    type="fingerprint",
                    text=_clip(fp),
                    start=idx,
                    end=idx + len(needle),
                )
            )
            fp_hit_count += 1

    leaked = bool(matches)
    severity = _severity(
        leaked=leaked,
        sp_full_hit=sp_full_hit,
        sp_partial_hit=sp_partial_hit,
        fp_hit_count=fp_hit_count,
        known_only=all(m.type == "known_pattern" for m in matches) if matches else True,
    )
    return ScanResult(leaked=leaked, matches=matches, severity=severity)


def _token_set(text: str) -> set:
    return {tok.lower() for tok in _TOKEN_RE.findall(text)}


def _clip(text: str, limit: int = 200) -> str:
    """Truncate match text so logs stay readable."""
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def _severity(
    *,
    leaked: bool,
    sp_full_hit: bool,
    sp_partial_hit: bool,
    fp_hit_count: int,
    known_only: bool,
) -> str:
    if not leaked:
        return "none"
    if sp_full_hit or fp_hit_count >= 2:
        return "high"
    if sp_partial_hit or fp_hit_count == 1:
        return "medium"
    if known_only:
        return "low"
    return "medium"
