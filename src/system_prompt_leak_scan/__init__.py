"""system_prompt_leak_scan -- detect system prompt leakage in model outputs.

Public surface (Python port of the JS sibling):

    from system_prompt_leak_scan import scan, ScanResult, Match

* ``scan(text, system_prompt=None, fingerprints=None)`` -- returns a
  :class:`ScanResult` describing detected leaks.
* ``Match`` -- per-finding record with type, span, and matched text.
* ``ScanResult`` -- structured result dataclass.

Three detection layers run in parallel:

1. Known leakage phrases ("As an AI assistant, my system prompt is...").
2. Exact substring of the configured ``system_prompt`` appearing in ``text``,
   plus partial-overlap detection above ``partial_threshold`` (default 0.6).
3. ``fingerprints`` -- caller-supplied unique-to-prompt phrases, scanned
   case-insensitively as substrings.

Zero runtime dependencies, stdlib only.
"""

from .scan import Match, ScanResult, scan

__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "Match",
    "ScanResult",
    "scan",
]
