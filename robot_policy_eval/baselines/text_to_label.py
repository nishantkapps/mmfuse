"""Map free-form model text to a command index (same heuristics as frozen SData eval)."""
from __future__ import annotations

import difflib


def posthoc_text_to_label(raw_text: str, commands: list[str]) -> tuple[int, str]:
    raw = (raw_text or "").strip()
    if not raw:
        return 0, "empty_fallback"
    raw_l = raw.lower()
    for i, c in enumerate(commands):
        cl = c.lower()
        if cl in raw_l or raw_l in cl:
            return i, "substring"
    best = difflib.get_close_matches(raw, commands, n=1, cutoff=0.25)
    if best:
        return commands.index(best[0]), "fuzzy_full"
    first = raw.split("\n")[0].strip()
    best2 = difflib.get_close_matches(first, commands, n=1, cutoff=0.2)
    if best2:
        return commands.index(best2[0]), "fuzzy_first_line"
    scores = []
    for i, c in enumerate(commands):
        ws = set(raw_l.split())
        cs = set(c.lower().split())
        scores.append((len(ws & cs), i))
    scores.sort(reverse=True)
    return scores[0][1], "token_overlap"
