"""
Guardrail layer for Project 4.

Validates an LLM-generated song explanation against the underlying song data
before it reaches the user, falling back to a deterministic explanation
built from score_song()'s reasons when validation fails.
"""

from typing import Dict, List, Tuple

from src.recommender import score_song

ENERGY_HIGH_WORDS = ["energetic", "high-energy", "intense", "upbeat"]
ENERGY_LOW_WORDS = ["calm", "mellow", "chill", "quiet", "relaxing"]
MAX_WORDS = 60


def _check_existence(text: str, song: Dict) -> Tuple[bool, str]:
    lower = text.lower()
    title = str(song.get("title", "")).lower()
    artist = str(song.get("artist", "")).lower()
    if title and title in lower:
        return True, f"text mentions song title '{song['title']}'"
    if artist and artist in lower:
        return True, f"text mentions artist '{song['artist']}'"
    return False, "text does not mention song title or artist"


def _check_energy_consistency(text: str, energy: float) -> Tuple[bool, str]:
    lower = text.lower()
    found_high = [w for w in ENERGY_HIGH_WORDS if w in lower]
    found_low = [w for w in ENERGY_LOW_WORDS if w in lower]

    if not found_high and not found_low:
        return True, "no energy-descriptor words found; check passes trivially"

    problems = []
    if found_high and energy < 0.6:
        problems.append(f"used high-energy word(s) {found_high} but energy={energy} < 0.6")
    if found_low and energy > 0.4:
        problems.append(f"used low-energy word(s) {found_low} but energy={energy} > 0.4")

    if problems:
        return False, "; ".join(problems)
    return True, f"energy descriptor(s) consistent with energy={energy}"


def _check_sanity(text: str) -> Tuple[bool, str]:
    if not text or not text.strip():
        return False, "text is empty"
    word_count = len(text.split())
    if word_count >= MAX_WORDS:
        return False, f"text has {word_count} words, expected under {MAX_WORDS}"
    return True, f"text has {word_count} words"


def _build_fallback_text(user_prefs: Dict, song: Dict) -> str:
    _, reasons = score_song(user_prefs, song)
    if reasons:
        return f"{song['title']} by {song['artist']} is recommended because of {', '.join(reasons)}."
    return f"{song['title']} by {song['artist']} has no strong matches with your preferences."


def validate_explanation(user_prefs: Dict, song: Dict, score: float, llm_result: Dict) -> Dict:
    text = llm_result.get("text", "")
    checks: List[Dict] = []

    passed, reason = _check_existence(text, song)
    checks.append({"check": "existence", "passed": passed, "reason": reason})

    passed, reason = _check_energy_consistency(text, song["energy"])
    checks.append({"check": "energy_consistency", "passed": passed, "reason": reason})

    passed, reason = _check_sanity(text)
    checks.append({"check": "sanity", "passed": passed, "reason": reason})

    if all(c["passed"] for c in checks):
        return {"final_text": text, "source": "llm", "checks": checks}

    return {
        "final_text": _build_fallback_text(user_prefs, song),
        "source": "fallback",
        "checks": checks,
    }
