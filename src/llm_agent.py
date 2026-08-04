"""
LLM explanation layer for Project 4.

Generates a short, grounded explanation of why a song suits a user's
preferences, using facts about the song and the RAG-retrieved context. Falls
back to a deterministic offline stand-in whenever the Claude API is
unavailable or fails, so the recommendation pipeline never crashes.
"""

import os
import re

SYSTEM_PROMPT = (
    "You write exactly ONE sentence (max 30 words) explaining why a song suits "
    "a listener's preferences. Use ONLY facts drawn from the song's attributes "
    "(genre, mood, energy, tempo_bpm, valence, danceability, acousticness) and "
    "the retrieved context provided to you. Do not invent facts, genres, "
    "instruments, or details that are not present in that information."
)


def _offline_stand_in(song: dict, context: str) -> str:
    stripped = re.sub(r"^#+.*$", "", context, flags=re.MULTILINE)
    sentences = re.split(r"(?<=[.!?])\s+", stripped.strip())
    first_sentence = next((s.strip() for s in sentences if s.strip()), "")
    return f"[OFFLINE STAND-IN] {song['title']} by {song['artist']}: {first_sentence}"


def generate_explanation(user_prefs: dict, song: dict, score: float, context: str) -> dict:
    """
    Returns {"text": str, "mode": str} explaining why `song` suits `user_prefs`.
    `mode` is "online" when the Claude API produced the text, "offline" otherwise.
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"text": _offline_stand_in(song, context), "mode": "offline"}

    try:
        import anthropic

        client = anthropic.Anthropic(api_key=api_key)
        user_message = (
            f"User preferences: {user_prefs}\n"
            f"Song: {song}\n"
            f"Match score: {score}\n"
            f"Retrieved context:\n{context}\n\n"
            "Write the one-sentence grounded explanation."
        )
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=150,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        text = next(
            (block.text for block in response.content if block.type == "text"), ""
        ).strip()
        if not text:
            return {"text": _offline_stand_in(song, context), "mode": "offline"}
        return {"text": text, "mode": "online"}
    except Exception:
        return {"text": _offline_stand_in(song, context), "mode": "offline"}
