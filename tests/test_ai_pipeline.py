from src.rag import SongNoteStore
from src.llm_agent import generate_explanation
from src.guardrails import validate_explanation

NOTES_PATH = "data/song_notes.md"


def make_song():
    return {
        "id": 1,
        "title": "Sunrise City",
        "artist": "Neon Echo",
        "genre": "pop",
        "mood": "happy",
        "energy": 0.82,
        "tempo_bpm": 118,
        "valence": 0.84,
        "danceability": 0.79,
        "acousticness": 0.18,
    }


def make_user_prefs():
    return {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.8}


def test_song_note_store_loads_all_notes_without_error():
    store = SongNoteStore(NOTES_PATH)
    assert len(store.notes) == 18
    assert all(isinstance(text, str) and text.strip() for text in store.notes.values())


def test_retrieve_context_contains_known_song_title():
    store = SongNoteStore(NOTES_PATH)
    context = store.retrieve_context(1, k=2)
    assert "Sunrise City" in context


def test_validate_explanation_returns_llm_for_well_formed_explanation():
    song = make_song()
    user_prefs = make_user_prefs()
    llm_result = {
        "text": "Sunrise City is an energetic, upbeat pop track that suits your happy mood.",
        "mode": "online",
    }

    result = validate_explanation(user_prefs, song, score=4.5, llm_result=llm_result)

    assert result["source"] == "llm"
    assert all(check["passed"] for check in result["checks"])


def test_validate_explanation_falls_back_for_wrong_song():
    song = make_song()
    user_prefs = make_user_prefs()
    llm_result = {
        "text": "Midnight Coding is an energetic pop banger!",
        "mode": "fabricated-for-test",
    }

    result = validate_explanation(user_prefs, song, score=4.5, llm_result=llm_result)

    assert result["source"] == "fallback"
    failed_checks = [check["check"] for check in result["checks"] if not check["passed"]]
    assert "existence" in failed_checks


def test_validate_explanation_falls_back_for_mismatched_energy():
    song = make_song()
    user_prefs = make_user_prefs()
    llm_result = {
        "text": "Sunrise City is a calm, mellow track perfect for relaxing.",
        "mode": "fabricated-for-test",
    }

    result = validate_explanation(user_prefs, song, score=4.5, llm_result=llm_result)

    assert result["source"] == "fallback"
    failed_checks = [check["check"] for check in result["checks"] if not check["passed"]]
    assert "energy_consistency" in failed_checks


def test_generate_explanation_never_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    song = make_song()
    user_prefs = make_user_prefs()
    store = SongNoteStore(NOTES_PATH)
    context = store.retrieve_context(1, k=2)

    result = generate_explanation(user_prefs, song, score=4.5, context=context)

    assert result["mode"] == "offline"
    assert isinstance(result["text"], str)
    assert result["text"].strip() != ""
