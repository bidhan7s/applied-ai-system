"""
Command line runner for the Music Recommender Simulation.

Includes both the baseline rule-based recommendations and AI-enhanced
recommendations that generate a grounded, guardrail-validated explanation
for each recommended song via the RAG + LLM + guardrails pipeline.
"""

from src.recommender import load_songs, recommend_songs
from src.rag import SongNoteStore
from src.llm_agent import generate_explanation
from src.guardrails import validate_explanation


PROFILES = {
    "High-Energy Pop": {"favorite_genre": "pop", "favorite_mood": "happy", "target_energy": 0.9},
    "Chill Lofi": {"favorite_genre": "lofi", "favorite_mood": "calm", "target_energy": 0.2},
    "Deep Intense Rock": {"favorite_genre": "rock", "favorite_mood": "intense", "target_energy": 0.85},
    "Adversarial (conflicting prefs)": {"favorite_genre": "rock", "favorite_mood": "sad", "target_energy": 0.9},
}


def print_recommendations(profile_name, user_prefs, songs, k=5):
    print(f"\n=== Recommendations for: {profile_name} ===")
    recommendations = recommend_songs(user_prefs, songs, k)
    for rec in recommendations:
        song, score, explanation = rec
        print(f"{song['title']} - Score: {score:.2f}")
        print(f"Because: {explanation}")


def print_ai_recommendations(profile_name, user_prefs, songs, note_store, k=3):
    print(f"\n=== AI-Enhanced Recommendations for: {profile_name} ===")
    recommendations = recommend_songs(user_prefs, songs, k)
    for rec in recommendations:
        song, score, explanation = rec
        context = note_store.retrieve_context(int(song["id"]), k=2)
        llm_result = generate_explanation(user_prefs, song, score, context)
        validated = validate_explanation(user_prefs, song, score, llm_result)

        print(f"\n{song['title']} - Score: {score:.2f}")
        print(f"Final explanation ({validated['source']}): {validated['final_text']}")
        for check in validated["checks"]:
            status = "PASS" if check["passed"] else "FAIL"
            print(f"  [{status}] {check['check']}: {check['reason']}")


def main() -> None:
    songs = load_songs("./data/songs.csv")
    print(f"Loaded songs: {len(songs)}")

    for name, prefs in PROFILES.items():
        print_recommendations(name, prefs, songs)

    note_store = SongNoteStore("./data/song_notes.md")
    for name, prefs in PROFILES.items():
        print_ai_recommendations(name, prefs, songs, note_store)


if __name__ == "__main__":
    main()