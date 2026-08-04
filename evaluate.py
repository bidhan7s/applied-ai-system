"""
Evaluation script for the Project 4 recommendation pipeline.

Runs three test cases against the real RAG + LLM + guardrails pipeline: a
normal passing case, an edge case with out-of-distribution preferences, and
a deliberate-failure case that feeds a fabricated bad explanation directly
into the guardrail.

Run with: python3 evaluate.py
"""

from src.recommender import load_songs, recommend_songs
from src.rag import SongNoteStore
from src.llm_agent import generate_explanation
from src.guardrails import validate_explanation
from src.main import PROFILES

results = []


def run_test(name, fn):
    try:
        passed, detail = fn()
    except Exception as e:
        passed, detail = False, f"raised {type(e).__name__}: {e}"
    results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] {name}")
    print(f"    {detail}")
    print()
    return passed


def test_normal_case(songs, note_store):
    user_prefs = PROFILES["High-Energy Pop"]
    song, score, _ = recommend_songs(user_prefs, songs, k=1)[0]
    context = note_store.retrieve_context(int(song["id"]), k=2)
    llm_result = generate_explanation(user_prefs, song, score, context)
    validated = validate_explanation(user_prefs, song, score, llm_result)

    all_passed = all(c["passed"] for c in validated["checks"])
    ok = validated["source"] == "llm" and all_passed
    detail = (
        f"song={song['title']!r}, source={validated['source']}, "
        f"all_checks_passed={all_passed}, final_text={validated['final_text']!r}"
    )
    return ok, detail


def test_edge_case(songs, note_store):
    user_prefs = {"favorite_genre": "classical", "favorite_mood": "epic", "target_energy": 0.9}
    song, score, _ = recommend_songs(user_prefs, songs, k=1)[0]
    context = note_store.retrieve_context(int(song["id"]), k=2)
    llm_result = generate_explanation(user_prefs, song, score, context)
    validated = validate_explanation(user_prefs, song, score, llm_result)

    ok = (
        isinstance(validated.get("final_text"), str)
        and validated["final_text"] != ""
        and validated.get("source") in ("llm", "fallback")
    )
    detail = (
        f"song={song['title']!r}, score={score:.2f} (out-of-distribution prefs: "
        f"genre=classical/mood=epic), source={validated['source']}, "
        f"final_text={validated['final_text']!r}"
    )
    return ok, detail


def test_deliberate_failure(songs, note_store):
    song = next(s for s in songs if s["title"] == "Midnight Coding")
    user_prefs = {"favorite_genre": "lofi", "favorite_mood": "chill", "target_energy": 0.4}
    fabricated_llm_result = {
        "text": "Sunrise City is an energetic banger!",
        "mode": "fabricated-for-test",
    }
    validated = validate_explanation(user_prefs, song, score=1.0, llm_result=fabricated_llm_result)

    failed_checks = [c for c in validated["checks"] if not c["passed"]]
    ok = validated["source"] == "fallback" and len(failed_checks) >= 1
    detail = (
        f"source={validated['source']}, "
        f"failed_checks={[c['check'] for c in failed_checks]}, "
        f"final_text={validated['final_text']!r}"
    )
    return ok, detail


def main():
    songs = load_songs("./data/songs.csv")
    note_store = SongNoteStore("./data/song_notes.md")

    print("=" * 70)
    print("Project 4 Pipeline Evaluation")
    print("=" * 70)
    print()

    run_test(
        "1. NORMAL CASE (High-Energy Pop, top recommendation)",
        lambda: test_normal_case(songs, note_store),
    )
    run_test(
        "2. EDGE CASE (out-of-distribution prefs: classical/epic)",
        lambda: test_edge_case(songs, note_store),
    )
    run_test(
        "3. DELIBERATE FAILURE CASE (fabricated wrong-song/energy text)",
        lambda: test_deliberate_failure(songs, note_store),
    )

    passed_count = sum(1 for _, p, _ in results if p)
    total = len(results)
    print("=" * 70)
    print(f"{passed_count}/{total} tests passed")
    print("=" * 70)


if __name__ == "__main__":
    main()
