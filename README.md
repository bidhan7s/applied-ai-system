# 🎵 Music Recommender Simulation — Project 4: RAG-Grounded LLM Explanations with Guardrails

## Base Project

This project is an extension of the **Music Recommender Simulation (Module
3)**. The original system's goal was to score songs against a user's stated
preferences — favorite genre, favorite mood, and target energy — and return
the highest-scoring songs as recommendations. It does this entirely through
deterministic, rule-based scoring: no machine learning, no LLM calls, just a
fixed point system (genre match, mood match, energy closeness) applied
consistently across the song catalog. That original scoring and ranking
logic (`src/recommender.py`) is unchanged and still powers every
recommendation in this project.

## Title and Summary

**Project 4: RAG-Grounded LLM Explanations with Guardrails** takes the
Module 3 recommender's deterministic rankings and adds a natural-language
explanation layer on top of them — one that's *grounded* in real song data
(via retrieval-augmented generation) and *checked* before it's ever shown to
a user (via a guardrail layer), rather than trusting an LLM's output at face
value. This matters because LLM-generated explanations are easy to get
wrong in subtle, plausible-sounding ways — naming the wrong song, or
describing a high-energy track as "calm" — and a recommender that explains
its choices inaccurately is arguably worse than one that doesn't explain
itself at all. The system still works fully offline (with a deterministic
fallback) so it never depends on an API key or network access to run or be
graded.

## Architecture Overview

The full pipeline is diagrammed in
[`diagrams/architecture.mmd`](diagrams/architecture.mmd) (Mermaid source —
render it with any Mermaid-compatible viewer, e.g. the Mermaid Live Editor
or a Markdown preview extension that supports Mermaid). The flow:

1. **Recommender** (`src/recommender.py`) scores every song in
   `data/songs.csv` against a user profile and ranks them —
   `recommend_songs()` returns the top-k songs with their scores.
2. **RAG retrieval** (`src/rag.py`) — `SongNoteStore` builds a TF-IDF index
   over `data/song_notes.md` and `retrieve_context()` pulls each
   top-ranked song's own note plus its most similar notes, to ground the
   next step in real, factual song descriptions.
3. **LLM explanation** (`src/llm_agent.py`) — `generate_explanation()`
   calls the Claude API if `ANTHROPIC_API_KEY` is set, using the retrieved
   context as grounding; if no key is set (or the call fails), it falls
   back to a deterministic offline stand-in built from that same context.
4. **Guardrails** (`src/guardrails.py`) — `validate_explanation()` runs
   three checks on whatever text came out of step 3: an **existence**
   check (does it mention the song's title/artist), an **energy
   consistency** check (do any energy-descriptor words match the song's
   actual energy value), and a **sanity** check (non-empty, under 60
   words). If any check fails, the explanation is replaced with a
   deterministic sentence built from `score_song()`'s own reasons.
5. **Output** — the final ranked recommendations, each with a validated,
   guardrail-checked explanation and a record of which check(s) passed or
   failed.

## Setup Instructions

```bash
# 1. (optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
.venv\Scripts\activate         # Windows

# 2. install dependencies
pip install -r requirements.txt

# 3. (optional) enable live Claude API calls for explanations
#    the system works fully offline without this — it falls back
#    to a deterministic stand-in explanation instead
export ANTHROPIC_API_KEY="your-key-here"
```

Run the system:

```bash
# baseline + AI-enhanced recommendations for all sample profiles
python3 -m src.main

# scripted evaluation: normal / edge / deliberate-failure cases
python3 evaluate.py

# full pytest suite
pytest tests/ -v
```

## Sample Interactions

These are real, unedited excerpts from `python3 -m src.main`'s AI-enhanced
output, captured in this environment (no `ANTHROPIC_API_KEY` set, so
`source` here is the offline path — with a key set, `source` would show the
same fields but the text would come from the live Claude API instead):

```
Sunrise City - Score: 4.88
Final explanation (llm): [OFFLINE STAND-IN] Sunrise City by Neon Echo: A bright pop track that matches its happy mood with high energy (0.82) and strong valence (0.84).
  [PASS] existence: text mentions song title 'Sunrise City'
  [PASS] energy_consistency: no energy-descriptor words found; check passes trivially
  [PASS] sanity: text has 24 words
```

```
Rooftop Lights - Score: 3.29
Final explanation (llm): [OFFLINE STAND-IN] Rooftop Lights by Indigo Parade: An indie pop track that earns its happy mood with energy at 0.76 and strong valence (0.81).
  [PASS] existence: text mentions song title 'Rooftop Lights'
  [PASS] energy_consistency: no energy-descriptor words found; check passes trivially
  [PASS] sanity: text has 24 words
```

```
Gym Hero - Score: 2.96
Final explanation (llm): [OFFLINE STAND-IN] Gym Hero by Max Pulse: An intense pop track with the highest energy in this range (0.93) and very low acousticness (0.05), pointing to a heavily produced, synthetic sound.
  [PASS] existence: text mentions song title 'Gym Hero'
  [PASS] energy_consistency: energy descriptor(s) consistent with energy=0.93
  [PASS] sanity: text has 31 words
```

The third example is the most telling: unlike the first two, its
`energy_consistency` check wasn't a trivial pass — the text says "intense,"
the guardrail confirmed that word actually matches the song's 0.93 energy
value, and only then let it through.

## Design Decisions

- **Why an offline fallback mode:** the whole point of a class project is
  that it needs to run reliably for grading and demos, without depending on
  a live API key, network access, or incurring API costs every time it's
  run. Building the offline stand-in directly into `generate_explanation()`
  (rather than as a separate mocked-out test path) means the exact same
  code runs in both modes — the only thing that changes is where the text
  comes from, not how it's validated.
- **Why TF-IDF over embeddings for retrieval:** the note corpus is 18 short
  documents. A learned or API-based embedding model would add an external
  dependency (another API key, another network call, another point of
  failure) to solve a retrieval problem that scikit-learn's TF-IDF +
  cosine similarity already solves well at this scale, entirely locally
  and deterministically. Simplicity won over marginal retrieval-quality
  gains that wouldn't be visible in a catalog this small anyway.
- **Why these specific guardrail checks:** existence and energy consistency
  target the two failure modes an LLM is actually likely to produce in
  this domain — naming the wrong song (a plausible-sounding hallucination
  when several songs share a genre or mood) and describing a song's energy
  with a word that contradicts its numeric value (since the model is
  reasoning about a number, not a felt experience). The sanity check is a
  cheap backstop against empty or bloated output. All three are
  deterministic string/number checks rather than a second LLM call judging
  the first — that keeps the guardrail itself trustworthy and fast, and
  avoids paying for (or depending on) another model call just to check the
  first one.

## Testing Summary

- **`python3 evaluate.py`: 3/3 passing.** The normal case confirms a
  well-formed explanation for the top recommendation passes all three
  guardrail checks; the edge case confirms out-of-distribution preferences
  (`genre="classical"`, `mood="epic"`) don't crash the pipeline and still
  produce a valid result; the deliberate-failure case confirms a fabricated
  explanation naming the wrong song with a mismatched energy word is
  actually caught and replaced, not silently passed through.
- **`pytest tests/ -v`: 8/8 passing.** The 6 tests in
  `tests/test_ai_pipeline.py` caught that `SongNoteStore` loads and
  retrieves context correctly (including that a song's own title appears
  in its retrieved context), that `validate_explanation()` correctly
  accepts well-formed text, correctly rejects a wrong-song mention and a
  mismatched energy descriptor (forcing the fallback path in both cases),
  and that `generate_explanation()` never raises even with no API key
  configured; the 2 pre-existing tests in `tests/test_recommender.py`
  continue to confirm the baseline rule-based scorer's ranking and
  explanation behavior is unaffected.

This README does not include an AI-collaboration reflection — that lives in
[`model_card.md`](model_card.md), under **Section 9: Personal Reflection**.

---

# Module 3: Original Recommender Documentation

The sections below are the original Module 3 documentation for the
underlying rule-based recommender, kept as-is for reference.

## Project Summary

This project simulates a simplified content-based music recommender, similar
in spirit to how platforms like Spotify suggest songs — but using only a
song's own attributes (genre, mood, energy) rather than other users'
listening behavior. A user "taste profile" is compared against a small song
catalog, each song is scored for how well it matches, and the top-ranked
results are returned as recommendations.
Real-world recommenders like Spotify or YouTube typically combine two
approaches. **Collaborative filtering** looks at other users' behavior —
if you and another listener both liked the same songs before, it
recommends what that other listener enjoyed, even if the songs don't
share obvious attributes. **Content-based filtering** (what this project
simulates) instead looks at a song's own attributes — genre, mood, tempo,
energy — and matches those against a user's stated or inferred
preferences. Real systems also track behavior signals like skips, saves,
replays, and listening duration to build a taste profile automatically,
rather than asking the user to state their preferences directly. This
simulation simplifies that: it uses only song attributes and a
manually-defined user profile, with no behavior tracking or
other-user data involved.

---

## How The System Works

Each `Song` in this system has the following features:
- **genre** — e.g. pop, rock, lofi
- **mood** — e.g. happy, intense, calm
- **energy** — a 0.0-1.0 scale representing intensity
- **tempo_bpm** — beats per minute

A `UserProfile` stores target preferences for these same fields:
- **favorite_genre**
- **favorite_mood**
- **target_energy**

The `Recommender` scores each song against the user profile using a
weighted rule:
- +1.5 points if the song's genre matches the user's favorite genre
- +2.0 points if the song's mood matches the user's favorite mood
- Up to +1.5 points based on how *close* the song's energy is to the
  user's target energy (closer = more points, not just "higher energy")

I initially weighted genre higher than mood (2.0 vs 1.0), but testing
revealed this let energy closeness override a wrong mood match too easily.
I rebalanced the weights to give mood more influence, which is reflected
in the numbers above — see "Experiments You Tried" below for details.

Every song in the catalog is scored this way, then the list is sorted from
highest to lowest score. The top K songs are returned as the final
recommendations, along with a plain-language list of "reasons" explaining
why each song scored the way it did (e.g. "genre match (+1.5)").

This mirrors real systems in a simplified way: real recommenders also
turn raw attributes into scores and rank across a catalog — just with far
more features and far more users' behavior data feeding in.

---

## Sample Recommendation Output

```
Loaded songs: 18

=== Recommendations for: High-Energy Pop ===
Sunrise City - Score: 4.88
Because: genre match (+1.5), mood match (+2.0), energy close (+1.38)
Rooftop Lights - Score: 3.29
Because: mood match (+2.0), energy close (+1.29)
Gym Hero - Score: 2.96
Because: genre match (+1.5), energy close (+1.46)
Storm Runner - Score: 1.48
Because: energy close (+1.48)
Fuego Tropical - Score: 1.46
Because: energy close (+1.46)

=== Recommendations for: Chill Lofi ===
Library Rain - Score: 2.78
Because: genre match (+1.5), energy close (+1.28)
Focus Flow - Score: 2.70
Because: genre match (+1.5), energy close (+1.2)
Midnight Coding - Score: 2.67
Because: genre match (+1.5), energy close (+1.17)
Spacewalk Thoughts - Score: 1.38
Because: energy close (+1.38)
Moonlight Meditation - Score: 1.38
Because: energy close (+1.38)

=== Recommendations for: Deep Intense Rock ===
Storm Runner - Score: 4.91
Because: genre match (+1.5), mood match (+2.0), energy close (+1.41)
Midnight Cipher - Score: 3.50
Because: mood match (+2.0), energy close (+1.5)
Gym Hero - Score: 3.38
Because: mood match (+2.0), energy close (+1.38)
Fuego Tropical - Score: 1.47
Because: energy close (+1.47)
Sunrise City - Score: 1.46
Because: energy close (+1.46)

=== Recommendations for: Adversarial (conflicting prefs) ===
Storm Runner - Score: 2.98
Because: genre match (+1.5), energy close (+1.48)
Gym Hero - Score: 1.46
Because: energy close (+1.46)
Fuego Tropical - Score: 1.46
Because: energy close (+1.46)
Thunder Forge - Score: 1.44
Because: energy close (+1.44)
Neon Rush - Score: 1.43
Because: energy close (+1.43)
```

---

## Experiments You Tried

I ran an adversarial test profile — rock genre, "sad" mood, 0.9 target
energy — a combination that doesn't really exist cleanly in my dataset.
"Storm Runner" (rock/intense/0.91 energy) ranked #1 despite not matching
mood at all, because its genre and energy match were strong enough to
compensate.

To test this, I changed the weights: genre from 2.0 → 1.5, mood from
1.0 → 2.0, and energy's max from 2.0 → 1.5, to make mood matter more.
Storm Runner's adversarial score dropped from 3.98 to 2.98 and correctly
lost its mood bonus — but it still ranked #1 overall, because there's no
high-energy "sad" rock song anywhere in the catalog to take its place.

This showed me the issue wasn't purely the weighting formula — it was
also a data coverage gap. Reweighting fixed *how much* a wrong-mood song
was penalized, but couldn't produce a better option that didn't exist.

Comparing the three standard profiles side by side: **High-Energy Pop**
consistently surfaced pop/happy tracks with high energy (Sunrise City,
Rooftop Lights), staying tightly within genre and mood matches. **Chill
Lofi** shifted completely to low-energy lofi/ambient tracks (Library
Rain, Focus Flow, Midnight Coding), never surfacing anything above ~0.4
energy. **Deep Intense Rock** locked onto rock/intense tracks (Storm
Runner, Midnight Cipher), even though "intense" and "happy" moods are
conceptually unrelated to genre. This confirms that genre and mood act
as strong gates on which songs even get considered — energy closeness
alone rarely overrides a genre/mood mismatch when both are absent.

---

## Limitations and Risks

- The catalog only has 18 songs, so many genre/mood/energy combinations
  (like "sad high-energy rock") simply don't exist, no matter how the
  scoring is tuned.
- The system doesn't understand lyrics, language, or cultural context —
  it only compares numeric/categorical attributes.
- Mood and genre are tightly coupled in this dataset (e.g. "happy" is
  almost always pop), which risks pushing users toward the same narrow
  slice of the catalog regardless of their actual taste.
- Energy contributes as much or more than genre/mood, so a strong energy
  match can override a mismatched mood — this is discussed in more depth
  in the model card.

You will go deeper on this in your model card.

---

## Reflection

Read and complete `model_card.md`:

[**Model Card**](model_card.md)

Building this recommender showed me that turning data into predictions is
really just a matter of designing a point system and trusting it to rank
consistently — there's no "magic," just weighted comparisons applied at
scale. The more surprising lesson was that bias doesn't only come from
bad weighting; it can come from the dataset itself. Even after I
rebalanced my scoring to fix an obvious flaw (mood being too easy to
override), the system still produced an imperfect result because the
right song simply wasn't in my catalog. That distinction — an algorithm
problem versus a data problem — is exactly the kind of unfairness real
recommender systems can have at a much larger scale, where missing or
underrepresented data quietly limits what certain users ever get shown.
