# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name  

VibeFinder 1.0


---

## 2. Intended Use  

This recommender is designed as a classroom simulation to demonstrate how
content-based music recommendation works — not a production system for
real users. It takes a user's stated genre, mood, and energy preference
and returns the top-K matching songs from a small fixed catalog, along
with an explanation of why each song was chosen. It assumes the user can
articulate their preferences directly as simple categorical/numeric
values, which real users typically cannot do — real systems infer this
from behavior instead. 

---

## 3. How the Model Works  

Each song has a genre, mood, and energy level (0.0-1.0), among other
attributes. A user profile specifies a favorite genre, favorite mood, and
target energy level. The system compares every song in the catalog
against the user's profile using a point system: a genre match adds
points, a mood match adds points, and energy contributes points based on
how close the song's energy is to the user's target (not just whether
it's higher or lower). All songs are scored this way, then sorted highest
to lowest, and the top K are returned along with a plain-language
breakdown of why each one scored the way it did.

I started with genre=2.0, mood=1.0, energy=up to 2.0, then experimented
with rebalancing to genre=1.5, mood=2.0, energy=up to 1.5 after noticing
mood was too easily overridden by energy (see Evaluation section).

---

## 4. Data  

The catalog contains 18 songs spanning genres including pop, rock, hip-hop,
lofi, soul, electronic, folk, metal, reggae, classical, and latin, with
moods ranging from happy and intense to calm, melancholic, and aggressive.
I expanded the original 10-song starter set with 8 additional songs to
increase genre/mood diversity. Even with this expansion, some
genre+mood+energy combinations are missing entirely — for example, there
is no high-energy song with a "sad" mood in the rock genre, which limits
how well the system can serve users with less common preference
combinations.
---


## 5. Strengths

The system works well for users whose preferences align with well-covered
parts of the dataset — high-energy pop, intense rock, and calm lofi
listeners all get recommendations that match genre, mood, and energy
simultaneously, with clear top-ranked results.

The scoring logic correctly distinguishes between similar-sounding profiles:
"High-Energy Pop" and "Deep Intense Rock" produce completely different
top-5 lists even though both target high energy, because genre and mood
still meaningfully shift the ranking.

The explanation output (e.g. "genre match (+1.5), mood match (+2.0), energy
close (+1.41)") makes the reasoning behind each recommendation transparent
— a user can see exactly why a song was suggested, rather than getting an
unexplained black-box ranking.
---

## 6. Limitations and Bias 

My scoring weights let energy closeness (worth as much as genre) override
mood matching, so a song with the wrong mood can still win if its energy is
close enough — I confirmed this directly with an adversarial test profile.

The dataset itself has a deeper problem: mood and genre are tightly coupled.
"Happy" songs are almost all pop, "chill" songs are almost all lofi, and
"romantic" only exists in one soul song. This means a user who wants
"happy rock" or "chill jazz" literally cannot get a good match, no matter
how the scoring weights are tuned — the songs don't exist in the catalog.

The dataset is also skewed toward high-energy songs (7 songs between
0.82-0.95 energy vs only 5 in the middle range), so users targeting high
energy get more close matches than users wanting moderate or low energy.

Overall, the system currently works best for high-energy pop/indie users,
and creates a filter bubble for anyone with mood preferences outside that
narrow, over-represented slice of the catalog — even a perfectly designed
scoring formula can't fix a dataset that doesn't contain the songs a user
actually wants. 

---

## 7. Evaluation

I tested four distinct user profiles: High-Energy Pop, Chill Lofi, Deep
Intense Rock, and an adversarial profile with conflicting preferences
(rock genre + sad mood + high energy).

For the first three, the top result matched genre, mood, and energy all
at once, and the results felt intuitively correct — for example, "Storm
Runner" (rock/intense/0.91 energy) ranked #1 for the Deep Intense Rock
profile, and "Library Rain" (lofi/calm-adjacent/low energy) ranked #1 for
Chill Lofi.

The adversarial profile was the most revealing test. I expected the system
to either fail to find a good match or surface something clearly wrong.
Instead, "Storm Runner" still ranked #1 despite not matching mood, because
its genre and energy match were strong enough to compensate. This
surprised me — I assumed a bad mood match would push a song further down
the list.

I then ran a sensitivity experiment: I rebalanced the scoring weights
(genre 2.0→1.5, mood 1.0→2.0, energy max 2.0→1.5) to make mood matter
more. Storm Runner's adversarial score dropped from 3.98 to 2.98, and it
correctly lost credit for the mood mismatch — but it still ranked #1
overall. This told me the issue wasn't only the weighting formula; it was
that my dataset has no high-energy "sad" rock song at all, so there was
nothing better available to recommend. That distinction — an algorithm
problem versus a data coverage problem — was the most useful thing I
learned from this test.

All existing unit tests continued to pass after the reweighting, confirming
the change didn't break the system's core structure.

---


## 8. Future Work

If I continued developing this, I would prioritize:

1. **Add valence to scoring** — right now two songs can share the same
   mood label ("intense") but feel completely different emotionally
   (dark/aggressive vs. bright/energetic). Using the valence value I
   already have in the dataset would let the system distinguish these.

2. **Expand the dataset to break genre-mood coupling** — right now moods
   like "happy" or "chill" are locked to one or two genres (pop, lofi).
   Adding songs like "happy rock" or "chill jazz" would let users with
   less common combinations get real matches instead of being pushed
   toward whatever genre happens to dominate their mood.

3. **Add an exploration/diversity mechanism** — since genre currently
   carries a lot of weight, the same 2-3 songs tend to dominate results
   for any given mood. Introducing a small amount of randomness or a
   rule that avoids repeating the same top artist/genre too often would
   make recommendations feel less repetitive.
---



## 9. Personal Reflection

My biggest learning moment was during the adversarial testing phase. I
expected that reweighting genre, mood, and energy would be enough to fix
a bad recommendation, but even after rebalancing the weights, "Storm
Runner" still ranked #1 despite the wrong mood — because my dataset simply
had no high-energy "sad" rock song to recommend instead. This taught me
that a scoring formula can only work with what's actually in the data;
tuning weights can't fix a gap that doesn't exist in the catalog.

Using an AI coding assistant helped me move fast — it wrote working
implementations of load_songs, score_song, and recommend_songs almost
instantly, and helped me fix a `ModuleNotFoundError` caused by running
files from the wrong directory, which I wouldn't have diagnosed as quickly
alone. I had to double-check it most when it drafted my model card's bias
section — the first version was full of code blocks, percentages, and
technical tables, when the assignment specifically asked for plain
language a non-programmer could follow, so I rewrote it in simpler terms
myself.

What surprised me is how "smart" a system can feel using nothing but a
handful of point-weighted if-statements — no machine learning, no training
data, just genre/mood/energy comparisons. It made me realize that a lot of
the "personalized" feeling in real recommendation apps might come from
simpler rules than I originally assumed, layered together at a much larger
scale.

If I extended this project, I'd want to try adding valence scoring next,
since I noticed two songs can share the same mood label (like "intense")
but feel completely different emotionally — one dark and aggressive, one
bright and energetic — and my current system treats them identically.

---

## 10. Project 4 Extension: RAG-Grounded LLM Explanations

### 10.1 Dataset

Project 4 uses the same 18-song catalog as VibeFinder 1.0, with each song
carrying seven attributes: genre, mood, energy, tempo_bpm, valence,
danceability, and acousticness. Alongside the CSV, each song also has a
short hand-written note (`data/song_notes.md`) describing its feel in
plain language, grounded in those same attribute values. The intended
purpose of this dataset in Project 4 is **not** production music
recommendation — it's small and hand-written specifically to demonstrate,
in a controllable and checkable way, how grounded LLM explanation
generation works: can an LLM write an accurate-sounding sentence about a
song using only retrieved facts, and can a guardrail actually catch it
when it doesn't?

### 10.2 The Approach, in Plain Language

1. The deterministic scorer (unchanged from VibeFinder 1.0) picks the
   candidate songs — same genre/mood/energy point system as before.
2. For each candidate, a retrieval step (RAG) pulls that song's own
   background note plus its most similar notes, so there's real,
   fact-checkable material to write from.
3. An LLM is asked to write one short sentence explaining why the song
   suits the user — but only using facts from those retrieved notes and
   the song's own attributes, not from general knowledge or guesswork.
4. Before that sentence is shown to anyone, a guardrail step checks it: does
   it actually mention the song (not a different one), does any
   energy-related word it uses match the song's real energy value, and is
   it a reasonably-sized, non-empty sentence? If it fails any of these, the
   system doesn't show the LLM's sentence at all — it falls back to a
   plain, deterministic explanation built from the same point-system
   reasons VibeFinder 1.0 already produces.

The guardrail step is the important addition here: it means the system
never blindly trusts what the LLM wrote — it only shows it after
double-checking it against facts that were already known to be true.

### 10.3 Limitations and Biases (Project 4)

- The dataset is still the same small, hand-written 18-song catalog from
  VibeFinder 1.0 — it isn't representative of the diversity of real music,
  so conclusions about how well the LLM explanations work don't
  necessarily generalize beyond this toy catalog.
- The guardrail's energy-consistency check relies on a hand-coded list of
  descriptor words (like "energetic," "intense," "calm," "mellow"). This
  list is incomplete by construction — an LLM could describe a song as
  "explosive" or "sedate" (words not on the list) and the check would
  simply pass trivially without ever actually verifying that claim,
  because it only checks words it was told to look for.
- The offline stand-in mode isn't a real LLM at all — it's a deterministic
  string-manipulation fallback that always produces a similarly-structured
  sentence ("[OFFLINE STAND-IN] {title} by {artist}: {first sentence of
  the note}"). It's useful for reliable grading/demo purposes, but it
  can't demonstrate genuine language generation or vary its phrasing the
  way a real model call would.
- The RAG retrieval step uses TF-IDF (keyword frequency) similarity, not
  semantic embeddings. This means it matches songs based on shared words in
  their notes, not on deeper conceptual similarity — two songs that "feel"
  similar but are described with completely different vocabulary might not
  be retrieved as similar to each other, even though a human (or an
  embedding model) would group them together.

### 10.4 Concrete Improvement Idea

The most impactful next step would be to **replace the TF-IDF retrieval
with embedding-based retrieval** (e.g. a sentence-embedding model instead
of `TfidfVectorizer`). This would let the system find genuinely similar
songs even when their notes use different words to describe similar
qualities (e.g. a note that says "laid-back" and one that says "relaxed"
would currently look less similar under TF-IDF than they should). A second
worthwhile improvement, closer to the guardrail itself, would be
**replacing the hardcoded energy-descriptor word list with a second,
narrowly-scoped LLM check** ("does this specific sentence make any claim
about energy level that contradicts the number 0.XX?") — that would catch
mismatched claims using words never anticipated in the current fixed list,
at the cost of an extra model call per explanation.

### 10.5 AI Collaboration Reflection (Project 4 Build)

**A helpful AI suggestion:** while writing `tests/test_ai_pipeline.py`
test-first, one of the required tests asserted that
`SongNoteStore.retrieve_context()` should contain a song's title. Writing
that test surfaced a real bug: the actual implementation didn't satisfy
it — note bodies never included the title at all, since the title only
lived in the markdown header line, which the parser deliberately stripped
out. That failing test caught a mismatch that would otherwise have gone
unnoticed, and led directly to fixing `retrieve_context()` (not
`get_note()` or `retrieve_similar()`, which stayed unchanged) so each
retrieved block is prefixed with its song's title and artist.

**A specific flawed AI output:** the first version of `_offline_stand_in()`
in `src/llm_agent.py` took only the retrieved `context` string, stripped
its markdown headers, and returned the first sentence — it never
explicitly included the song's title or artist anywhere in the returned
text. Because the title lived only in that stripped header, the guardrail's
existence check failed on the offline fallback path even though the
sentence's facts were accurate — a smoke test showed `source: fallback`
with a failed `existence` check on what was supposed to be a normal,
passing case. This was caught by running the guardrail smoke test and
noticing the mismatch between the expected and actual `source`, and fixed
by changing `_offline_stand_in`'s signature to `(song, context)` and
explicitly prefixing the output with `"{song['title']} by {song['artist']}:
"` before the retrieved sentence.

### 10.6 System Limitations

- **No persistence** — nothing about a recommendation, an explanation, or
  a guardrail decision is saved anywhere; every run starts from a clean
  slate reading the same CSV and markdown files.
- **Small dataset** — as noted above, 18 songs is enough to demonstrate the
  pipeline mechanics but far too small to say anything meaningful about
  real-world explanation quality or retrieval accuracy at scale.
- **Offline mode is an approximation, not a replacement** — the
  deterministic offline stand-in exists so the system can run without an
  API key, but it does not replicate the reasoning quality, phrasing
  variety, or failure modes of an actual LLM call; guardrail behavior
  verified only against the offline stand-in may not fully represent how
  the guardrail performs against real, more varied Claude output.