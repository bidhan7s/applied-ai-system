"""
RAG (retrieval-augmented generation) layer for Project 4.

Parses data/song_notes.md into per-song note text, builds a TF-IDF index
over the notes, and retrieves similar notes by cosine similarity so a
downstream LLM prompt can be grounded with relevant context.
"""

import re
from typing import Dict, List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

HEADER_RE = re.compile(r"^##\s*(\d+)\s*\|\s*(.+)$", re.MULTILINE)


class SongNoteStore:
    """
    Loads song notes from a markdown file and serves TF-IDF based retrieval
    over them.
    """

    def __init__(self, notes_path: str):
        self.notes: Dict[int, str] = {}
        self.headers: Dict[int, str] = {}
        self._parse_notes(notes_path)
        self.song_ids: List[int] = list(self.notes.keys())
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(
            [self.notes[song_id] for song_id in self.song_ids]
        )

    def _parse_notes(self, notes_path: str) -> None:
        with open(notes_path, encoding="utf-8") as f:
            content = f.read()

        matches = list(HEADER_RE.finditer(content))
        for i, match in enumerate(matches):
            song_id = int(match.group(1))
            self.headers[song_id] = match.group(2).strip()
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(content)
            self.notes[song_id] = content[start:end].strip()

    def get_note(self, song_id: int) -> str:
        """Returns the note text for a single song id."""
        return self.notes[song_id]

    def _similar_ids(self, song_id: int, k: int) -> List[int]:
        idx = self.song_ids.index(song_id)
        similarities = cosine_similarity(
            self.tfidf_matrix[idx], self.tfidf_matrix
        )[0]

        ranked = sorted(
            (i for i in range(len(self.song_ids)) if i != idx),
            key=lambda i: similarities[i],
            reverse=True,
        )
        return [self.song_ids[i] for i in ranked[:k]]

    def retrieve_similar(self, song_id: int, k: int = 2) -> List[str]:
        """Returns the k most similar OTHER notes by TF-IDF cosine similarity."""
        return [self.notes[sid] for sid in self._similar_ids(song_id, k)]

    def retrieve_context(self, song_id: int, k: int = 2) -> str:
        """Returns the song's own note plus its k most similar notes, joined.

        Each block is prefixed with a "## <title> — <artist>" header so the
        song's own title is always present in the returned context.
        """
        ids = [song_id] + self._similar_ids(song_id, k)
        blocks = [f"## {self.headers[sid]}\n{self.notes[sid]}" for sid in ids]
        return "\n\n".join(blocks)
