from __future__ import annotations

import json
import math
import sqlite3
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


DEFAULT_MEMORY_DB = "long_term_memory.sqlite"


class Embedder(Protocol):
    def embed(self, text: str) -> list[float] | None:
        ...


@dataclass(frozen=True)
class MemoryRecord:
    id: int
    category: str
    content: str
    score: float = 0.0


class LongTermMemory:
    """Permanent SQLite memory, separate from conversation history."""

    def __init__(self, data_path: Path, db_file: str = DEFAULT_MEMORY_DB, embedder: Embedder | None = None) -> None:
        self.data_path = data_path.resolve()
        self.db_path = (self.data_path / db_file).resolve()
        self.embedder = embedder
        self._ensure_inside_data()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def remember(self, content: str, category: str | None = None) -> str:
        text = content.strip()
        if not text:
            return "Diz-me o que queres que eu memorize."

        memory_category = category or classify_memory(text)
        embedding = self._embed_json(text)

        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                INSERT INTO memories (category, content, embedding)
                VALUES (?, ?, ?)
                """,
                (memory_category, text, embedding),
            )

        return f"Memorizei isto como {memory_category}: {text}"

    def forget(self, query: str) -> str:
        text = query.strip()
        if not text:
            return "Diz-me o que queres que eu esqueca."

        matches = self.search(text, limit=5)
        if not matches:
            return f"Nao encontrei memorias sobre '{text}'."

        best = matches[0]
        with sqlite3.connect(self.db_path) as connection:
            connection.execute("DELETE FROM memories WHERE id = ?", (best.id,))

        return f"Esqueci esta memoria: {best.content}"

    def search(self, query: str, limit: int = 5) -> list[MemoryRecord]:
        text = query.strip()
        if not text:
            return []

        # Use Ollama embeddings when possible; otherwise fall back to a small
        # text search so memory still works offline or with unsupported models.
        records = self._all_records()
        query_embedding = self._embed(text)
        if query_embedding:
            scored = [
                MemoryRecord(record.id, record.category, record.content, _cosine_similarity(query_embedding, embedding))
                for record, embedding in records
                if embedding
            ]
            scored = [record for record in scored if record.score > 0]
            scored.sort(key=lambda record: record.score, reverse=True)
            if scored:
                return scored[:limit]

        return self._text_search(text, limit)

    def answer_about(self, query: str) -> str:
        matches = self.search(query, limit=5)
        if not matches:
            return f"Nao tenho memoria permanente sobre '{query}'."

        lines = [f"- [{record.category}] {record.content}" for record in matches]
        return "Sei isto na memoria permanente:\n" + "\n".join(lines)

    def context_for(self, query: str, limit: int = 5) -> str:
        matches = self.search(query, limit=limit)
        if not matches:
            return ""

        lines = [f"- [{record.category}] {record.content}" for record in matches]
        return "\n".join(lines)

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

    def _all_records(self) -> list[tuple[MemoryRecord, list[float] | None]]:
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                "SELECT id, category, content, embedding FROM memories ORDER BY created_at DESC"
            ).fetchall()

        records: list[tuple[MemoryRecord, list[float] | None]] = []
        for row in rows:
            embedding = _loads_embedding(row[3])
            records.append((MemoryRecord(id=row[0], category=row[1], content=row[2]), embedding))
        return records

    def _text_search(self, query: str, limit: int) -> list[MemoryRecord]:
        terms = _terms(query)
        if not terms:
            return []

        scored: list[MemoryRecord] = []
        for record, _embedding in self._all_records():
            content = record.content.lower()
            category = record.category.lower()
            score = sum(1 for term in terms if term in content or term in category)
            if score > 0:
                scored.append(MemoryRecord(record.id, record.category, record.content, float(score)))

        scored.sort(key=lambda record: record.score, reverse=True)
        return scored[:limit]

    def _embed(self, text: str) -> list[float] | None:
        if self.embedder is None:
            return None
        return self.embedder.embed(text)

    def _embed_json(self, text: str) -> str | None:
        embedding = self._embed(text)
        if not embedding:
            return None
        return json.dumps(embedding, ensure_ascii=True)

    def _ensure_inside_data(self) -> None:
        if self.db_path != self.data_path and self.data_path not in self.db_path.parents:
            raise ValueError("Long-term memory database must stay inside the data folder.")


def classify_memory(content: str) -> str:
    text = _normalize_text(content)
    if any(word in text for word in ("prefiro", "gosto", "nao gosto", "preferencia")):
        return "preferencia"
    if any(word in text for word in ("projeto", "projecto", "assistenteia", "app", "aplicacao")):
        return "projeto"
    return "contexto"


def _loads_embedding(value: str | None) -> list[float] | None:
    if not value:
        return None
    try:
        data = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, list):
        return None
    return [float(item) for item in data if isinstance(item, (int, float))]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _terms(text: str) -> list[str]:
    normalized = _normalize_text(text)
    return [term.strip(".,;:!?()[]{}\"'") for term in normalized.split() if len(term.strip()) >= 3]


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
