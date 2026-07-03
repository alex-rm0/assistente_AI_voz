from __future__ import annotations

import gc
from datetime import date
from pathlib import Path

from assistant.long_term_memory import LongTermMemory, MemoryCategory, classify_memory


class FakeEmbedder:
    def embed(self, text: str) -> list[float] | None:
        normalized = text.lower()
        if "python" in normalized:
            return [1.0, 0.0, 0.0]
        if "rvcc" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.0, 0.0, 1.0]


def test_memory_classifies_human_like_categories() -> None:
    assert classify_memory("O meu nome e Alexandre") == MemoryCategory.USER_PROFILE.value
    assert classify_memory("Gosto de Python") == MemoryCategory.PREFERENCES.value
    assert classify_memory("O projeto AssistenteIA e recorrente") == MemoryCategory.PROJECTS.value
    assert classify_memory("Tenho de rever o README") == MemoryCategory.TASKS.value
    assert classify_memory("A minha mae e importante") == MemoryCategory.RELATIONSHIPS.value


def test_memory_persists_in_sqlite(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    memory.remember("Gosto de Python")
    reopened = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    response = reopened.answer_about("Python")

    assert "Gosto de Python" in response
    assert MemoryCategory.PREFERENCES.value in response
    del memory
    del reopened
    gc.collect()


def test_semantic_search_uses_embeddings(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.remember("Gosto de Python")
    memory.remember("Estou a trabalhar num portefolio RVCC")

    matches = memory.search("codigo Python", limit=1)

    assert len(matches) == 1
    assert matches[0].content == "Gosto de Python"
    del memory
    gc.collect()


def test_forget_removes_best_matching_memory(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())
    memory.remember("Gosto de Python")

    result = memory.forget("Python")

    assert "Esqueci" in result
    assert not memory.search("Python")
    del memory
    gc.collect()


def test_context_summary_is_saved_to_memory_and_timeline(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    result = memory.remember_context_summary(
        "Entre as 09:00 e as 11:00 o Alexandre trabalhou no projeto AssistenteIA usando VSCode e Git.",
        event_date=date(2026, 6, 7),
        project="AssistenteIA",
    )

    assert "Guardei o resumo de contexto" in result
    assert "AssistenteIA" in memory.answer_about("AssistenteIA")
    assert "VSCode" in memory.timeline_for_date(date(2026, 6, 7))
    del memory
    gc.collect()


def test_language_preferences_are_persisted(tmp_path: Path) -> None:
    memory = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    memory.set_preference("idioma_base", "pt-PT")
    memory.set_preference("idioma_atual", "en")

    reopened = LongTermMemory(tmp_path, embedder=FakeEmbedder())

    assert reopened.get_preference("idioma_base") == "pt-PT"
    assert reopened.get_preference("idioma_atual") == "en"
    assert reopened.get_preference("inexistente", "pt-PT") == "pt-PT"
    del memory
    del reopened
    gc.collect()
