"""KNOWN_ISSUES §5: the "Resumo de sessao" memories carry the labels
"projeto=", "tarefas=", "ferramentas=" in their text, so the generic text
search matched them for almost every query ("projeto", "tarefa",
"assistenteIA" returned only session summaries). Session continuity reads
session_manager, not long-term memory, so generic search and context_for
must skip them; normal memories with the same words must still be found.
"""
from __future__ import annotations

import gc
from datetime import date
from pathlib import Path

import pytest

from assistant.long_term_memory import LongTermMemory

QUERIES = ("projeto", "tarefa", "assistenteIA")
SESSION_SUMMARY = (
    "Resumo de sessao: projeto=assistenteIA; atividade=atividade de trabalho local; "
    "factos=sessao iniciada em 2026-10-29T09:00:00. sessao terminada em 2026-10-29T09:05:00; "
    "ficheiros=sem ficheiros registados; tarefas=Cria uma tarefa para rever o Planner; "
    "ferramentas=sem ferramentas registadas; decisoes=sem decisoes registadas; "
    "proximo_passo=sem inferencia."
)
NORMAL = "O projeto assistenteIA tem uma tarefa pendente: rever o Planner."


class _Embedder:
    def embed(self, text: str) -> list[float] | None:
        return [1.0, 0.0]


@pytest.fixture(params=["text_search", "embeddings"])
def memory(tmp_path: Path, request):
    if request.param == "embeddings":
        memory = LongTermMemory(tmp_path, embedder=_Embedder(), embeddings_enabled=True)
    else:
        memory = LongTermMemory(tmp_path, embedder=None)
    memory.remember(NORMAL)
    memory.remember_context_summary(SESSION_SUMMARY, event_date=date.today(), project="assistenteIA")
    yield memory
    del memory
    gc.collect()


@pytest.mark.parametrize("query", QUERIES)
def test_search_skips_session_summaries(memory: LongTermMemory, query: str) -> None:
    contents = [record.content for record in memory.search(query)]

    assert NORMAL in contents
    assert not [content for content in contents if content.startswith("Resumo de sessao")], contents


@pytest.mark.parametrize("query", QUERIES)
def test_context_for_skips_session_summaries(memory: LongTermMemory, query: str) -> None:
    context = memory.context_for(query)

    assert NORMAL in context
    assert "Resumo de sessao" not in context


def test_session_summary_stays_in_the_timeline(memory: LongTermMemory) -> None:
    assert "Resumo de sessao" in memory.timeline_for_date(date.today())
