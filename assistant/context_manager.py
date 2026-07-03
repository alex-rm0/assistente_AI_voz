from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum


class ContextType(str, Enum):
    PERSONAL_CONTEXT = "PERSONAL_CONTEXT"
    WORK_CONTEXT = "WORK_CONTEXT"
    TECH_CONTEXT = "TECH_CONTEXT"
    PRODUCTIVITY_CONTEXT = "PRODUCTIVITY_CONTEXT"
    TRAVEL_CONTEXT = "TRAVEL_CONTEXT"
    SOCIAL_CONTEXT = "SOCIAL_CONTEXT"


@dataclass(frozen=True)
class ContextDefinition:
    name: ContextType
    description: str
    memory_category: str


@dataclass(frozen=True)
class ActiveContext:
    name: ContextType
    description: str
    memory_category: str
    weight: float
    reason: str


CONTEXT_DEFINITIONS: dict[ContextType, ContextDefinition] = {
    ContextType.PERSONAL_CONTEXT: ContextDefinition(
        name=ContextType.PERSONAL_CONTEXT,
        description="Dados pessoais, gostos, preferencias, habitos e historia do utilizador.",
        memory_category="perfil_utilizador",
    ),
    ContextType.WORK_CONTEXT: ContextDefinition(
        name=ContextType.WORK_CONTEXT,
        description="Projetos, trabalho atual, objetivos profissionais e tarefas ligadas a projetos.",
        memory_category="projetos",
    ),
    ContextType.TECH_CONTEXT: ContextDefinition(
        name=ContextType.TECH_CONTEXT,
        description="Programacao, erros, Python, PySide6, Ollama, Git, testes e arquitetura tecnica.",
        memory_category="projetos",
    ),
    ContextType.PRODUCTIVITY_CONTEXT: ContextDefinition(
        name=ContextType.PRODUCTIVITY_CONTEXT,
        description="Tarefas, lembretes, organizacao pessoal, prioridades e planeamento.",
        memory_category="tarefas",
    ),
    ContextType.TRAVEL_CONTEXT: ContextDefinition(
        name=ContextType.TRAVEL_CONTEXT,
        description="Viagens, ferias, destinos, itinerarios, alojamento e preparacao de deslocacoes.",
        memory_category="preferencias",
    ),
    ContextType.SOCIAL_CONTEXT: ContextDefinition(
        name=ContextType.SOCIAL_CONTEXT,
        description="Pessoas importantes, familia, amigos, relacoes e compromissos sociais.",
        memory_category="relacoes",
    ),
}


class ContextManager:
    """Selects relevant contexts automatically from the user's message."""

    def identify(self, user_message: str) -> list[ActiveContext]:
        text = _normalize_text(user_message)
        scores: dict[ContextType, tuple[float, list[str]]] = {}

        for context_type, weighted_keywords in _KEYWORDS.items():
            total = 0.0
            reasons: list[str] = []
            for keyword, weight in weighted_keywords:
                if keyword in text:
                    total += weight
                    reasons.append(f"detetado '{keyword}'")
            if total > 0:
                scores[context_type] = (min(total, 1.0), reasons)

        # A personal layer is useful for broad plans that depend on preferences.
        if ContextType.TRAVEL_CONTEXT in scores or ContextType.SOCIAL_CONTEXT in scores:
            _add_score(scores, ContextType.PERSONAL_CONTEXT, 0.35, "contexto pessoal relevante")

        if ContextType.TECH_CONTEXT in scores:
            _add_score(scores, ContextType.WORK_CONTEXT, 0.45, "pedido tecnico ligado a trabalho/projeto")

        if not scores:
            _add_score(scores, ContextType.PERSONAL_CONTEXT, 0.2, "contexto geral de conversa")

        active: list[ActiveContext] = []
        for context_type, (weight, reasons) in scores.items():
            definition = CONTEXT_DEFINITIONS[context_type]
            active.append(
                ActiveContext(
                    name=definition.name,
                    description=definition.description,
                    memory_category=definition.memory_category,
                    weight=round(weight, 2),
                    reason=", ".join(reasons[:3]),
                )
            )

        return sorted(active, key=lambda item: item.weight, reverse=True)

    def system_prompt(self, active_contexts: list[ActiveContext]) -> str:
        context_lines = "\n".join(
            f"- {item.name.value} ({item.weight:.2f}): {item.description}"
            for item in active_contexts
        )
        return (
            "Es o AssistenteIA, um assistente local persistente para Windows 11.\n"
            "Usa sempre portugues de Portugal. Trata o utilizador por tu, de forma informal e consistente. "
            "Evita portugues do Brasil. Usa 'aplicacoes', nunca 'aplicativos'; 'ecra', nunca 'tela'; "
            "'estou a acompanhar' ou 'estou a observar', nunca 'estou assistindo'; "
            "'ficheiros', nunca 'arquivos'; 'aceder', nunca 'acessar'; 'utilizador', nunca 'usuario'.\n"
            "Mantem-te seguro: nao executes comandos do sistema, nao acedas a ficheiros fora da pasta workspace "
            "e nao finjas ter capacidades que ainda nao existem nesta versao. "
            "Nunca inventes informacoes sobre janelas, aplicacoes, programas ou atividade do computador; "
            "se a informacao nao vier do Context Observer, diz que nao tens acesso a essa informacao.\n\n"
            "Contextos ativos identificados automaticamente:\n"
            f"{context_lines or '- PERSONAL_CONTEXT (0.20): conversa geral'}\n\n"
            "Adapta a resposta aos contextos ativos sem dizer que foram escolhidos, a menos que o utilizador pergunte."
        )

    def debug_summary(self, active_contexts: list[ActiveContext]) -> str:
        lines = ["Contextos ativos:"]
        for item in active_contexts:
            lines.append(f"- {item.name.value}: peso {item.weight:.2f}; razao: {item.reason}")
        return "\n".join(lines)


_KEYWORDS: dict[ContextType, tuple[tuple[str, float], ...]] = {
    ContextType.PERSONAL_CONTEXT: (
        ("chamo-me", 0.7),
        ("chamo me", 0.7),
        ("meu nome", 0.7),
        ("prefiro", 0.6),
        ("gosto", 0.5),
        ("habito", 0.4),
        ("familia", 0.4),
    ),
    ContextType.WORK_CONTEXT: (
        ("projeto", 0.5),
        ("projecto", 0.5),
        ("trabalho", 0.5),
        ("relatorio", 0.4),
        ("rvcc", 0.5),
        ("portfolio", 0.4),
        ("workspace", 0.4),
    ),
    ContextType.TECH_CONTEXT: (
        ("python", 0.7),
        ("pyside6", 0.7),
        ("ollama", 0.6),
        ("erro", 0.6),
        ("codigo", 0.7),
        ("testes", 0.6),
        ("git", 0.5),
        ("agent", 0.5),
        ("arquitetura", 0.5),
        ("implementa", 0.5),
        ("corrige", 0.5),
    ),
    ContextType.PRODUCTIVITY_CONTEXT: (
        ("lembra-me", 0.8),
        ("lembra me", 0.8),
        ("tarefa", 0.7),
        ("tarefas", 0.7),
        ("agenda", 0.6),
        ("hoje", 0.4),
        ("amanha", 0.5),
        ("organiza", 0.5),
        ("prioridade", 0.5),
    ),
    ContextType.TRAVEL_CONTEXT: (
        ("ferias", 0.8),
        ("viagem", 0.8),
        ("viajar", 0.8),
        ("australia", 0.8),
        ("hotel", 0.5),
        ("voo", 0.5),
        ("itinerario", 0.6),
        ("destino", 0.5),
    ),
    ContextType.SOCIAL_CONTEXT: (
        ("amigo", 0.5),
        ("amiga", 0.5),
        ("familia", 0.6),
        ("mae", 0.6),
        ("pai", 0.6),
        ("filho", 0.6),
        ("filha", 0.6),
        ("reuniao", 0.4),
        ("aniversario", 0.6),
    ),
}


def _add_score(
    scores: dict[ContextType, tuple[float, list[str]]],
    context_type: ContextType,
    weight: float,
    reason: str,
) -> None:
    current_weight, reasons = scores.get(context_type, (0.0, []))
    scores[context_type] = (min(current_weight + weight, 1.0), [*reasons, reason])


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
