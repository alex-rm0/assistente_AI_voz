from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from enum import Enum


class DelegationTarget(str, Enum):
    LOCAL = "local"
    CHATGPT = "chatgpt"
    CODEX = "codex"
    EXTERNAL_TOOL = "external_tool"


@dataclass(frozen=True)
class DelegationDecision:
    target: DelegationTarget
    reason: str
    prepared_prompt: str = ""


class DelegationManager:
    """Decides whether a request should be handled locally or delegated."""

    def decide(
        self,
        user_message: str,
        profile_name: str = "AUTO_CONTEXT",
        context: str = "",
    ) -> DelegationDecision:
        text = _normalize_text(user_message)

        if _explicitly_requests_codex(text) or _looks_like_codebase_work(text):
            return DelegationDecision(
                target=DelegationTarget.CODEX,
                reason="O pedido envolve trabalho de codigo/projeto que e melhor tratado pelo Codex.",
                prepared_prompt=_build_prompt("Codex", user_message, profile_name, context),
            )

        if _explicitly_requests_chatgpt(text) or _looks_like_broad_reasoning(text):
            return DelegationDecision(
                target=DelegationTarget.CHATGPT,
                reason="O pedido pede raciocinio, escrita ou exploracao ampla; o ChatGPT e uma boa opcao.",
                prepared_prompt=_build_prompt("ChatGPT", user_message, profile_name, context),
            )

        if _looks_like_external_tool_request(text):
            return DelegationDecision(
                target=DelegationTarget.EXTERNAL_TOOL,
                reason="O pedido depende de uma ferramenta externa especifica.",
                prepared_prompt=_build_prompt("ferramenta externa", user_message, profile_name, context),
            )

        return DelegationDecision(
            target=DelegationTarget.LOCAL,
            reason="O pedido pode ser resolvido localmente pelo AssistenteIA.",
        )

    def format_response(self, decision: DelegationDecision) -> str:
        if decision.target == DelegationTarget.LOCAL:
            return ""

        labels = {
            DelegationTarget.CHATGPT: "ChatGPT",
            DelegationTarget.CODEX: "Codex",
            DelegationTarget.EXTERNAL_TOOL: "uma ferramenta externa",
        }
        label = labels[decision.target]
        return (
            f"Este pedido e melhor resolvido pelo {label}.\n"
            f"Estrategia: {decision.reason}\n\n"
            "Vou preparar o contexto:\n\n"
            f"```text\n{decision.prepared_prompt}\n```"
        )


def _build_prompt(target: str, user_message: str, profile_name: str, context: str) -> str:
    parts = [
        f"Destino sugerido: {target}",
        f"Contextos ativos no AssistenteIA: {profile_name}",
        "",
        "Pedido original:",
        user_message.strip(),
    ]
    if context.strip():
        parts.extend(("", "Contexto relevante do AssistenteIA:", context.strip()))
    parts.extend(
        (
            "",
            "Objetivo:",
            "Ajuda a resolver este pedido com clareza, em portugues de Portugal, mantendo seguranca e privacidade.",
        )
    )
    return "\n".join(parts)


def _explicitly_requests_codex(text: str) -> bool:
    return any(phrase in text for phrase in ("manda para o codex", "prepara para o codex", "usar codex"))


def _explicitly_requests_chatgpt(text: str) -> bool:
    return any(phrase in text for phrase in ("manda para o chatgpt", "prepara para o chatgpt", "usar chatgpt"))


def _looks_like_codebase_work(text: str) -> bool:
    code_words = (
        "implementa",
        "corrige",
        "refatora",
        "cria testes",
        "faz testes",
        "erro no codigo",
        "altera o projeto",
        "atualiza o projeto",
        "commit",
        "git",
        "pull request",
    )
    return any(word in text for word in code_words)


def _looks_like_broad_reasoning(text: str) -> bool:
    broad_words = (
        "escreve um ensaio",
        "faz uma estrategia",
        "brainstorm",
        "ideias para",
        "argumenta",
        "explica profundamente",
        "plano de negocio",
    )
    return any(word in text for word in broad_words)


def _looks_like_external_tool_request(text: str) -> bool:
    external_words = (
        "abre o browser",
        "abre no browser",
        "abre o excel",
        "abre o word",
        "abre ferramenta",
        "usa uma ferramenta externa",
    )
    return any(word in text for word in external_words)


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
