from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlannerAction:
    tool_name: str
    arguments: dict[str, str]
    label: str
    requires_confirmation: bool = True


@dataclass(frozen=True)
class PlannerResult:
    intent: str
    related_project: str = ""
    relevant_context: str = ""
    suggested_plan: list[str] = field(default_factory=list)
    recommended_actions: list[PlannerAction] = field(default_factory=list)
    needs_confirmation: bool = False
    delegate_to: str = ""
    direct_response: str = ""

    @property
    def is_normal_conversation(self) -> bool:
        return self.intent == "conversa normal"

    def debug_text(self) -> str:
        lines = [
            "Planner:",
            f"- intencao: {self.intent}",
            f"- projeto: {self.related_project or '(nenhum)'}",
            f"- contexto relevante: {self.relevant_context or '(sem contexto relevante)'}",
            f"- precisa de confirmacao: {'sim' if self.needs_confirmation else 'nao'}",
            f"- delegar para: {self.delegate_to or '(nao)'}",
        ]
        if self.suggested_plan:
            lines.append("- plano sugerido:")
            lines.extend(f"  {index}. {step}" for index, step in enumerate(self.suggested_plan, start=1))
        if self.recommended_actions:
            lines.append("- acoes recomendadas:")
            lines.extend(
                f"  - {action.tool_name} {action.arguments} ({action.label})"
                for action in self.recommended_actions
            )
        return "\n".join(lines)


def plan_user_request(
    user_message: str,
    computer_context: str = "",
    pending_tasks: str = "",
    relevant_memory: str = "",
    recent_timeline: str = "",
    last_session_summary: str = "",
    tools_available: str = "",
    presence_state: str = "ACTIVE_CONVERSATION",
) -> PlannerResult:
    """Create a non-executing plan from user intent and local context."""

    text = _normalize(user_message)
    project = _detect_project(text, user_message, relevant_memory, recent_timeline, last_session_summary)
    context = _compact_context(computer_context, pending_tasks, relevant_memory, recent_timeline, last_session_summary)

    if _looks_like_workspace_environment_request(text):
        related_project = project or "AssistenteIA"
        return PlannerResult(
            intent="preparar ambiente de trabalho",
            related_project=related_project,
            relevant_context=context,
            suggested_plan=[
                "abrir o projeto conhecido no editor configurado",
                "confirmar se o VS Code ou ferramentas relacionadas ja estao abertas",
                "abrir a pasta do projeto se for util",
                "retomar pelo proximo passo de trabalho",
            ],
            recommended_actions=[
                PlannerAction(
                    tool_name="open_project",
                    arguments={"project_name": related_project},
                    label=f"abrir o projeto {related_project}",
                )
            ],
            needs_confirmation=True,
            direct_response=(
                f"Posso preparar o ambiente do projeto {related_project}. "
                "Antes de abrir qualquer coisa, preciso da tua confirmacao."
            ),
        )

    if _looks_like_resume_project(text):
        related_project = project or "AssistenteIA"
        return PlannerResult(
            intent="retomar projeto",
            related_project=related_project,
            relevant_context=context,
            suggested_plan=[
                "rever onde ficamos na ultima sessao",
                "verificar tarefas pendentes relacionadas",
                "confirmar que ferramentas de trabalho estao prontas",
                "propor o proximo passo concreto",
            ],
            recommended_actions=[],
            needs_confirmation=False,
            direct_response=(
                f"Boa. Podemos retomar o projeto {related_project}. "
                "Eu comecaria por rever onde ficamos, olhar para as tarefas pendentes "
                "e escolher um proximo passo pequeno."
            ),
        )

    if _looks_like_travel_planning(text):
        return PlannerResult(
            intent="planeamento pessoal",
            related_project="",
            relevant_context="viagem/lazer",
            suggested_plan=[
                "perguntar destino ou preferencias",
                "recolher datas, duracao e orcamento",
                "organizar prioridades da viagem",
                "sugerir pesquisa web se for necessario",
            ],
            recommended_actions=[],
            needs_confirmation=False,
            delegate_to="pesquisa web opcional",
            direct_response=(
                "Claro. Para planearmos bem, comecava por tres coisas: destino, datas "
                "e orcamento aproximado. Depois montamos uma estrutura simples para voos, alojamento, "
                "roteiro e margem para imprevistos."
            ),
        )

    if _looks_like_task_management(text):
        return PlannerResult(
            intent="gerir tarefas",
            related_project=project,
            relevant_context=_compact_context(pending_tasks, relevant_memory, recent_timeline),
            suggested_plan=[
                "identificar se o pedido e criar, listar, adiar, concluir ou cancelar tarefa",
                "usar a memoria/tarefas como fonte de verdade",
                "pedir esclarecimento se houver varias tarefas possiveis",
                "confirmar o resultado apos alterar dados reais",
            ],
            recommended_actions=[],
            needs_confirmation=False,
            direct_response="",
        )

    return PlannerResult(
        intent="conversa normal",
        relevant_context=context,
        suggested_plan=["responder diretamente ou usar ferramentas apenas se necessario"],
    )


def _looks_like_resume_project(text: str) -> bool:
    return any(phrase in text for phrase in ("vamos continuar", "continuar o projeto", "retomar", "onde ficamos"))


def _looks_like_travel_planning(text: str) -> bool:
    return any(word in text for word in ("ferias", "férias", "viagem", "viajar")) and any(
        word in text for word in ("planeia", "planear", "planeamento", "organiza", "ajuda")
    )


def _looks_like_workspace_environment_request(text: str) -> bool:
    return any(phrase in text for phrase in ("ambiente de trabalho", "workspace de trabalho", "prepara o ambiente")) and any(
        word in text for word in ("abre", "abrir", "prepara")
    )


def _looks_like_task_management(text: str) -> bool:
    return any(word in text for word in ("tarefa", "tarefas", "lembrete", "lembretes")) or any(
        phrase in text
        for phrase in (
            "o que tenho para fazer",
            "tenho de",
            "lembra-me",
            "marca como concluida",
            "cancela isso",
        )
    )


def _detect_project(
    text: str,
    user_message: str,
    relevant_memory: str,
    recent_timeline: str,
    last_session_summary: str = "",
) -> str:
    if "assistenteia" in text or "assistente ia" in text or "assistente" in text:
        return "AssistenteIA"
    match = re.search(r"\b(?:projeto|projecto)\s+([^.,;:!?]+)", user_message, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    combined = _normalize(f"{relevant_memory}\n{recent_timeline}\n{last_session_summary}")
    if "assistenteia" in combined or "assistente ia" in combined:
        return "AssistenteIA"
    return ""


def _compact_context(*parts: str) -> str:
    clean = [part.strip() for part in parts if part and part.strip()]
    if not clean:
        return ""
    value = "\n".join(clean)
    return value[:1200]


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
