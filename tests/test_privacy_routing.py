"""North Star §6: privacy classes are checked at the model router boundary.

PUBLIC/PERSONAL may go to the cloud; PRIVATE only when the turn itself is an
explicit request about that document; RESTRICTED never. A request with no
class is PRIVATE. All tests use fake providers: no network.
"""
from __future__ import annotations

import pytest

from assistant.model_provider import ModelResponse, ProviderConfigurationError, ProviderTimeoutError
from assistant.model_router import (
    AutomaticRoutingConfig,
    ModelRouter,
    ModelRoutingConfig,
    ModelRoutingInput,
    RoutedLLM,
)
from assistant.privacy import (
    PERSONAL,
    PRIVATE,
    PUBLIC,
    RESTRICTED,
    content_privacy_class,
    normalize_privacy_class,
    provider_allowed,
    stricter,
)
from assistant.response_composer import ComposerRequest, ResponseComposer


class FakeProvider:
    def __init__(self, name: str, *, error: Exception | None = None) -> None:
        self._name = name
        self.model = f"{name}-model"
        self.error = error
        self.calls: list[dict] = []

    @property
    def name(self) -> str:
        return self._name

    def chat(self, messages, *, model=None, timeout_seconds=None, **kwargs) -> ModelResponse:
        self.calls.append({"messages": messages, "model": model, "timeout_seconds": timeout_seconds})
        if self.error is not None:
            raise self.error
        return ModelResponse(text=f"reply from {self._name}", provider=self._name, model=model or self.model,
                             input_tokens=10, output_tokens=5, latency_ms=1.0, estimated_cost_usd=0.001 if self._name == "anthropic" else 0.0)


def make_router(mode: str = "automatic", *, paid: bool = True, key: bool = True, claude_enabled: bool = True, budget=None) -> ModelRouter:
    return ModelRouter(
        ModelRoutingConfig(mode=mode, mode_source="test", automatic=AutomaticRoutingConfig(claude_enabled=claude_enabled)),
        ollama_model="ollama-model",
        anthropic_model="anthropic-model",
        budget=budget,
        env={},
        anthropic_key_available=lambda: key,
        paid_calls_allowed=lambda: paid,
    )


def make_llm(mode: str = "automatic", *, claude_error: Exception | None = None, **router_kwargs):
    ollama, claude = FakeProvider("ollama"), FakeProvider("anthropic", error=claude_error)
    llm = RoutedLLM(providers={"ollama": ollama, "anthropic": claude}, router=make_router(mode, **router_kwargs), system_prompt="system")
    return llm, ollama, claude


def conversation(privacy_class: str | None, *, explicit: bool = False, source: str = "RESPONSE_COMPOSER") -> ModelRoutingInput:
    kwargs = {} if privacy_class is None else {"privacy_class": privacy_class}
    return ModelRoutingInput(source=source, user_message="Hoje estou cansado.", explicit_document_request=explicit, **kwargs)


# --- the rule itself -----------------------------------------------------------


@pytest.mark.parametrize("privacy_class,explicit,expected", [
    (PUBLIC, False, True),
    (PERSONAL, False, True),
    (PRIVATE, False, False),
    (PRIVATE, True, True),
    (RESTRICTED, False, False),
    (RESTRICTED, True, False),
    ("", False, False),
    ("SOMETHING_ELSE", False, False),
    (None, True, True),  # no class is PRIVATE, and PRIVATE with the explicit mark is allowed
])
def test_provider_allowed_for_the_cloud(privacy_class, explicit, expected) -> None:
    assert provider_allowed("anthropic", privacy_class, explicit_document_request=explicit) is expected


@pytest.mark.parametrize("privacy_class", [PUBLIC, PERSONAL, PRIVATE, RESTRICTED, "", None])
def test_local_provider_is_always_allowed(privacy_class) -> None:
    assert provider_allowed("ollama", privacy_class) is True


def test_no_class_is_private_and_the_stricter_class_wins() -> None:
    assert normalize_privacy_class(None) == PRIVATE
    assert normalize_privacy_class("personal") == PERSONAL
    assert ModelRoutingInput(source="X", user_message="x").privacy_class == PRIVATE
    assert stricter(PERSONAL, RESTRICTED) == RESTRICTED
    assert stricter(PRIVATE, PUBLIC) == PRIVATE


@pytest.mark.parametrize("text,expected", [
    ("Ficheiro lido da workspace: notas.pdf\n\nConteúdo extraído:\n...", PRIVATE),
    ("Documento ativo: relatorio.docx.", PRIVATE),
    ("Ação documental: review", PRIVATE),
    # A dictated email draft is Alexandre's own request, not mail read from a mailbox:
    # nothing in the text makes it RESTRICTED. That class comes from the source (F5).
    ("Assunto: Reunião de amanhã\n\nOlá Francisca,", None),
    ("Hoje estou cansado.", None),
    ("Qual é o assunto da reunião?", None),
])
def test_content_markers_raise_the_class(text, expected) -> None:
    assert content_privacy_class([text]) == expected


# --- at the router boundary, in every mode -------------------------------------


@pytest.mark.parametrize("privacy_class,explicit,provider,reason", [
    (PUBLIC, False, "anthropic", "general_conversation"),
    (PERSONAL, False, "anthropic", "general_conversation"),
    (PRIVATE, False, "ollama", "privacy_class_private"),
    (PRIVATE, True, "anthropic", "general_conversation"),
    (RESTRICTED, False, "ollama", "privacy_class_restricted"),
    (RESTRICTED, True, "ollama", "privacy_class_restricted"),
    (None, False, "ollama", "privacy_class_private"),
])
def test_automatic_mode_checks_the_class(privacy_class, explicit, provider, reason) -> None:
    decision = make_router().decide(conversation(privacy_class, explicit=explicit))

    assert (decision.provider, decision.reason_code) == (provider, reason)
    assert decision.paid_call is (provider == "anthropic")
    assert decision.privacy_class == (privacy_class or PRIVATE)


@pytest.mark.parametrize("privacy_class,provider", [(PERSONAL, "anthropic"), (PRIVATE, "ollama"), (RESTRICTED, "ollama"), (None, "ollama")])
def test_claude_mode_checks_the_class_too(privacy_class, provider) -> None:
    decision = make_router("claude").decide(conversation(privacy_class))

    assert decision.provider == provider
    if provider == "ollama":
        assert decision.reason_code.startswith("privacy_class_") and decision.paid_call is False


def test_local_mode_never_needs_a_class() -> None:
    assert make_router("local").decide(conversation(None)).provider == "ollama"


# --- RoutedLLM: the class travels with the call ---------------------------------


def test_unmarked_call_stays_local_and_marked_personal_call_goes_to_claude() -> None:
    llm, ollama, claude = make_llm()

    llm.chat("Hoje estou cansado.", source="RESPONSE_COMPOSER")
    assert (len(ollama.calls), len(claude.calls)) == (1, 0)
    assert llm.last_routing_decision.reason_code == "privacy_class_private"

    llm.mark_next_call_privacy(PERSONAL)
    llm.chat("Hoje estou cansado.", source="RESPONSE_COMPOSER")
    assert len(claude.calls) == 1

    # The mark is for one call only.
    llm.chat("Hoje estou cansado.", source="RESPONSE_COMPOSER")
    assert (len(ollama.calls), len(claude.calls)) == (2, 1)


def test_document_content_in_the_payload_overrides_a_personal_mark() -> None:
    llm, ollama, claude = make_llm()

    llm.mark_next_call_privacy(PERSONAL)
    llm.chat("Mensagem do Alexandre:\nResume\n\nIntenção:\nx\n\nContexto relevante:\nFicheiro lido da workspace: notas.pdf", source="RESPONSE_COMPOSER")
    assert len(claude.calls) == 0 and llm.last_routing_decision.reason_code == "privacy_class_private"

    llm.mark_next_call_privacy(PERSONAL)
    llm.chat("O que achas?", history=[{"role": "assistant", "content": "Documento ativo: relatorio.docx."}], source="RESPONSE_COMPOSER")
    assert len(claude.calls) == 0 and len(ollama.calls) == 2

    # With the explicit mark (the user asked about that document) the same content may go out.
    llm.mark_next_call_privacy(PRIVATE, True)
    llm.chat("Revê isto.\n\nFicheiro lido da workspace: notas.pdf", source="RESPONSE_COMPOSER")
    assert len(claude.calls) == 1


def test_restricted_data_never_reaches_claude_whatever_else_is_true() -> None:
    """What mail read from a mailbox will be marked as (F5): never sent out,
    in any mode, even with the explicit-document mark."""
    for mode in ("automatic", "claude"):
        llm, ollama, claude = make_llm(mode)
        for explicit in (False, True):
            llm.mark_next_call_privacy(RESTRICTED, explicit)
            llm.chat("Resume este email.", source="RESPONSE_COMPOSER")
        assert claude.calls == [] and len(ollama.calls) == 2
        assert llm.last_routing_decision.reason_code == "privacy_class_restricted"


# --- the composer labels its requests -------------------------------------------


class _MarkRecorder:
    def __init__(self) -> None:
        self.marks: list[tuple[str, bool]] = []

    def mark_next_call_privacy(self, privacy_class: str, explicit_document_request: bool = False) -> None:
        self.marks.append((privacy_class, explicit_document_request))

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        return "ok"


@pytest.mark.parametrize("intent,task_profile,expected", [
    ("conversation", None, (PERSONAL, False)),
    ("general_knowledge_query", None, (PERSONAL, False)),
    ("session_reflection", None, (PERSONAL, False)),
    ("document_review", None, (PRIVATE, True)),
    ("document_rewrite", {"task_type": "document_rewrite"}, (PRIVATE, True)),
    ("conversation", {"task_type": "document_refinement"}, (PRIVATE, True)),
    # A dictated draft is a common task Alexandre asked for, as it was before this change.
    ("compose_email_task", None, (PERSONAL, False)),
    ("writing_task", None, (PERSONAL, False)),
])
def test_composer_marks_the_class_of_each_request(intent, task_profile, expected) -> None:
    llm = _MarkRecorder()

    ResponseComposer(llm).compose(ComposerRequest(intent=intent, user_message="Resume isto por favor.", task_profile=task_profile))

    assert llm.marks == [expected]


def test_composer_regeneration_is_marked_personal() -> None:
    llm = _MarkRecorder()

    ResponseComposer(llm).regenerate("Hoje estou cansado.", [], "emotion_inversion")

    assert llm.marks == [(PERSONAL, False)]


def test_composer_through_the_router() -> None:
    llm, ollama, claude = make_llm()
    composer = ResponseComposer(llm)

    composer.compose(ComposerRequest(intent="conversation", user_message="Hoje estou cansado."))
    composer.compose(ComposerRequest(intent="document_review", user_message="Revê o relatório."))
    # Document content that turns up in an ordinary conversation was not asked about: it stays local.
    composer.compose(ComposerRequest(intent="conversation", user_message="E agora?", context="Ficheiro lido: notas.pdf."))

    assert len(claude.calls) == 2  # the conversation and the explicit document request
    assert len(ollama.calls) == 1
    assert [d.reason_code for d in llm.routing_decisions] == ["general_conversation", "general_conversation", "privacy_class_private"]


# kept here so a missing import fails loudly rather than silently skipping the fallback tests
assert ProviderConfigurationError and ProviderTimeoutError
