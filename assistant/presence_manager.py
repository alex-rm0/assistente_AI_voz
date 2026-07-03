from __future__ import annotations

import unicodedata
from enum import Enum


class PresenceState(str, Enum):
    ACTIVE_CONVERSATION = "ACTIVE_CONVERSATION"
    PASSIVE_MONITORING = "PASSIVE_MONITORING"
    FOCUS_MODE = "FOCUS_MODE"
    PRIVATE_MODE = "PRIVATE_MODE"
    OFFLINE = "OFFLINE"


class PresenceManager:
    """Stores the global presence state and exposes behaviour flags."""

    def __init__(self, initial_state: str | PresenceState = PresenceState.ACTIVE_CONVERSATION) -> None:
        self._state = self._coerce_state(initial_state)

    @property
    def state(self) -> PresenceState:
        return self._state

    def set_state(self, state: str | PresenceState) -> PresenceState:
        self._state = self._coerce_state(state)
        return self._state

    def can_wake_from_state(self, message: str) -> bool:
        text = _normalize_text(message)
        exact_commands = {
            "volta",
            "acorda",
            "reativa-te",
            "reativa te",
            "ativa-te",
            "ativa te",
        }
        if text.strip(" .,!?:;") in exact_commands:
            return True

        return any(
            phrase in text
            for phrase in (
                "volta ao modo conversa",
                "modo conversa",
                "podes voltar a falar comigo",
            )
        )

    def requested_state_from_message(self, message: str) -> PresenceState | None:
        text = _normalize_text(message)
        if self.can_wake_from_state(message):
            return PresenceState.ACTIVE_CONVERSATION

        if any(
            phrase in text
            for phrase in (
                "vou trabalhar calado",
                "fica so a acompanhar",
                "nao fales comigo agora",
            )
        ):
            return PresenceState.PASSIVE_MONITORING

        if any(
            phrase in text
            for phrase in (
                "vou concentrar-me",
                "vou concentrar me",
                "nao me interrompas",
                "modo foco",
            )
        ):
            return PresenceState.FOCUS_MODE

        if any(
            phrase in text
            for phrase in (
                "nao guardes isto",
                "modo privado",
                "isto e privado",
            )
        ):
            return PresenceState.PRIVATE_MODE

        if "podes falar comigo" in text:
            return PresenceState.ACTIVE_CONVERSATION

        if any(phrase in text for phrase in ("desliga-te", "desliga te", "fica offline")):
            return PresenceState.OFFLINE

        return None

    def state_report(self) -> str:
        return f"Estou em {self.state.value}. {self.description()}"

    def confirmation_for(self, state: str | PresenceState) -> str:
        confirmed_state = self._coerce_state(state)
        confirmations = {
            PresenceState.ACTIVE_CONVERSATION: "Entendido, volto ao modo conversa.",
            PresenceState.PASSIVE_MONITORING: "Entendido, vou ficar em modo observador.",
            PresenceState.FOCUS_MODE: "Entendido, vou ficar em modo foco.",
            PresenceState.PRIVATE_MODE: "Entendido, vou ficar em modo privado e não vou guardar esta conversa.",
            PresenceState.OFFLINE: "Entendido, vou ficar offline.",
        }
        return confirmations[confirmed_state]

    def can_respond(self) -> bool:
        return self._state in {
            PresenceState.ACTIVE_CONVERSATION,
            PresenceState.PRIVATE_MODE,
        }

    def can_use_tools(self) -> bool:
        return self._state == PresenceState.ACTIVE_CONVERSATION

    def can_make_suggestions(self) -> bool:
        return self._state == PresenceState.ACTIVE_CONVERSATION

    def can_ask_confirmation(self) -> bool:
        return self._state == PresenceState.ACTIVE_CONVERSATION

    def can_store_memory(self) -> bool:
        return self._state not in {
            PresenceState.PRIVATE_MODE,
            PresenceState.OFFLINE,
        }

    def can_observe_activity(self) -> bool:
        return self._state in {
            PresenceState.PASSIVE_MONITORING,
            PresenceState.FOCUS_MODE,
        }

    def can_interrupt(self) -> bool:
        return self._state in {
            PresenceState.ACTIVE_CONVERSATION,
            PresenceState.FOCUS_MODE,
        }

    def description(self) -> str:
        descriptions = {
            PresenceState.ACTIVE_CONVERSATION: "Conversa ativa: posso responder, usar ferramentas e pedir confirmacoes.",
            PresenceState.PASSIVE_MONITORING: "Monitorizacao passiva: acompanho contexto, mas nao respondo nem inicio conversa.",
            PresenceState.FOCUS_MODE: "Modo foco: fico em silencio e so interromperia em situacoes importantes.",
            PresenceState.PRIVATE_MODE: "Modo privado: respondo sem gravar memoria nem observar atividade.",
            PresenceState.OFFLINE: "Offline: tudo esta desligado.",
        }
        return descriptions[self._state]

    @staticmethod
    def names() -> list[str]:
        return [state.value for state in PresenceState]

    @staticmethod
    def _coerce_state(state: str | PresenceState) -> PresenceState:
        if isinstance(state, PresenceState):
            return state
        try:
            return PresenceState(state)
        except ValueError:
            return PresenceState.ACTIVE_CONVERSATION


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char))
