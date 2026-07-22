from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PreferenceAssessment:
    domain: str
    known_preferences: list[str] = field(default_factory=list)
    inferred_preferences: list[str] = field(default_factory=list)
    missing_preferences: list[str] = field(default_factory=list)
    next_question: str = ""
    enough_for_recommendation: bool = False


class UserPreferenceBuilder:
    """Builds preference context before Echo recommends anything.

    This module does not persist memory yet. It extracts and interprets
    preference signals so the cognitive loop can decide whether to ask one
    better question instead of producing a list of generic options.
    """

    def assess(self, user_message: str, recent_context: str = "") -> PreferenceAssessment:
        text = _normalize(f"{recent_context}\n{user_message}")
        domain = _detect_domain(text)
        known = _known_preferences(text, domain)
        inferred = _inferred_preferences(text, domain)
        missing = _missing_preferences(domain, known, inferred)
        enough = _enough_for_recommendation(domain, known, inferred)
        question = "" if enough else _next_question(domain, text, known, inferred, missing)

        return PreferenceAssessment(
            domain=domain,
            known_preferences=known,
            inferred_preferences=inferred,
            missing_preferences=missing,
            next_question=question,
            enough_for_recommendation=enough,
        )


def _detect_domain(text: str) -> str:
    if any(word in text for word in ("ferias", "viagem", "viajar", "road trip", "norte de portugal", "praia")):
        return "travel"
    if any(word in text for word in ("portatil", "computador", "laptop")):
        return "laptop"
    if any(word in text for word in ("casa", "apartamento", "moradia")):
        return "home"
    if any(word in text for word in ("carro", "automovel", "automóvel")):
        return "car"
    if any(word in text for word in ("exame", "estudar", "estudo", "disciplina")):
        return "study"
    return "general"


def _known_preferences(text: str, domain: str) -> list[str]:
    preferences: list[str] = []

    if domain == "travel":
        if "norte de portugal" in text or "norte" in text:
            preferences.append("quer viajar pelo Norte de Portugal")
        if "road trip" in text:
            preferences.append("quer fazer uma road trip")
        if any(word in text for word in ("namorada", "namorado", "companheira", "companheiro")):
            preferences.append("vai viajar acompanhado")
        if any(word in text for word in ("descansar", "descanso", "relaxar")):
            preferences.append("procura descanso")
        if any(word in text for word in ("aventura", "aventurado", "explorar")):
            preferences.append("procura alguma aventura")
        if any(word in text for word in ("natureza", "montanha", "geres", "gerês")):
            preferences.append("valoriza natureza")

    if domain == "laptop":
        if any(word in text for word in ("programar", "codigo", "código")):
            preferences.append("precisa de programar")
        if any(word in text for word in ("jogar", "gaming", "jogos")):
            preferences.append("quer jogar")
        if any(word in text for word in ("faculdade", "estudar", "aulas")):
            preferences.append("vai usar para estudar")

    if domain == "home":
        if any(word in text for word in ("centro", "perto", "localizacao", "localização")):
            preferences.append("valoriza localização")
        if any(word in text for word in ("espaco", "espaço", "quartos")):
            preferences.append("valoriza espaço")

    if domain == "car":
        if any(word in text for word in ("economico", "económico", "consumo")):
            preferences.append("valoriza baixo consumo")
        if any(word in text for word in ("familia", "família", "espaco", "espaço")):
            preferences.append("precisa de espaço")

    if domain == "study":
        if any(word in text for word in ("slides", "apontamentos", "material")):
            preferences.append("tem material de estudo")
        if any(word in text for word in ("uma semana", "semana", "amanha", "amanhã")):
            preferences.append("tem prazo próximo")

    return preferences


def _inferred_preferences(text: str, domain: str) -> list[str]:
    inferred: list[str] = []

    if domain == "travel":
        if "road trip" in text:
            inferred.append("prefere experiências a destinos isolados")
        if any(word in text for word in ("namorada", "namorado", "companheira", "companheiro")):
            inferred.append("a viagem deve funcionar bem para duas pessoas")
        if "norte de portugal" in text and not any(word in text for word in ("descansar", "aventura", "natureza")):
            inferred.append("ainda não está claro que tipo de experiência procura")

    if domain == "study" and "nervoso" in text:
        inferred.append("precisa primeiro de ganhar clareza antes de receber um plano")

    return inferred


def _missing_preferences(domain: str, known: list[str], inferred: list[str]) -> list[str]:
    if domain == "travel":
        missing: list[str] = []
        if not any("descanso" in item or "aventura" in item or "natureza" in item for item in known):
            missing.append("tipo de experiência")
        if not any("acompanhado" in item for item in known):
            missing.append("companhia")
        if not any("road trip" in item for item in known):
            missing.append("ritmo da viagem")
        return missing

    if domain == "laptop":
        return ["uso principal"] if not known else []
    if domain == "home":
        return ["prioridade principal"] if not known else []
    if domain == "car":
        return ["uso principal"] if not known else []
    if domain == "study":
        return ["prazo e materiais"] if len(known) < 2 else []
    return ["preferência principal"]


def _enough_for_recommendation(domain: str, known: list[str], inferred: list[str]) -> bool:
    if domain == "travel":
        return len(known) >= 3 and any("experiência" in item or "natureza" in item or "descanso" in item for item in [*known, *inferred])
    if domain in {"laptop", "home", "car", "study"}:
        return len(known) >= 2
    return False


def _next_question(domain: str, text: str, known: list[str], inferred: list[str], missing: list[str]) -> str:
    if domain == "travel":
        if "road trip" in text:
            return "Gosto dessa ideia. Costumas preferir dormir sempre no mesmo sítio ou ir mudando de alojamento ao longo da viagem?"
        if any(word in text for word in ("namorada", "namorado", "companheira", "companheiro")):
            return "Perfeito. Então faz sentido procurar algo pensado para os dois e sem grandes mudanças de planos todos os dias."
        if "norte de portugal" in text or text.strip() == "norte":
            return "Acho que ainda me falta perceber como gostas de viajar. Isso vai influenciar muito mais a escolha do destino do que simplesmente saber que queres ir para o Norte."
        return "Boa ideia. Antes de começarmos a procurar sítios, deixa-me perceber uma coisa. Quando pensas em férias, procuras mais descansar, conhecer sítios novos ou fazer uma viagem com alguma aventura?"

    if domain == "laptop":
        return "Antes de sugerir modelos, preciso de perceber o uso principal. É mais para estudar, trabalhar, programar ou jogar?"
    if domain == "home":
        return "Antes de pensar em opções, preciso de perceber o que pesa mais para ti: localização, espaço ou tranquilidade?"
    if domain == "car":
        return "Antes de sugerir carros, preciso de perceber o uso principal. É mais cidade, viagens longas ou algo familiar?"
    if domain == "study":
        return "Antes de montar um plano, preciso só de perceber duas coisas. Quando é o exame? E já tens material para estudar?"
    return "Antes de sugerir alguma coisa, preciso de perceber melhor o que valorizas mais."


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    without_marks = "".join(char for char in normalized if not unicodedata.combining(char))
    return " ".join(without_marks.split())
