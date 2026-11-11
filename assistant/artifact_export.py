"""Read-only export intent and text projection for existing task artefacts."""
import re
import unicodedata


def export_kind(message: str) -> str | None:
    text = "".join(c for c in unicodedata.normalize("NFKD", message.casefold()) if not unicodedata.combining(c))
    if re.search(r"\b(?:nao|nunca|not|don't|dont)\b.*\b(?:guard|grav|sav)\w*", text):
        return None
    if not re.search(r"\b(?:guarda|guardar|guardes|grava|gravar|save|exporta|exportar|export)\b", text):
        return None
    if not re.search(r"\b(?:ficheiro|fecheiro|file|workspace|rascunho|draft|plano|plan|itinerario|itinerary|comparacao|comparison|resumo|summary|notas|notes)\b", text):
        return None
    aliases = {
        "compose_email": r"\b(?:email|e-mail|mail)\b",
        "plan_trip": r"\b(?:ferias|viagem|itinerario|itinerary|trip|holiday|vacation)\b",
        "generic_comparison": r"\b(?:comparacao|comparison)\b",
        "study_topic": r"\b(?:estudo|estudos|study)\b",
        "research_topic": r"\b(?:pesquisa|research)\b",
        "document": r"\b(?:documento|document|versao|version)\b",
    }
    return next((kind for kind, pattern in aliases.items() if re.search(pattern, text)), "")


def presentation_to_text(presentation: dict, language: str) -> str:
    lines = ["Current summary" if language == "en-GB" else "Resumo atual", str(presentation.get("title") or "")]
    for section in presentation.get("sections", []):
        if not section.get("text") and not section.get("items"):
            continue
        lines.extend(["", str(section.get("title") or "")])
        if section.get("text"):
            lines.append(str(section["text"]))
        for item in section.get("items", []):
            details = "; ".join(str(value) for value in item.get("details", []) if value)
            lines.append("- " + str(item.get("label") or "") + (f" ({details})" if details else ""))
    return "\n".join(lines).strip()
