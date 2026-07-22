from __future__ import annotations

import re
import sqlite3
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_PERSONAL_MODEL_DB = "personal_model.sqlite"
DEFAULT_CONFIDENCE = 100
HYPOTHESIS_THRESHOLD = 60
STRONG_KNOWLEDGE_THRESHOLD = 90

VALID_CATEGORIES = {
    "identidade",
    "vida",
    "trabalho",
    "estudos",
    "projetos",
    "ferramentas",
    "preferencias",
    "habitos",
    "relacoes",
    "objetivos",
}


@dataclass(frozen=True)
class PersonalModelEntry:
    id: int
    category: str
    key: str
    description: str
    confidence: int
    evidence: str
    source: str
    status: str
    created_at: str
    updated_at: str


class PersonalModel:
    """Structured model of what Echo knows about Alexandre.

    This is intentionally separate from raw conversation memory: entries need a
    category, evidence, source and confidence so hypotheses do not silently
    become facts.
    """

    def __init__(self, data_path: Path, db_file: str = DEFAULT_PERSONAL_MODEL_DB) -> None:
        self.data_path = data_path.resolve()
        self.db_path = (self.data_path / db_file).resolve()
        self.data_path.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def add_or_update_entry(
        self,
        category: str,
        key: str,
        description: str,
        confidence: int = DEFAULT_CONFIDENCE,
        evidence: str = "",
        source: str = "utilizador",
        status: str = "ativo",
    ) -> PersonalModelEntry:
        normalized_category = normalize_category(category)
        normalized_key = _clean_key(key) or _infer_key(description)
        clean_description = description.strip()
        if not clean_description:
            raise ValueError("A descricao nao pode estar vazia.")
        confidence_value = _clamp_confidence(confidence)
        now = datetime.now().isoformat(timespec="seconds")
        existing = self._entry_by_category_key(normalized_category, normalized_key)

        with sqlite3.connect(self.db_path) as connection:
            if existing is None:
                cursor = connection.execute(
                    """
                    INSERT INTO personal_model_entries (
                        category, key, description, confidence, evidence, source,
                        status, created_at, updated_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        normalized_category,
                        normalized_key,
                        clean_description,
                        confidence_value,
                        evidence.strip(),
                        source.strip() or "utilizador",
                        status.strip() or "ativo",
                        now,
                        now,
                    ),
                )
                entry_id = int(cursor.lastrowid)
                created_at = now
            else:
                merged_evidence = _merge_text(existing.evidence, evidence)
                merged_source = _merge_text(existing.source, source or "utilizador")
                connection.execute(
                    """
                    UPDATE personal_model_entries
                    SET description = ?,
                        confidence = ?,
                        evidence = ?,
                        source = ?,
                        status = ?,
                        updated_at = ?
                    WHERE id = ?
                    """,
                    (
                        clean_description,
                        max(existing.confidence, confidence_value),
                        merged_evidence,
                        merged_source,
                        status.strip() or existing.status or "ativo",
                        now,
                        existing.id,
                    ),
                )
                entry_id = existing.id
                created_at = existing.created_at

        return PersonalModelEntry(
            id=entry_id,
            category=normalized_category,
            key=normalized_key,
            description=clean_description,
            confidence=max(existing.confidence, confidence_value) if existing else confidence_value,
            evidence=_merge_text(existing.evidence, evidence) if existing else evidence.strip(),
            source=_merge_text(existing.source, source or "utilizador") if existing else (source.strip() or "utilizador"),
            status=status.strip() or (existing.status if existing else "ativo"),
            created_at=created_at,
            updated_at=now,
        )

    def remember_explicit(self, content: str) -> str:
        text = content.strip()
        if not text:
            return "Diz-me o que queres que eu guarde no meu modelo sobre ti."
        category = infer_category(text)
        key = _infer_key(text)
        entry = self.add_or_update_entry(
            category=category,
            key=key,
            description=text,
            confidence=100,
            evidence="Afirmação explícita do Alexandre.",
            source="utilizador",
            status="confirmado",
        )
        return format_personal_model_save_confirmation(entry)

    def search_personal_model(self, query: str, limit: int = 5) -> list[PersonalModelEntry]:
        text = _normalize(query)
        if not text:
            return []
        words = [word for word in re.findall(r"\w+", text) if len(word) > 2]
        entries = self._all_entries(include_inactive=False)
        scored: list[tuple[int, PersonalModelEntry]] = []
        for entry in entries:
            haystack = _normalize(
                f"{entry.category} {entry.key} {entry.description} {entry.evidence} {entry.source}"
            )
            score = sum(1 for word in words if word in haystack)
            if text in haystack:
                score += 3
            if score > 0:
                scored.append((score, entry))
        scored.sort(key=lambda item: (item[0], item[1].confidence, item[1].updated_at), reverse=True)
        return [entry for _, entry in scored[:limit]]

    def list_entries_by_category(self, category: str) -> list[PersonalModelEntry]:
        normalized_category = normalize_category(category)
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                """
                SELECT id, category, key, description, confidence, evidence, source,
                       status, created_at, updated_at
                FROM personal_model_entries
                WHERE category = ? AND status != 'apagado'
                ORDER BY confidence DESC, updated_at DESC, id DESC
                """,
                (normalized_category,),
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def get_relevant_context(self, query: str, limit: int = 5) -> str:
        entries = self.search_personal_model(query, limit=limit)
        if not entries:
            return ""
        lines = ["[personal_model]"]
        for entry in entries:
            strength = _confidence_label(entry.confidence)
            lines.append(
                f"- {strength}: [{entry.category}] {entry.description} "
                f"(confiança {entry.confidence}%; evidência: {entry.evidence or 'não indicada'})"
            )
        return "\n".join(lines)

    def decrease_confidence(self, query: str, amount: int = 20) -> PersonalModelEntry | None:
        matches = self.search_personal_model(query, limit=1)
        if not matches:
            return None
        entry = matches[0]
        new_confidence = _clamp_confidence(entry.confidence - abs(int(amount)))
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE personal_model_entries
                SET confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (new_confidence, now, entry.id),
            )
        return self._entry_by_id(entry.id)

    def delete_entry(self, query: str) -> PersonalModelEntry | None:
        matches = self.search_personal_model(query, limit=1)
        if not matches:
            return None
        entry = matches[0]
        now = datetime.now().isoformat(timespec="seconds")
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                UPDATE personal_model_entries
                SET status = 'apagado', updated_at = ?
                WHERE id = ?
                """,
                (now, entry.id),
            )
        return entry

    def entries_about(self, query: str = "") -> list[PersonalModelEntry]:
        return self.search_personal_model(query, limit=8) if query else self._all_entries(include_inactive=False)[:8]

    def answer_about(self, query: str = "", show_details: bool = False) -> str:
        entries = self.entries_about(query)
        if not entries:
            target = f" sobre {query}" if query else ""
            return (
                f"Ainda estamos no início{target}, mas vou construindo esse retrato com cuidado "
                "à medida que formos falando."
            )
        return format_personal_model_response(entries, query=query, show_details=show_details)

    def facts_about(self, query: str = "") -> list[str]:
        return [entry_to_fact(entry) for entry in self.entries_about(query)]

    def answer_category(self, category: str, show_details: bool = False) -> str:
        entries = self.list_entries_by_category(category)
        if not entries:
            return f"Ainda não tenho conhecimento suficiente sobre {normalize_category(category)}."
        return format_personal_model_response(
            entries,
            query=normalize_category(category),
            show_details=show_details,
        )

    def facts_for_category(self, category: str) -> list[str]:
        return [entry_to_fact(entry) for entry in self.list_entries_by_category(category)]

    def _entry_by_category_key(self, category: str, key: str) -> PersonalModelEntry | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, category, key, description, confidence, evidence, source,
                       status, created_at, updated_at
                FROM personal_model_entries
                WHERE category = ? AND key = ?
                LIMIT 1
                """,
                (category, key),
            ).fetchone()
        return _entry_from_row(row) if row else None

    def _entry_by_id(self, entry_id: int) -> PersonalModelEntry | None:
        with sqlite3.connect(self.db_path) as connection:
            row = connection.execute(
                """
                SELECT id, category, key, description, confidence, evidence, source,
                       status, created_at, updated_at
                FROM personal_model_entries
                WHERE id = ?
                """,
                (entry_id,),
            ).fetchone()
        return _entry_from_row(row) if row else None

    def _all_entries(self, include_inactive: bool = False) -> list[PersonalModelEntry]:
        status_filter = "" if include_inactive else "WHERE status != 'apagado'"
        with sqlite3.connect(self.db_path) as connection:
            rows = connection.execute(
                f"""
                SELECT id, category, key, description, confidence, evidence, source,
                       status, created_at, updated_at
                FROM personal_model_entries
                {status_filter}
                ORDER BY confidence DESC, updated_at DESC, id DESC
                """
            ).fetchall()
        return [_entry_from_row(row) for row in rows]

    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS personal_model_entries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    category TEXT NOT NULL,
                    key TEXT NOT NULL,
                    description TEXT NOT NULL,
                    confidence INTEGER NOT NULL DEFAULT 50,
                    evidence TEXT NOT NULL DEFAULT '',
                    source TEXT NOT NULL DEFAULT '',
                    status TEXT NOT NULL DEFAULT 'ativo',
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(category, key)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_personal_model_category
                ON personal_model_entries(category)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_personal_model_status
                ON personal_model_entries(status)
                """
            )


def normalize_category(category: str) -> str:
    normalized = _normalize(category)
    aliases = {
        "identidade": "identidade",
        "vida": "vida",
        "trabalho": "trabalho",
        "estudo": "estudos",
        "estudos": "estudos",
        "projeto": "projetos",
        "projetos": "projetos",
        "ferramenta": "ferramentas",
        "ferramentas": "ferramentas",
        "preferencia": "preferencias",
        "preferencias": "preferencias",
        "habito": "habitos",
        "habitos": "habitos",
        "relacao": "relacoes",
        "relacoes": "relacoes",
        "objetivo": "objetivos",
        "objetivos": "objetivos",
    }
    return aliases.get(normalized, "preferencias" if normalized not in VALID_CATEGORIES else normalized)


def format_personal_model_save_confirmation(entry: PersonalModelEntry) -> str:
    description = _naturalize_description(entry.description)
    tail = _category_save_tail(entry.category)
    if tail:
        return f"Fico com isso em mente: {description}. {tail}"
    return f"Fico com isso em mente: {description}."


def format_personal_model_response(
    entries: list[PersonalModelEntry],
    query: str = "",
    show_details: bool = False,
) -> str:
    if not entries:
        return "Ainda estamos no início, mas já estou a tentar perceber os padrões certos."

    natural_items = [entry_to_fact(entry) for entry in entries]
    if query:
        intro = f"Sobre {query}, para já tenho isto em mente:"
    else:
        intro = "Para já, sei algumas coisas sobre ti:"

    if len(natural_items) == 1:
        lines = [f"{intro} {natural_items[0]}"]
    else:
        lines = [intro]
        lines.extend(f"- {item}" for item in natural_items)

    if len(entries) <= 2:
        lines.append("Ainda é pouco, mas já me ajuda a adaptar melhor a forma como te acompanho.")
    else:
        lines.append("Isto ajuda-me a perceber melhor o teu contexto e a escolher melhor quando devo sugerir ferramentas, perguntas ou próximos passos.")

    if show_details:
        lines.append("")
        lines.append("Detalhes:")
        for entry in entries:
            lines.append(
                f"- Categoria: {entry.category}; chave: {entry.key}; confiança: {entry.confidence}%; "
                f"origem: {entry.source}; estado: {entry.status}; evidência: {entry.evidence or 'não indicada'}."
            )

    return "\n".join(lines)


def infer_category(text: str) -> str:
    normalized = _normalize(text)
    if any(word in normalized for word in ("chamo", "nome", "idioma", "lingua", "moro", "vivo em")):
        return "identidade"
    if any(word in normalized for word in ("estudo", "curso", "disciplina", "aprender", "rvcc")):
        return "estudos"
    if any(word in normalized for word in ("trabalho", "empresa", "funcao", "profissao")):
        return "trabalho"
    if any(word in normalized for word in ("projeto", "assistenteia", "echo")):
        return "projetos"
    if any(word in normalized for word in ("vscode", "vs code", "codex", "chatgpt", "git", "office", "chrome")):
        return "ferramentas"
    if any(word in normalized for word in ("gosto", "prefiro", "preferencia", "quero", "nao gosto")):
        return "preferencias"
    if any(word in normalized for word in ("costumo", "habitualmente", "rotina", "manha", "noite")):
        return "habitos"
    if any(word in normalized for word in ("joao", "maria", "colega", "familia", "amigo")):
        return "relacoes"
    if any(word in normalized for word in ("objetivo", "meta", "pretendo")):
        return "objetivos"
    if any(word in normalized for word in ("ferias", "lazer", "treino", "hobby")):
        return "vida"
    return "preferencias"


def _entry_from_row(row: tuple) -> PersonalModelEntry:
    return PersonalModelEntry(
        id=int(row[0]),
        category=str(row[1] or ""),
        key=str(row[2] or ""),
        description=str(row[3] or ""),
        confidence=int(row[4] or 0),
        evidence=str(row[5] or ""),
        source=str(row[6] or ""),
        status=str(row[7] or ""),
        created_at=str(row[8] or ""),
        updated_at=str(row[9] or ""),
    )


def _confidence_label(confidence: int) -> str:
    if confidence >= STRONG_KNOWLEDGE_THRESHOLD:
        return "conhecimento forte"
    if confidence < HYPOTHESIS_THRESHOLD:
        return "hipótese"
    return "conhecimento provável"


def entry_to_fact(entry: PersonalModelEntry) -> str:
    description = _naturalize_description(entry.description)
    if entry.confidence < HYPOTHESIS_THRESHOLD:
        return f"ainda como hipótese, {description}"
    if entry.confidence >= STRONG_KNOWLEDGE_THRESHOLD:
        return description
    return f"provavelmente, {description}"


def _naturalize_description(description: str) -> str:
    text = " ".join(description.strip().strip(".").split())
    if not text:
        return "essa informação"
    lowered = text.lower()
    prefixes = (
        "o alexandre ",
        "alexandre ",
        "o utilizador ",
        "eu ",
    )
    for prefix in prefixes:
        if lowered.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text and text[0].isupper():
        text = text[0].lower() + text[1:]
    return text


def _category_save_tail(category: str) -> str:
    tails = {
        "identidade": "Isto ajuda-me a tratar-te de forma mais coerente.",
        "vida": "Vou usar isto apenas quando for útil para contexto pessoal.",
        "trabalho": "Vou ter isto em conta quando falarmos do teu trabalho.",
        "estudos": "Isto ajuda-me a ajustar melhor o apoio nos teus estudos.",
        "projetos": "Vou associar isto ao teu contexto de projetos.",
        "ferramentas": "Isto ajuda-me a perceber melhor que ferramentas te devo sugerir.",
        "preferencias": "Vou adaptar melhor as respostas a essa preferência.",
        "habitos": "Vou tratar isto como um padrão a confirmar com o tempo.",
        "relacoes": "Vou guardar apenas o contexto útil e necessário.",
        "objetivos": "Isto ajuda-me a orientar melhor os próximos passos.",
    }
    return tails.get(category, "")


def _infer_key(text: str) -> str:
    normalized = _normalize(text)
    replacements = (
        "o utilizador ",
        "alexandre ",
        "eu ",
        "meu ",
        "minha ",
        "prefiro ",
        "gosto de ",
        "costumo ",
        "trabalho ",
        "estudo ",
    )
    for item in replacements:
        normalized = normalized.replace(item, " ")
    words = [word for word in re.findall(r"\w+", normalized) if len(word) > 2]
    return "-".join(words[:6]) or "entrada"


def _clean_key(value: str) -> str:
    return "-".join(re.findall(r"\w+", _normalize(value)))[:80]


def _clamp_confidence(value: int) -> int:
    return max(0, min(100, int(value)))


def _merge_text(left: str, right: str) -> str:
    parts = []
    for item in (left.strip(), right.strip()):
        if item and item not in parts:
            parts.append(item)
    return "; ".join(parts)


def _normalize(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text.lower())
    return "".join(char for char in normalized if not unicodedata.combining(char)).strip()
