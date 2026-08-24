from __future__ import annotations

import json
from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.presence_manager import PresenceManager
from assistant.task_intelligence import task_knowledge, task_reasoning
from assistant.tool_registry import ToolRegistry


class DummyMemory:
    def __init__(self):
        self.items = []

    def load(self):
        return list(self.items)

    def append_pair(self, user_message, assistant_message):
        self.items.append({"role": "user", "content": user_message})
        self.items.append({"role": "assistant", "content": assistant_message})

    def clear(self):
        self.items.clear()


class DummyLongTermMemory:
    def get_preference(self, key, default=None):
        return default

    def set_preference(self, key, value):
        pass


class DummyLLM:
    system_prompt = ""
    chat_call_count = 0
    chat_call_sources = []

    def chat(self, *args, **kwargs):
        raise AssertionError("Task research tests must not call the LLM")


def _engine(tmp_path: Path, registry: ToolRegistry | None = None) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return AssistantEngine(
        llm=DummyLLM(),
        memory=DummyMemory(),
        long_term_memory=DummyLongTermMemory(),
        tools=registry or ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt="",
        presence_manager=PresenceManager(),
    )


def _registry_with_claims(claims_by_query: dict[str, list[dict]]) -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry()
    queries: list[str] = []

    @registry.register("web_search", "Pesquisa web controlada.", ("web_search",))
    def web_search(query: str) -> str:
        queries.append(query)
        claims = []
        for key, value in claims_by_query.items():
            if key.lower() in query.lower():
                claims = value
                break
        return json.dumps(
            {
                "results": [
                    {
                        "title": f"Fonte para {query}",
                        "url": f"https://example.test/{len(queries)}",
                        "domain": "example.test",
                        "source_quality": "independent_review",
                        "snippet": f"Dados encontrados para {query}.",
                        "claims": claims,
                    }
                ]
            }
        )

    return registry, queries


def test_task_research_asks_for_exact_phone_models_before_camera_search(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.respond("vamos comparar Samsung com Apple, preciso de um telefone novo")

    response = engine.respond("pesquisa qual tem melhor câmara")
    telemetry = engine.get_last_turn_telemetry()

    assert "modelos exatos" in response
    assert telemetry["selected_path"] == "RESEARCH_REQUEST"
    assert telemetry["research_needs_clarification"] is True
    assert telemetry["llm_calls"] == 0


def test_task_research_ingests_web_claims_with_task_provenance(tmp_path: Path) -> None:
    registry, queries = _registry_with_claims(
        {
            "Samsung S24": [
                {"subject": "Samsung S24", "predicate": "camera_quality", "value": "good", "evidence": "Câmara forte em luz diurna."}
            ],
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "Vídeo e baixa luz mais consistentes."}
            ],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")
    task = engine._work_session.active_task
    assert task is not None

    response = engine.respond("confirma pela internet qual tem melhor câmara")
    entries = task_knowledge(engine._work_session, task.id)
    telemetry = engine.get_last_turn_telemetry()

    assert "Depois da pesquisa" in response
    assert any(entry.subject == "Samsung S24" and entry.source.source_type == "web" for entry in entries)
    assert any(entry.subject == "iPhone 15" and entry.predicate == "camera_quality" for entry in entries)
    assert all(entry.task_id == task.id for entry in entries)
    assert len(queries) == 2
    assert telemetry["llm_calls"] == 0
    assert telemetry["research_task_id"] == task.id
    assert telemetry["grounding_sources"] == ["TASK_KNOWLEDGE", "WEB_SOURCE"]


def test_task_research_preserves_conflicting_web_claims(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Samsung S24": [
                {"subject": "Samsung S24", "predicate": "battery", "value": "12 h", "evidence": "Teste A mediu 12 h."},
                {"subject": "Samsung S24", "predicate": "battery", "value": "15 h", "evidence": "Teste B mediu 15 h."},
            ],
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "battery", "value": "13 h", "evidence": "Teste C mediu 13 h."}
            ],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")

    response = engine.respond("pesquisa a bateria")
    conflict = engine.respond("há informação contraditória?")

    assert "conflito" in response.lower()
    assert "12 h" in conflict and "15 h" in conflict
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_task_research_updates_recommendation_after_new_web_evidence(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Samsung S24": [
                {"subject": "Samsung S24", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara."}
            ],
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "Câmara superior em testes."}
            ],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")
    engine.respond("para mim a câmara é mais importante do que a bateria")

    engine.respond("pesquisa só a câmara")
    reasoning = task_reasoning(engine._work_session, engine._work_session.active_task.id)

    assert reasoning.current_recommendation == "iPhone 15"
    assert reasoning.stale is False


def test_task_research_does_not_leak_between_tasks(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {"subject": "Estocolmo", "predicate": "activities", "value": "Vasa Museum", "evidence": "Atividade recomendada."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")
    phone_task = engine._work_session.active_task
    engine.respond("vou a Estocolmo de 10 a 15 de outubro e quero ajuda a planear")
    stockholm_task = engine._work_session.active_task

    engine.respond("pesquisa o que vale a pena fazer")

    assert phone_task is not None and stockholm_task is not None
    assert not any(entry.source.source_type == "web" for entry in task_knowledge(engine._work_session, phone_task.id))
    assert any(entry.subject == "Estocolmo" for entry in task_knowledge(engine._work_session, stockholm_task.id))


def test_travel_discovery_research_asks_for_origin_when_needed(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    engine.respond("quero férias de 7 dias até 1200 euros, sem voos muito longos, e quero descansar mas também visitar coisas")

    response = engine.respond("pesquisa destinos que façam sentido")

    assert "de onde vais partir" in response
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_travel_discovery_research_uses_origin_from_task_knowledge(tmp_path: Path) -> None:
    registry, queries = _registry_with_claims(
        {
            "Porto": [
                {"subject": "trip", "predicate": "fit_for_trip", "value": "opções realistas", "evidence": "Destinos com voos curtos a partir do Porto."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos, descanso mas também coisas para visitar")

    response = engine.respond("pesquisa destinos que façam sentido")

    assert "Depois da pesquisa" in response
    assert any("Porto" in query for query in queries)
    assert engine.get_last_turn_telemetry()["research_tool_called"] is True
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_known_destination_with_dates_starts_travel_task_without_explicit_plan_phrase(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    response = engine.respond("vou a Estocolmo de 10 a 15 de outubro")
    task = engine._work_session.active_task

    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)
    assert task.task_type == "plan_trip"
    assert "Estocolmo" in response
    assert any(entry.predicate == "destination" and entry.value == "Estocolmo" for entry in entries)
    assert any(entry.predicate == "date_range" and entry.value == "10 a 15 de outubro" for entry in entries)
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_generic_supplier_research_attaches_support_claims_to_options(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Alpha": [{"subject": "Alpha", "predicate": "support_quality", "value": "better", "evidence": "Suporte 24h confirmado."}],
            "Beta": [{"subject": "Beta", "predicate": "support_quality", "value": "limited", "evidence": "Só horário laboral."}],
            "Gamma": [{"subject": "Gamma", "predicate": "support_quality", "value": "limited", "evidence": "Resposta lenta em avaliações."}],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero escolher entre Alpha, Beta e Gamma")
    task = engine._work_session.active_task

    engine.respond("vê se estas empresas têm bom suporte")
    entries = task_knowledge(engine._work_session, task.id)

    assert {entry.subject for entry in entries if entry.predicate == "support_quality"} == {"Alpha", "Beta", "Gamma"}
    assert all(entry.source.source_type == "web" for entry in entries if entry.predicate == "support_quality")


def test_research_sources_are_queryable_and_can_be_excluded(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "Baixa luz superior."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")
    engine.respond("pesquisa a câmara")

    sources = engine.respond("de onde tiraste isso?")
    excluded = engine.respond("essa fonte está desatualizada, ignora-a")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)

    assert "https://example.test/" in sources
    assert "fora do raciocínio" in excluded
    assert any(entry.status == "excluded" for entry in entries if entry.source.source_type == "web")


def test_research_sources_do_not_swallow_why_followup(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "Baixa luz superior."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")
    engine.respond("pesquisa a câmara")
    engine.respond("ignora example.test")

    response = engine.respond("porque mudou a conclusão?")
    telemetry = engine.get_last_turn_telemetry()

    assert "Estas são as fontes" not in response
    assert telemetry["selected_path"] == "TASK_INTELLIGENCE"
    assert telemetry["llm_calls"] == 0
