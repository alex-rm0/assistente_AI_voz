from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import build_local_semantic_extractor
from assistant.conversation import AssistantEngine
from assistant.model_provider import ModelResponse
from assistant.presence_manager import PresenceManager
from assistant.task_intelligence import cognitive_task_view_model, task_knowledge, task_reasoning
from assistant.task_research import SemanticCandidate, _parse_semantic_candidates_with_status, _source_kind
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


class StubSemanticExtractor:
    def __init__(self, candidates: list[SemanticCandidate]):
        self.candidates = candidates
        self.requests = []

    def extract(self, request):
        self.requests.append(request)
        return list(self.candidates)


class RaisingSemanticExtractor:
    def __init__(self, exc: Exception | None = None):
        self.exc = exc or AssertionError("Semantic extraction should not run")
        self.requests = []

    def extract(self, request):
        self.requests.append(request)
        raise self.exc


class DynamicSemanticExtractor:
    def __init__(self, candidates_by_marker: dict[str, list[SemanticCandidate]]):
        self.candidates_by_marker = candidates_by_marker
        self.requests = []

    def extract(self, request):
        self.requests.append(request)
        candidates: list[SemanticCandidate] = []
        for marker, marker_candidates in self.candidates_by_marker.items():
            if marker in request.evidence:
                candidates.extend(marker_candidates)
        return candidates


def _engine(
    tmp_path: Path,
    registry: ToolRegistry | None = None,
    semantic_extractor: StubSemanticExtractor | None = None,
    llm=None,
) -> AssistantEngine:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True)
    return AssistantEngine(
        llm=llm or DummyLLM(),
        memory=DummyMemory(),
        long_term_memory=DummyLongTermMemory(),
        tools=registry or ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt="",
        presence_manager=PresenceManager(),
        semantic_extractor=semantic_extractor,
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


def _registry_with_results(results_by_query: dict[str, list[dict]]) -> tuple[ToolRegistry, list[str]]:
    registry = ToolRegistry()
    queries: list[str] = []

    @registry.register("web_search", "Pesquisa web controlada.", ("web_search",))
    def web_search(query: str) -> str:
        queries.append(query)
        results = []
        for key, value in results_by_query.items():
            if key.lower() in query.lower():
                results = value
                break
        return json.dumps({"results": results})

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

    assert "Pesquisa concluída" in response
    assert any(entry.subject == "Samsung S24" and entry.source.source_type == "web" for entry in entries)
    assert any(entry.subject == "iPhone 15" and entry.predicate == "camera_quality" for entry in entries)
    assert all(entry.task_id == task.id for entry in entries)
    assert len(queries) == 2
    assert telemetry["llm_calls"] == 0
    assert telemetry["research_task_id"] == task.id
    assert telemetry["grounding_sources"] == ["TASK_KNOWLEDGE", "WEB_SOURCE"]


def test_task_research_does_not_turn_measurement_variants_into_conflicts(tmp_path: Path) -> None:
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

    assert "conflito" not in response.lower()
    assert "Não vejo informação contraditória" in conflict
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_task_research_preserves_genuine_web_conflicts(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {"subject": "Estocolmo", "predicate": "opening_hours", "value": "open Monday", "evidence": "Fonte A diz que abre à segunda."},
                {"subject": "Estocolmo", "predicate": "opening_hours", "value": "closed Monday", "evidence": "Fonte B diz que fecha à segunda."},
            ],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa horários e atividades")
    conflict = engine.respond("há informação contraditória?")

    assert "conflito" in response.lower()
    assert "Estocolmo" in conflict
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
                {"subject": "Valência", "predicate": "fit_for_trip", "value": "cidade com voo curto e coisas para visitar", "evidence": "Valência aparece como opção com voos curtos a partir do Porto."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos, descanso mas também coisas para visitar")

    response = engine.respond("pesquisa destinos que façam sentido")

    assert "destino plausível" in response
    assert any("Porto" in query for query in queries)
    assert any(entry.subject == "Valência" and entry.predicate == "destination_candidate" for entry in task_knowledge(engine._work_session, engine._work_session.active_task.id))
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


def test_model_numbers_are_not_interpreted_as_prices(tmp_path: Path) -> None:
    engine = _engine(tmp_path)

    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)

    assert any(entry.kind == "option" and entry.subject == "Samsung Galaxy S24" for entry in entries)
    assert any(entry.kind == "option" and entry.subject == "iPhone 15" for entry in entries)
    assert not [entry for entry in entries if entry.predicate == "price"]


def test_related_phone_variant_is_not_direct_evidence_for_base_model(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Samsung Galaxy S24": [
                {
                    "subject": "Samsung Galaxy S24 Ultra",
                    "predicate": "camera_quality",
                    "value": "better",
                    "evidence": "O S24 Ultra tem zoom superior.",
                }
            ],
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara geral."}
            ],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    task = engine._work_session.active_task

    response = engine.respond("pesquisa qual tem melhor câmara")
    entries = task_knowledge(engine._work_session, task.id)

    assert "modelos relacionados" in response
    assert not any(
        entry.subject == "Samsung Galaxy S24"
        and entry.predicate == "camera_quality"
        and entry.source.source_type == "web"
        for entry in entries
    )


def test_iphone_pro_max_title_is_not_exact_evidence_for_base_iphone(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 Pro Max - Professional Camera Test and Review",
                    "url": "https://www.youtube.com/watch?v=pro-max",
                    "domain": "youtube.com",
                    "source_quality": "web",
                    "snippet": "The iPhone 15 Pro Max camera has excellent zoom and video.",
                    "claims": [
                        {
                            "subject": "iPhone 15",
                            "predicate": "camera_quality",
                            "value": "better",
                            "evidence": "The iPhone 15 Pro Max camera has excellent zoom and video.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    task = engine._work_session.active_task

    response = engine.respond("pesquisa mais sobre a câmara do iPhone")
    entries = task_knowledge(engine._work_session, task.id)

    assert "modelos relacionados" in response
    assert not any(entry.subject == "iPhone 15" and entry.predicate == "camera_quality" and entry.source.source_type == "web" for entry in entries)


def test_base_iphone_passage_is_exact_even_on_mixed_variant_page(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 and iPhone 15 Pro camera comparison",
                    "url": "https://example.test/iphone-15-mixed",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "The base iPhone 15 performs well, while the iPhone 15 Pro has extra zoom.",
                    "claims": [
                        {
                            "subject": "iPhone 15",
                            "predicate": "camera_quality",
                            "value": "good",
                            "evidence": "The base iPhone 15 performs well in daylight and video.",
                        },
                        {
                            "subject": "iPhone 15 Pro",
                            "predicate": "camera_quality",
                            "value": "better",
                            "evidence": "The iPhone 15 Pro has extra zoom.",
                        },
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    task = engine._work_session.active_task

    engine.respond("pesquisa mais sobre a câmara do iPhone")
    entries = task_knowledge(engine._work_session, task.id)

    assert any(entry.subject == "iPhone 15" and entry.metadata.get("applicability") == "exact" for entry in entries)
    assert not any(entry.subject == "iPhone 15 Pro" and entry.status == "active" for entry in entries)


def test_travel_research_uses_origin_as_origin_not_destination(tmp_path: Path) -> None:
    registry, queries = _registry_with_claims(
        {
            "European city break": [
                {"subject": "Estocolmo", "predicate": "fit_for_trip", "value": "boa opção", "evidence": "Voos curtos a partir do Porto."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos")

    engine.respond("pesquisa destinos que façam sentido")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)

    assert any("from Porto" in query or "FROM Porto" in query for query in queries)
    assert not any(entry.predicate == "destination" and entry.value == "Porto" for entry in entries)


def test_irrelevant_same_keyword_result_is_rejected(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {"subject": "Camões Institute Stockholm protocol", "predicate": "activities", "value": "academic protocol", "evidence": "University cooperation protocol."},
                {"subject": "Estocolmo", "predicate": "activities", "value": "Vasa Museum", "evidence": "Museum and tourism attraction in Stockholm."},
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")
    task = engine._work_session.active_task

    response = engine.respond("pesquisa atividades")
    entries = task_knowledge(engine._work_session, task.id)

    assert "Descartei" in response
    assert any(entry.value == "Vasa Museum" for entry in entries)
    assert not any("Camões" in entry.subject for entry in entries)


def test_article_titles_are_not_materialized_as_destination_candidates(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "European city break": [
                {"subject": "7 Day Portugal Itinerary", "predicate": "fit_for_trip", "value": "roteiro genérico", "evidence": "Article title, not a destination."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos")
    task = engine._work_session.active_task

    response = engine.respond("pesquisa destinos que façam sentido")
    entries = task_knowledge(engine._work_session, task.id)

    assert (
        "conhecimento seguro" in response
        or "descartei" in response.lower()
        or "ainda não consegui extrair destinos fiáveis" in response
    )
    assert not any(entry.subject == "7 Day Portugal Itinerary" for entry in entries)


def test_travel_discovery_creates_destination_candidates_not_article_snippets(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "European city break": [
                {
                    "subject": "Valência",
                    "predicate": "fit_for_trip",
                    "value": "voo curto, cidade com praia e visitas culturais",
                    "evidence": "Valência combina voo curto a partir do Porto, praia urbana e centro histórico.",
                },
                {
                    "subject": "Malta",
                    "predicate": "fit_for_trip",
                    "value": "descanso e sítios históricos",
                    "evidence": "Malta encaixa em descanso e visitas históricas, com voos curtos sazonais.",
                },
                {
                    "subject": "Sevilha",
                    "predicate": "fit_for_trip",
                    "value": "cidade próxima com muitas visitas",
                    "evidence": "Sevilha é próxima e concentra monumentos, bairros históricos e boa oferta urbana.",
                },
                {
                    "subject": "20 Best European City Breaks",
                    "predicate": "fit_for_trip",
                    "value": "raw guide title",
                    "evidence": "Travel guide title, not a destination candidate.",
                },
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos, quero descansar mas também ter coisas para visitar")

    response = engine.respond("pesquisa destinos que façam sentido")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    view = cognitive_task_view_model(engine._work_session, task)
    view_text = str(view)

    candidates = {entry.subject for entry in entries if entry.kind == "candidate" and entry.predicate == "destination_candidate"}
    assert {"Valência", "Malta", "Sevilha"}.issubset(candidates)
    assert "20 Best European City Breaks" not in candidates
    assert not any(entry.subject == "trip" and "Travel guide" in entry.value for entry in entries)
    assert "destinos" in response
    assert "Valência" in response
    assert "Travel guide title" not in view_text
    assert "snippet" not in view_text.lower()
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_travel_discovery_extracts_candidates_from_raw_evidence_and_excludes_origin(tmp_path: Path) -> None:
    registry, queries = _registry_with_claims(
        {
            "European city break": [
                {
                    "subject": "trip",
                    "predicate": "web_evidence",
                    "value": "While Porto has historic wine houses, Alicante Amsterdam Athens Barcelona Berlin and Seville are common city break destination ideas from Porto.",
                    "evidence": "While Porto has historic wine houses, Alicante Amsterdam Athens Barcelona Berlin and Seville are common city break destination ideas from Porto for short flights and sightseeing.",
                }
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Porto", "destination", "Porto has historic wine houses", 0.9),
            SemanticCandidate("Alicante", "destination", "Alicante Amsterdam Athens Barcelona", 0.9),
            SemanticCandidate("Amsterdam", "destination", "Alicante Amsterdam Athens Barcelona", 0.9),
            SemanticCandidate("Athens", "destination", "Alicante Amsterdam Athens Barcelona", 0.9),
            SemanticCandidate("Barcelona", "destination", "Alicante Amsterdam Athens Barcelona", 0.9),
            SemanticCandidate("Berlin", "destination", "Barcelona Berlin and Seville", 0.9),
            SemanticCandidate("Roman Empire", "destination", "not present", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos, quero descansar mas também ter coisas para visitar")

    response = engine.respond("pesquisa destinos que façam sentido")
    research_telemetry = engine.get_last_turn_telemetry()
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    view_text = str(cognitive_task_view_model(engine._work_session, task))

    candidates = {entry.subject for entry in entries if entry.kind == "candidate" and entry.predicate == "destination_candidate"}
    assert {"Alicante", "Amsterdam", "Athens", "Barcelona", "Berlin"}.intersection(candidates)
    assert "Porto" not in candidates
    assert not any(entry.subject == "trip" and entry.predicate == "web_evidence" for entry in entries)
    assert "informação útil sobre trip" not in response
    assert "While Porto" not in view_text
    assert all("destination" in query.lower() and "from porto" in query.lower() for query in queries)
    assert not any("things to do in porto" in query.lower() or "porto itinerary" in query.lower() for query in queries)
    assert research_telemetry["llm_calls"] == 0
    assert research_telemetry["semantic_extraction_calls"] >= 1


def test_stockholm_research_creates_semantic_activities_and_keeps_surface_concise(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "Vasa Museum",
                    "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                },
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "Gamla Stan",
                    "evidence": "Visit Stockholm destaca Gamla Stan para passear e visitar.",
                },
                {
                    "subject": "Camões Institute Stockholm protocol",
                    "predicate": "activities",
                    "value": "academic protocol",
                    "evidence": "University cooperation protocol.",
                },
                {
                    "subject": "Stockholm Freight A/S",
                    "predicate": "transport",
                    "value": "road freight",
                    "evidence": "Road freight company based in Stockholm.",
                },
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa o que vale a pena fazer")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    view = cognitive_task_view_model(engine._work_session, task)
    view_text = str(view)

    assert any(entry.predicate == "activities" and entry.value == "Vasa Museum" for entry in entries)
    assert any(entry.predicate == "activities" and entry.value == "Gamla Stan" for entry in entries)
    assert not any("Camões Institute" in entry.subject for entry in entries)
    assert not any("Freight" in entry.subject for entry in entries)
    assert "atividades" in response or "Estocolmo" in response
    assert "Vasa Museum" in view_text
    assert "University cooperation" not in view_text
    assert len(view_text) < 1400
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_stockholm_place_extraction_uses_named_entities_not_sentences(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "Explore the island of Djurgården, home to some of Stockholm's museums.",
                    "evidence": "Explore the island of Djurgården, home to some of Stockholm's museums.",
                },
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "Passing by the famous Vasa Museet is recommended.",
                    "evidence": "Passing by the famous Vasa Museet is recommended.",
                },
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "Opening hours - Nobel Prize Museum",
                    "evidence": "Opening hours - Nobel Prize Museum",
                },
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "Nationalmuseum is a major art museum.",
                    "evidence": "Nationalmuseum is a major art museum.",
                },
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Djurgården", "place", "island of Djurgården", 0.9),
            SemanticCandidate("Vasa Museum", "attraction", "Vasa Museet", 0.9, ("Vasa Museet",)),
            SemanticCandidate("Nobel Prize Museum", "attraction", "Nobel Prize Museum", 0.9),
            SemanticCandidate("Nationalmuseum", "place", "Nationalmuseum", 0.9),
            SemanticCandidate("Tuesday", "activity", "Tuesday", 0.9),
            SemanticCandidate("Swedish", "place", "Swedish", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    research_telemetry = engine.get_last_turn_telemetry()
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    values = {entry.value for entry in entries if entry.predicate == "activities"}
    response = engine.respond("o que encontraste?")

    assert {"Djurgården", "Vasa Museum", "Nobel Prize Museum", "Nationalmuseum"}.issubset(values)
    assert not any("Explore the island" in value or "Passing by" in value for value in values)
    assert "Djurgården" in response
    assert "Explore the island" not in response
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0
    assert research_telemetry["semantic_extraction_calls"] >= 1


def test_raw_destination_prose_fails_safely_without_semantic_extractor(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "European city break": [
                {
                    "subject": "trip",
                    "predicate": "web_evidence",
                    "value": "Uncover Marrakech Alicante Amsterdam Athens Barcelona with cheap city breaks by Jeanne.",
                    "evidence": "Uncover Marrakech Alicante Amsterdam Athens Barcelona with cheap city breaks by Jeanne.",
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos")

    response = engine.respond("pesquisa destinos que façam sentido")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)

    assert (
        "descartei" in response
        or "ainda não consegui transformá-los em conhecimento seguro" in response
        or "ainda não consegui extrair destinos fiáveis" in response
    )
    assert not any(entry.predicate == "destination_candidate" for entry in entries)
    assert engine.get_last_turn_telemetry()["semantic_extraction_calls"] == 0
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_semantic_extractor_bad_candidates_are_rejected(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "European city break": [
                {
                    "subject": "trip",
                    "predicate": "web_evidence",
                    "value": "Alicante and Marrakech are mentioned as short flight destination ideas from Porto.",
                    "evidence": "Alicante and Marrakech are mentioned as short flight destination ideas from Porto.",
                }
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Tuesday", "destination", "Tuesday", 0.99),
            SemanticCandidate("City Breaks", "destination", "City Breaks", 0.99),
            SemanticCandidate("Porto", "destination", "Porto", 0.99),
            SemanticCandidate("Alicante", "destination", "Alicante", 0.99),
            SemanticCandidate("Lisboa", "destination", "Lisboa", 0.99),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos")

    engine.respond("pesquisa destinos que façam sentido")
    task = engine._work_session.active_task
    candidates = {entry.subject for entry in task_knowledge(engine._work_session, task.id) if entry.predicate == "destination_candidate"}

    assert candidates == {"Alicante"}


def test_travel_task_start_does_not_duplicate_semantic_fields_as_visible_notes(tmp_path: Path) -> None:
    engine = _engine(tmp_path)
    message = "quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos"

    engine.respond(message)
    task = engine._work_session.active_task
    view_text = str(cognitive_task_view_model(engine._work_session, task))

    assert task is not None
    assert task.parameters.get("origin") == "Porto"
    assert task.parameters.get("budget_max") == "1200 euros"
    assert message not in task.collected_facts
    assert "Notas:" not in view_text


def test_raw_transport_snippet_is_converted_or_kept_out_of_task_surface(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {
                    "subject": "Estocolmo",
                    "predicate": "transport",
                    "value": "There is a transit fee for passing to and from Arlanda on the airport train.",
                    "evidence": "There is a transit fee for passing to and from Arlanda on the airport train.",
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa transportes")
    task = engine._work_session.active_task
    view_text = str(cognitive_task_view_model(engine._work_session, task))

    assert "There is a transit fee" not in view_text
    assert any(entry.predicate == "transport" for entry in task_knowledge(engine._work_session, task.id))


def test_source_kind_classifies_community_subdomains_before_official_domains() -> None:
    assert _source_kind({"url": "https://discussions.apple.com/thread/123", "domain": "discussions.apple.com", "title": "User discussion"}) == "community"
    assert _source_kind({"url": "https://support.apple.com/en-us/123", "domain": "support.apple.com", "title": "Apple Support"}) == "official"
    assert _source_kind({"url": "https://www.apple.com/iphone/specs", "domain": "www.apple.com", "title": "iPhone specs"}) == "official"
    assert _source_kind({"url": "https://community.samsung.com/t5/phones", "domain": "community.samsung.com", "title": "Community"}) == "community"
    assert _source_kind({"url": "https://www.samsung.com/pt/smartphones/specs", "domain": "www.samsung.com", "title": "Samsung specs"}) == "official"


def test_same_source_duplicate_claims_do_not_create_multiple_active_entries(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera review",
                    "url": "https://example.test/iphone-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Repeated camera result.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara em pouca luz."},
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara em pouca luz."},
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara em pouca luz."},
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara em pouca luz."},
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    entries = [
        entry for entry in task_knowledge(engine._work_session, task.id)
        if entry.source.reference == "https://example.test/iphone-camera" and entry.predicate == "camera_quality"
    ]

    assert len(entries) == 1


def test_recommendation_does_not_overreach_when_only_one_priority_side_has_evidence(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera lab review",
                    "url": "https://example.test/iphone-camera-lab",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Teste técnico de câmara.",
                    "claims": [
                        {
                            "subject": "iPhone 15",
                            "predicate": "camera_quality",
                            "value": "better",
                            "evidence": "A câmara do iPhone tem vídeo estável e melhor consistência em baixa luz.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("a câmara é a parte mais importante para mim")
    engine.respond("pesquisa mais sobre a câmara do iPhone")

    response = engine.respond("qual parece melhor?")
    next_step = engine.respond("qual é o próximo passo?")
    reasoning = task_reasoning(engine._work_session, engine._work_session.active_task.id)

    assert "Ainda não escolheria" in response
    assert "câmara de Samsung Galaxy S24" in response
    assert "câmara de Samsung Galaxy S24" in next_step
    assert reasoning.current_recommendation == ""
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_travel_cost_estimates_do_not_create_giant_conflict_blocks(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {"subject": "Estocolmo", "predicate": "price", "value": "120 € por dia", "evidence": "Estimativa média de 120 € por dia."},
                {"subject": "Estocolmo", "predicate": "price", "value": "180 € por dia", "evidence": "Estimativa alta de 180 € por dia."},
                {"subject": "Estocolmo", "predicate": "price", "value": "95 € por dia", "evidence": "Estimativa económica de 95 € por dia."},
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa custos")
    task = engine._work_session.active_task
    view_text = str(cognitive_task_view_model(engine._work_session, task))
    conflict = engine.respond("há informação contraditória?")

    assert "Há conflito" not in response
    assert "Atenção" not in view_text
    assert "120 € por dia; 180 € por dia; 95 € por dia" not in view_text
    assert "Não vejo informação contraditória" in conflict
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_research_source_can_be_opened_with_confirmation(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "Baixa luz superior."}
            ]
        }
    )
    opened: list[str] = []

    @registry.register("open_url", "Abre URL.", ("desktop:open_url", "confirmation-required"))
    def open_url(url: str) -> str:
        opened.append(url)
        return f"URL aberto: {url}"

    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung S24 com iPhone 15")
    engine.respond("pesquisa a câmara")

    response = engine.respond("abre essa fonte")
    telemetry = engine.get_last_turn_telemetry()

    assert "Queres que abra" in response
    assert not opened
    assert telemetry["llm_calls"] == 0
    assert telemetry["research_source_open_requested"] is True

    confirmed = engine.respond("sim")

    assert opened == ["https://example.test/2"]
    assert "URL aberto" in confirmed
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_show_sources_is_content_operation_not_workspace_attention(tmp_path: Path) -> None:
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

    response = engine.respond("mostra as fontes")
    telemetry = engine.get_last_turn_telemetry()
    events = [json.loads(event) for event in engine.consume_ui_events()]

    assert "usei" in response.lower()
    assert "fonte" in response.lower()
    assert telemetry["selected_path"] == "CONTENT_OPERATION"
    assert telemetry["content_operation"] == "show"
    assert telemetry["content_target_type"] == "source_list"
    assert telemetry["research_sources_requested"] is True
    assert telemetry["llm_calls"] == 0
    assert any(event["type"] == "research_results_ready" for event in events)


def test_named_research_source_dxomark_outranks_unrelated_source(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung Galaxy S24 official specs",
                    "url": "https://www.samsung.com/galaxy/s24/",
                    "domain": "samsung.com",
                    "source_quality": "official",
                    "snippet": "Samsung official product page.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "Câmara forte."}
                    ],
                },
                {
                    "title": "DXOMARK Samsung Galaxy S24 Camera test",
                    "url": "https://www.dxomark.com/samsung-galaxy-s24-camera-test/",
                    "domain": "dxomark.com",
                    "source_quality": "independent_review",
                    "snippet": "Independent camera test.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "better", "evidence": "Teste independente da câmara."}
                    ],
                },
                {
                    "title": "Android Central Galaxy S24 review",
                    "url": "https://www.androidcentral.com/phones/galaxy-s24-review",
                    "domain": "androidcentral.com",
                    "source_quality": "independent_review",
                    "snippet": "General Android Central review.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "Análise geral positiva."}
                    ],
                },
            ],
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera review",
                    "url": "https://example.test/iphone-15-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "iPhone camera review.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara."}
                    ],
                }
            ],
        }
    )
    opened: list[str] = []

    @registry.register("open_url", "Abre URL.", ("desktop:open_url", "confirmation-required"))
    def open_url(url: str) -> str:
        opened.append(url)
        return f"URL aberto: {url}"

    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    engine.respond("pesquisa qual tem melhor câmara")

    response = engine.respond("abre a fonte da DXOMARK")
    telemetry = engine.get_last_turn_telemetry()

    assert "DXOMARK" in response or "dxomark" in telemetry["research_source_url"]
    assert telemetry["selected_path"] == "CONTENT_OPERATION"
    assert telemetry["research_source_url"] == "https://www.dxomark.com/samsung-galaxy-s24-camera-test/"
    assert telemetry["llm_calls"] == 0
    assert not opened


def test_source_ordinal_deictic_and_continuation_are_coherent(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung official source",
                    "url": "https://www.samsung.com/source",
                    "domain": "samsung.com",
                    "source_quality": "official",
                    "snippet": "Official source.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "Detalhes oficiais."}
                    ],
                },
                {
                    "title": "DXOMARK camera source",
                    "url": "https://www.dxomark.com/source",
                    "domain": "dxomark.com",
                    "source_quality": "independent_review",
                    "snippet": "DXOMARK source.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "better", "evidence": "Teste DXOMARK."}
                    ],
                },
            ],
            "iPhone 15": [
                {
                    "title": "iPhone review source",
                    "url": "https://example.test/iphone",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "iPhone source.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Teste iPhone."}
                    ],
                }
            ],
        }
    )
    opened: list[str] = []

    @registry.register("open_url", "Abre URL.", ("desktop:open_url", "confirmation-required"))
    def open_url(url: str) -> str:
        opened.append(url)
        return f"URL aberto: {url}"

    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    engine.respond("pesquisa qual tem melhor câmara")

    first = engine.respond("lê a primeira fonte")
    first_telemetry = engine.get_last_turn_telemetry()
    detail = engine.respond("mais detalhe")
    second = engine.respond("agora a segunda")
    second_telemetry = engine.get_last_turn_telemetry()
    summarized = engine.respond("resume")
    open_response = engine.respond("abre")
    open_telemetry = engine.get_last_turn_telemetry()

    assert "DXOMARK camera source" in first
    assert first_telemetry["content_target_resolution"] == "ordinal"
    assert "DXOMARK camera source" in detail
    assert "iPhone review source" in second
    assert second_telemetry["research_source_url"] == ""
    assert "iPhone review source" in summarized
    assert "Queres que abra" in open_response
    assert open_telemetry["research_source_url"] == "https://example.test/iphone"
    assert open_telemetry["llm_calls"] == 0
    assert not opened


def test_content_operations_distinguish_comparison_summary_recommendation_and_sources(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Samsung Galaxy S24": [
                {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "Boa câmara."}
            ],
            "iPhone 15": [
                {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "Câmara superior."}
            ],
        }
    )
    opened: list[str] = []

    @registry.register("open_url", "Abre URL.", ("desktop:open_url", "confirmation-required"))
    def open_url(url: str) -> str:
        opened.append(url)
        return f"URL aberto: {url}"

    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    engine.respond("para mim a câmara é mais importante do que a bateria")
    engine.respond("pesquisa a câmara")

    read_response = engine.respond("lê-me a comparação")
    read_telemetry = engine.get_last_turn_telemetry()
    summary_response = engine.respond("resume a comparação")
    summary_telemetry = engine.get_last_turn_telemetry()
    answer_response = engine.respond("qual é melhor?")
    answer_telemetry = engine.get_last_turn_telemetry()

    assert "Estamos a comparar" in read_response
    assert read_telemetry["selected_path"] == "CONTENT_OPERATION"
    assert summary_telemetry["content_operation"] == "summarize"
    assert len(summary_response) <= len(read_response)
    assert "iPhone 15" in answer_response
    assert answer_telemetry["selected_path"] == "TASK_INTELLIGENCE"
    assert answer_telemetry["llm_calls"] == 0


def test_task_section_and_research_synthesis_are_addressable(tmp_path: Path) -> None:
    registry, _queries = _registry_with_claims(
        {
            "Estocolmo": [
                {"subject": "Estocolmo", "predicate": "activities", "value": "Vasa Museum", "evidence": "Museu recomendado."}
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")
    engine.respond("pesquisa o que vale a pena fazer")

    found = engine.respond("diz-me o que encontraste")
    found_telemetry = engine.get_last_turn_telemetry()
    missing = engine.respond("lê o que falta")
    missing_telemetry = engine.get_last_turn_telemetry()

    assert "Encontrei" in found
    assert found_telemetry["content_target_type"] == "research_synthesis"
    assert "Ainda falta" in missing
    assert missing_telemetry["content_target_type"] == "section"
    assert missing_telemetry["llm_calls"] == 0


def test_source_references_are_scoped_to_active_task_after_switching(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung source",
                    "url": "https://www.samsung.com/source",
                    "domain": "samsung.com",
                    "source_quality": "official",
                    "snippet": "Samsung data.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "Samsung evidence."}
                    ],
                }
            ],
            "iPhone 15": [
                {
                    "title": "Apple source",
                    "url": "https://www.apple.com/iphone-15/",
                    "domain": "apple.com",
                    "source_quality": "official",
                    "snippet": "Apple data.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Apple evidence."}
                    ],
                }
            ],
            "Estocolmo": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {"subject": "Estocolmo", "predicate": "activities", "value": "Djurgården", "evidence": "Zona recomendada."}
                    ],
                }
            ],
        }
    )
    opened: list[str] = []

    @registry.register("open_url", "Abre URL.", ("desktop:open_url", "confirmation-required"))
    def open_url(url: str) -> str:
        opened.append(url)
        return f"URL aberto: {url}"

    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 com iPhone 15")
    phone_task = engine._work_session.active_task
    engine.respond("pesquisa a câmara")
    phone_source = engine.respond("lê a primeira fonte")

    engine.respond("vou a Estocolmo de 10 a 15 de outubro")
    stockholm_task = engine._work_session.active_task
    engine.respond("pesquisa o que vale a pena fazer")
    stockholm_source = engine.respond("lê a primeira fonte")

    engine.respond("volta a comparar Samsung")
    back_to_phone = engine.respond("abre a fonte da Samsung")
    telemetry = engine.get_last_turn_telemetry()

    assert phone_task is not None and stockholm_task is not None
    assert "Samsung source" in phone_source
    assert "Visit Stockholm" in stockholm_source
    assert "Queres que abra" in back_to_phone
    assert telemetry["research_source_url"] == "https://www.samsung.com/source"
    assert telemetry["llm_calls"] == 0
    assert not opened


def test_explicit_iphone_research_target_overrides_comparison_scope(tmp_path: Path) -> None:
    registry, queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera source",
                    "url": "https://example.test/iphone-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "iPhone camera evidence.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "A câmara do iPhone tem melhor consistência."}
                    ],
                }
            ],
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung camera source",
                    "url": "https://example.test/samsung-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Samsung camera evidence.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "A câmara do Samsung é boa."}
                    ],
                }
            ],
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")

    response = engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    telemetry = engine.get_last_turn_telemetry()

    assert queries == ["iPhone 15 base model camera review independent test low light video"]
    assert "iPhone 15" in response
    assert "Samsung Galaxy S24" not in response
    assert any(entry.subject == "iPhone 15" and entry.source.source_type == "web" for entry in entries)
    assert not any(entry.subject == "Samsung Galaxy S24" and entry.source.source_type == "web" for entry in entries)
    assert telemetry["llm_calls"] == 0


def test_evidence_question_uses_sources_without_mutating_task_notes(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera lab review",
                    "url": "https://example.test/iphone-camera-lab",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Teste técnico de câmara.",
                    "claims": [
                        {
                            "subject": "iPhone 15",
                            "predicate": "camera_quality",
                            "value": "good",
                            "evidence": "A análise técnica mostra boa exposição, vídeo estável e bom detalhe em pouca luz.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    before = list(task_knowledge(engine._work_session, task.id))

    response = engine.respond("com estas fontes, já consegues dizer se a câmara do iPhone é boa?")
    telemetry = engine.get_last_turn_telemetry()
    after = list(task_knowledge(engine._work_session, task.id))

    assert telemetry["selected_path"] == "TASK_INTELLIGENCE"
    assert telemetry["llm_calls"] == 0
    assert len(after) == len(before)
    assert "iPhone" in response
    assert "câmara" in response
    assert "evidência" in response
    assert "preço" not in response.lower()
    assert not any(entry.value == "com estas fontes, já consegues dizer se a câmara do iPhone é boa?" for entry in after)


def test_evidence_sufficiency_distinguishes_single_subject_from_full_comparison(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera lab review",
                    "url": "https://example.test/iphone-camera-lab",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Teste técnico de câmara.",
                    "claims": [
                        {
                            "subject": "iPhone 15",
                            "predicate": "camera_quality",
                            "value": "good",
                            "evidence": "A análise técnica mostra boa exposição, vídeo estável e bom detalhe em pouca luz.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    before = len(task_knowledge(engine._work_session, task.id))

    response = engine.respond("com estas fontes, já chega?")
    telemetry = engine.get_last_turn_telemetry()

    assert telemetry["selected_path"] == "TASK_INTELLIGENCE"
    assert telemetry["llm_calls"] == 0
    assert len(task_knowledge(engine._work_session, task.id)) == before
    assert "não para fechar a comparação" in response
    assert "Samsung Galaxy S24" in response


def test_evidence_question_does_not_overclaim_comparison_from_one_subject(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera lab review",
                    "url": "https://example.test/iphone-camera-lab",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Teste técnico de câmara.",
                    "claims": [
                        {
                            "subject": "iPhone 15",
                            "predicate": "camera_quality",
                            "value": "better",
                            "evidence": "A câmara do iPhone tem vídeo estável e melhor consistência em baixa luz.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    before = len(task_knowledge(engine._work_session, task.id))

    response = engine.respond("isto prova que é melhor que o Samsung?")
    telemetry = engine.get_last_turn_telemetry()

    assert telemetry["selected_path"] == "TASK_INTELLIGENCE"
    assert telemetry["llm_calls"] == 0
    assert len(task_knowledge(engine._work_session, task.id)) == before
    assert response.startswith("Não.")
    assert "não provam" in response
    assert "Samsung Galaxy S24" in response


def test_absent_named_source_does_not_fallback_to_unrelated_source(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung official source",
                    "url": "https://www.samsung.com/source",
                    "domain": "samsung.com",
                    "source_quality": "official",
                    "snippet": "Samsung data.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "Samsung evidence."}
                    ],
                }
            ]
        }
    )
    opened: list[str] = []

    @registry.register("open_url", "Abre URL.", ("desktop:open_url", "confirmation-required"))
    def open_url(url: str) -> str:
        opened.append(url)
        return f"URL aberto: {url}"

    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do Samsung")

    response = engine.respond("abre a fonte da DXOMARK")
    telemetry = engine.get_last_turn_telemetry()

    assert "Não tenho uma fonte da DXOMARK nesta pesquisa." == response
    assert telemetry["content_operation_detected"] is True
    assert telemetry["research_source_open_requested"] is False
    assert telemetry["research_source_url"] == ""
    assert telemetry["llm_calls"] == 0
    assert not opened
    assert not engine.agent.has_pending_confirmation()


def test_source_read_detail_summary_and_raw_excerpt_are_distinct_and_natural(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera review",
                    "url": "https://example.test/iphone-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "The iPhone 15 camera review says the phone performs well in daylight and low light, with stable video and reliable colors.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "A câmara do iPhone tem vídeo estável e melhor consistência em baixa luz."}
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")

    default_read = engine.respond("lê a primeira fonte")
    detailed = engine.respond("mais detalhe")
    summary = engine.respond("resume")
    raw = engine.respond("lê exatamente o excerto")

    assert default_read != detailed
    assert summary != detailed
    assert len(detailed) > len(default_read)
    assert len(summary) <= len(detailed)
    assert "camera_quality" not in default_read
    assert "camera quality" not in default_read
    assert " = " not in default_read
    assert "The iPhone 15 camera review says" not in default_read
    assert "vídeo estável" in raw
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_rejected_related_sources_do_not_distort_default_ordinal_order(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung Galaxy S24 Ultra Long Term Camera Review",
                    "url": "https://example.test/s24-ultra-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Ultra review.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24 Ultra", "predicate": "camera_quality", "value": "better", "evidence": "O Ultra tem zoom superior."}
                    ],
                },
                {
                    "title": "Samsung Galaxy S24 Camera Review",
                    "url": "https://example.test/s24-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Base model review.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "O modelo base tem boa câmara."}
                    ],
                },
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do Samsung")

    response = engine.respond("lê a primeira fonte")
    telemetry = engine.get_last_turn_telemetry()

    assert "Samsung Galaxy S24 Camera Review" in response
    assert "Ultra Long Term" not in response
    assert telemetry["content_target_resolution"] == "ordinal"
    assert telemetry["llm_calls"] == 0


def test_show_sources_distinguishes_analysed_from_used_sources(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Samsung Galaxy S24": [
                {
                    "title": "Samsung Galaxy S24 Ultra Long Term Camera Review",
                    "url": "https://example.test/s24-ultra-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Ultra review.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24 Ultra", "predicate": "camera_quality", "value": "better", "evidence": "O Ultra tem zoom superior."}
                    ],
                },
                {
                    "title": "Samsung Galaxy S24 Camera Review",
                    "url": "https://example.test/s24-camera",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Base review.",
                    "claims": [
                        {"subject": "Samsung Galaxy S24", "predicate": "camera_quality", "value": "good", "evidence": "O modelo base tem boa câmara."}
                    ],
                },
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do Samsung")

    response = engine.respond("mostra as fontes")

    assert "Analisei 2 resultados e usei 1 fonte relevante." in response
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_research_strings_preserve_ptpt_characters_in_python_state(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "Fonte com acentos – opções",
                    "url": "https://example.test/acentos",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Câmara, opções, 899 € – teste.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Câmara, opções, 899 € – teste."}
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    response = engine.respond("lê exatamente o excerto")

    assert any("Câmara, opções, 899 € – teste." in entry.source.original_text for entry in entries)
    assert "Câmara, opções, 899 € – teste." in response


def test_tiktok_without_substantive_evidence_is_not_accepted(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera test on TikTok",
                    "url": "https://www.tiktok.com/@creator/video/1",
                    "domain": "tiktok.com",
                    "source_quality": "web",
                    "snippet": "We're having trouble playing this video. Speech not recognized. 1200 likes.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "We're having trouble playing this video. Speech not recognized. 1200 likes."}
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")

    response = engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    sources = engine.get_last_turn_telemetry()["research_result_count"]

    assert "não consegui transformá-los em conhecimento seguro" in response or "descartei" in response.lower()
    assert not any(entry.source.reference.startswith("https://www.tiktok.com") for entry in entries)
    assert sources == 0
    assert engine.get_last_turn_telemetry()["llm_calls"] == 0


def test_youtube_with_useful_transcript_is_accepted_but_not_strong(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 low light video test",
                    "url": "https://www.youtube.com/watch?v=abc",
                    "domain": "youtube.com",
                    "source_quality": "web",
                    "snippet": "iPhone 15 4K 30 fps low light camera test.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "O vídeo testa o iPhone 15 em 4K a 30 fps, com pouca luz, e mostra vídeo estável."}
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")

    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    entries = [entry for entry in task_knowledge(engine._work_session, task.id) if entry.source.source_type == "web"]

    assert len(entries) == 1
    assert entries[0].metadata["source_kind"] == "video"
    assert entries[0].metadata["source_strength_for_claim"] == "useful"
    assert 0.5 < entries[0].confidence <= 0.7


def test_porto_city_break_canonicalizes_to_origin_and_is_rejected(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "European city break": [
                {
                    "title": "Portugal | Porto City Break",
                    "url": "https://example.test/porto-city-break",
                    "domain": "example.test",
                    "source_quality": "travel_guide",
                    "snippet": "Porto City Break ideas for travellers starting in Porto.",
                    "claims": [
                        {
                            "subject": "Porto City Break",
                            "predicate": "fit_for_trip",
                            "value": "city break package",
                            "evidence": "Porto City Break ideas for travellers starting in Porto.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Portugal | Porto City Break", "destination", "Porto City Break", 0.9),
            SemanticCandidate("Porto City Break", "destination", "Porto City Break", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("quero férias de 7 dias a partir do Porto, até 1200 euros, com voos curtos")

    response = engine.respond("pesquisa destinos que façam sentido")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)

    assert "ainda não consegui extrair destinos fiáveis" in response
    assert not any(entry.predicate == "destination_candidate" for entry in entries)
    assert not any(entry.subject in {"Portugal", "Porto City Break", "Porto"} for entry in entries if entry.kind == "candidate")
    telemetry = engine.get_last_turn_telemetry()
    assert telemetry["semantic_extraction_calls"] >= 1
    assert telemetry["semantic_candidates_before_validation"] >= 2
    assert telemetry["semantic_candidates_after_validation"] == 0
    assert "invalid_destination_candidate" in telemetry["semantic_extraction_failure_reason"]


def test_stockholm_relevant_source_survives_zero_semantic_candidates(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Sweden - Top 10 things to do in Stockholm",
                    "url": "https://visitsweden.com/stockholm-top-10",
                    "domain": "visitsweden.com",
                    "source_quality": "official",
                    "snippet": "Top 10 things to do in Stockholm, including museums and attractions.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "Top 10 things to do in Stockholm",
                            "evidence": "Top 10 things to do in Stockholm, including museums and attractions.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = StubSemanticExtractor([])
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa o que vale a pena fazer")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    sources = engine._work_session.knowledge.get("research_sources", [])
    telemetry = engine.get_last_turn_telemetry()

    assert "fontes relevantes sobre Estocolmo" in response
    assert not any(entry.predicate == "activities" for entry in entries)
    assert any(entry.kind == "source" and entry.source.reference == "https://visitsweden.com/stockholm-top-10" for entry in entries)
    assert any(source["status"] == "accepted" and source["relevance"] == "relevant" for source in sources)
    assert telemetry["semantic_extraction_calls"] >= 1
    assert telemetry["semantic_extraction_status"] == "empty_result"
    assert telemetry["semantic_candidates_before_validation"] == 0
    assert telemetry["semantic_candidates_after_validation"] == 0


def test_stockholm_extraction_failure_keeps_source_relevant_not_irrelevant(tmp_path: Path) -> None:
    class FailingSemanticExtractor:
        requests = []

        def extract(self, request):
            self.requests.append(request)
            raise RuntimeError("extractor unavailable")

    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Nobel Prize Museum - Visit Stockholm",
                    "url": "https://visitstockholm.com/nobel-prize-museum",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Nobel Prize Museum is an attraction in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Nobel Prize Museum is an attraction in Stockholm.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = FailingSemanticExtractor()
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa o que vale a pena fazer")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    sources = engine._work_session.knowledge.get("research_sources", [])
    telemetry = engine.get_last_turn_telemetry()

    assert "fontes relevantes sobre Estocolmo" in response
    assert not any(entry.predicate == "activities" for entry in entries)
    assert any(entry.kind == "source" for entry in entries)
    assert not any(source.get("relevance") == "irrelevant" for source in sources)
    assert any(source["status"] == "accepted" and source["relevance"] == "relevant" for source in sources)
    assert telemetry["semantic_extraction_calls"] >= 1
    assert telemetry["semantic_extraction_status"] == "call_failed"
    assert telemetry["semantic_extraction_failure_reason"] == "RuntimeError"


def test_semantic_candidate_parser_accepts_valid_json_and_fenced_json() -> None:
    payload = '{"candidates":[{"canonical_name":"Vasa Museum","entity_type":"place","evidence_quote_or_fragment":"Vasa Museum","confidence":0.9}]}'
    plain = _parse_semantic_candidates_with_status(payload)
    fenced = _parse_semantic_candidates_with_status(f"```json\n{payload}\n```")

    assert plain.status == "success"
    assert fenced.status == "success"
    assert plain.candidates[0].canonical_name == "Vasa Museum"
    assert fenced.candidates[0].canonical_name == "Vasa Museum"


def test_semantic_candidate_parser_rejects_invalid_prose_safely() -> None:
    result = _parse_semantic_candidates_with_status("Aqui estão alguns sítios: Vasa Museum e Gamla Stan.")

    assert result.status == "invalid_json"
    assert result.candidates == []


def test_capable_local_llm_is_used_for_semantic_extraction_in_engine_path(tmp_path: Path) -> None:
    class WrappedLocalLLM:
        system_prompt = ""
        supports_local_semantic_extraction = True
        provider_name = "ollama"

        def __init__(self):
            self.chat_call_count = 0
            self.chat_call_sources = []
            self.requests = []

        def chat(self, user_message, **kwargs):
            self.chat_call_count += 1
            self.chat_call_sources.append(kwargs.get("source") or "")
            self.requests.append((user_message, kwargs))
            return json.dumps(
                {
                    "candidates": [
                        {
                            "canonical_name": "Vasa Museum",
                            "entity_type": "place",
                            "evidence_quote_or_fragment": "Vasa Museum",
                            "confidence": 0.9,
                            "aliases": ["Vasa Museet"],
                        }
                    ]
                }
            )

    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                        }
                    ],
                }
            ]
        }
    )
    llm = WrappedLocalLLM()
    engine = _engine(tmp_path, registry, llm=llm)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    task = engine._work_session.active_task
    telemetry = engine.get_last_turn_telemetry()

    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)
    assert any(entry.predicate == "activities" and entry.value == "Vasa Museum" for entry in entries)
    assert any(entry.predicate == "activities" and entry.kind != "source" for entry in entries)
    assert llm.requests
    assert llm.requests[0][1]["response_format"] == "json"
    assert llm.requests[0][1]["source"] == "SEMANTIC_EXTRACTOR_LOCAL"
    assert telemetry["semantic_extraction_calls"] == 1
    assert telemetry["semantic_extraction_status"] == "success"
    assert telemetry["semantic_candidates_before_validation"] == 1
    assert telemetry["semantic_candidates_after_validation"] == 1


def test_semantic_extraction_uses_internal_evidence_when_display_snippet_is_empty(tmp_path: Path) -> None:
    class WrappedLocalLLM:
        system_prompt = ""
        supports_local_semantic_extraction = True
        provider_name = "ollama"

        def __init__(self):
            self.chat_call_count = 0
            self.chat_call_sources = []
            self.prompts = []

        def chat(self, user_message, **kwargs):
            self.chat_call_count += 1
            self.chat_call_sources.append(kwargs.get("source") or "")
            self.prompts.append(user_message)
            return json.dumps(
                {
                    "candidates": [
                        {
                            "canonical_name": "Vasa Museum",
                            "entity_type": "attraction",
                            "evidence_quote_or_fragment": "Vasa Museum",
                            "confidence": 0.9,
                        }
                    ]
                }
            )

    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "",
                    "content": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                        }
                    ],
                }
            ]
        }
    )
    llm = WrappedLocalLLM()
    engine = _engine(tmp_path, registry, llm=llm)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert llm.prompts
    assert "Vasa Museum" in llm.prompts[0]
    assert telemetry["semantic_extraction_status"] == "success"


def test_app_runtime_factory_wires_local_semantic_extractor_into_engine_path(tmp_path: Path) -> None:
    class FakeOllamaProvider:
        name = "ollama"

        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            return ModelResponse(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "canonical_name": "Vasa Museum",
                                "entity_type": "place",
                                "evidence_quote_or_fragment": "Vasa Museum",
                                "confidence": 0.9,
                            }
                        ]
                    }
                ),
                provider="ollama",
                model="llama3.1:8b",
                input_tokens=10,
                output_tokens=12,
                latency_ms=8,
            )

    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                        }
                    ],
                }
            ]
        }
    )
    provider = FakeOllamaProvider()
    semantic_extractor = build_local_semantic_extractor(provider, "")
    engine = _engine(tmp_path, registry, semantic_extractor=semantic_extractor)

    assert engine.semantic_extractor is semantic_extractor

    engine.respond("vou a Estocolmo de 10 a 15 de outubro")
    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert provider.calls
    assert provider.calls[0][1]["response_format"] == "json"
    assert telemetry["semantic_extraction_calls"] == 1
    assert telemetry["semantic_extraction_status"] == "success"
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    assert any(entry.predicate == "activities" and entry.value == "Vasa Museum" for entry in entries)
    assert not all(entry.kind == "source" for entry in entries)


def test_non_research_turn_does_not_invoke_semantic_extraction(tmp_path: Path) -> None:
    extractor = RaisingSemanticExtractor()
    engine = _engine(tmp_path, semantic_extractor=extractor)

    response = engine.respond("vou a Estocolmo de 10 a 15 de outubro")
    telemetry = engine.get_last_turn_telemetry()

    assert "Estocolmo" in response
    assert extractor.requests == []
    assert telemetry["semantic_extraction_calls"] == 0
    assert telemetry["llm_calls"] == 0


def test_local_semantic_extractor_passes_bounded_timeout_to_provider() -> None:
    class FakeOllamaProvider:
        name = "ollama"

        def __init__(self):
            self.calls = []

        def chat(self, messages, **kwargs):
            self.calls.append((messages, kwargs))
            raise TimeoutError("semantic extraction timeout")

    provider = FakeOllamaProvider()
    extractor = build_local_semantic_extractor(provider, "")
    result = extractor.extract(
        type(
            "Request",
            (),
            {
                "task_type": "plan_trip",
                "extraction_goal": "stockholm_places",
                "known_context": {},
                "excluded_entities": [],
                "evidence": "Visit Stockholm recomenda o Vasa Museum.",
                "source_title": "Visit Stockholm",
            },
        )()
    )

    assert provider.calls
    assert provider.calls[0][1]["timeout_seconds"] == 45
    assert result.status == "call_failed"
    assert result.failure_reason == "TimeoutError"


def test_semantic_extraction_timeout_keeps_research_turn_finalizable(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry, semantic_extractor=RaisingSemanticExtractor(TimeoutError("timeout")))
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert response.strip()
    assert telemetry["selected_path"] == "RESEARCH_REQUEST"
    assert telemetry["semantic_extraction_calls"] == 1
    assert telemetry["semantic_extraction_status"] == "call_failed"
    assert telemetry["semantic_extraction_failure_reason"] == "TimeoutError"


def test_invalid_semantic_extraction_json_keeps_research_turn_finalizable(tmp_path: Path) -> None:
    class InvalidJsonLLM:
        system_prompt = ""
        supports_local_semantic_extraction = True
        provider_name = "ollama"

        def __init__(self):
            self.chat_call_count = 0
            self.chat_call_sources = []

        def chat(self, user_message, **kwargs):
            self.chat_call_count += 1
            self.chat_call_sources.append(kwargs.get("source") or "")
            return "Aqui estão alguns sítios: Vasa Museum e Gamla Stan."

    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry, llm=InvalidJsonLLM())
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    response = engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert response.strip()
    assert telemetry["selected_path"] == "RESEARCH_REQUEST"
    assert telemetry["semantic_extraction_calls"] == 1
    assert telemetry["semantic_extraction_status"] == "invalid_json"
    assert telemetry["semantic_candidates_after_validation"] == 0


def test_activity_sources_are_semantically_extracted_in_one_batch(tmp_path: Path) -> None:
    results = [
        {
            "title": f"Visit Stockholm attractions {index}",
            "url": f"https://www.visitstockholm.com/source-{index}",
            "domain": "visitstockholm.com",
            "source_quality": "official",
            "snippet": "Things to do in Stockholm attractions museums.",
            "claims": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "What to do in Stockholm? The 15 Best Attractions",
                    "evidence": "Visit Stockholm recomenda o Vasa Museum, ABBA Museum, Skansen, Gröna Lund e Djurgården.",
                }
            ],
        }
        for index in range(10)
    ]
    registry, _queries = _registry_with_results({"Estocolmo official tourism things to do attractions": results})
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Djurgården", "place", "Djurgården", 0.9),
            SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9),
            SemanticCandidate("ABBA Museum", "attraction", "ABBA Museum", 0.9),
            SemanticCandidate("Skansen", "attraction", "Skansen", 0.9),
            SemanticCandidate("Gröna Lund", "attraction", "Gröna Lund", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert len(extractor.requests) == 1
    assert telemetry["semantic_extraction_calls"] == 1
    assert telemetry["semantic_extraction_batches"] == 1
    assert telemetry["semantic_candidates_after_validation"] == 5


def test_large_semantic_evidence_is_split_into_bounded_batches(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistant.task_research.SEMANTIC_BATCH_CHAR_CAP", 700)
    monkeypatch.setattr("assistant.task_research.SEMANTIC_BATCH_SOURCE_CAP", 4)
    results = [
        {
            "title": f"Visit Stockholm long source {index}",
            "url": f"https://www.visitstockholm.com/long-{index}",
            "domain": "visitstockholm.com",
            "source_quality": "official",
            "snippet": "Things to do in Stockholm attractions museums.",
            "claims": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "What to do in Stockholm? The 15 Best Attractions",
                    "evidence": f"Source marker {index}. Visit Stockholm recomenda o Vasa Museum. " + ("informação útil " * 45),
                }
            ],
        }
        for index in range(9)
    ]
    registry, _queries = _registry_with_results({"Estocolmo official tourism things to do attractions": results})
    extractor = DynamicSemanticExtractor(
        {"Vasa Museum": [SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9)]}
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert 2 <= len(extractor.requests) <= 3
    assert telemetry["semantic_extraction_batches"] == len(extractor.requests)
    assert telemetry["semantic_extraction_calls"] == len(extractor.requests)


def test_price_evidence_does_not_invoke_stockholm_place_extraction(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo travel cost restaurants attractions prices": [
                {
                    "title": "Stockholm prices",
                    "url": "https://example.test/stockholm-prices",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Prices in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "price",
                            "value": "Vasa Museum and restaurants can affect the trip budget.",
                            "evidence": "Vasa Museum, Nytorget 6 and Meatballs for the People appear in price examples.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = RaisingSemanticExtractor()
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o custo")
    entries = task_knowledge(engine._work_session, engine._work_session.active_task.id)
    telemetry = engine.get_last_turn_telemetry()

    assert extractor.requests == []
    assert telemetry["semantic_extraction_calls"] == 0
    assert not any(entry.predicate == "activities" and entry.value in {"Vasa Museum", "Nytorget 6"} for entry in entries)


def test_transport_evidence_does_not_invoke_stockholm_place_extraction(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official public transport airport city pass": [
                {
                    "title": "SL transport",
                    "url": "https://sl.se/stockholm",
                    "domain": "sl.se",
                    "source_quality": "official",
                    "snippet": "Public transport in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "transport",
                            "value": "SL metro bus tram commuter rail cover local transport.",
                            "evidence": "SL public transport covers metro, bus, tram and commuter rail.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = RaisingSemanticExtractor()
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa transportes")
    telemetry = engine.get_last_turn_telemetry()

    assert extractor.requests == []
    assert telemetry["semantic_extraction_calls"] == 0


def test_generic_web_evidence_alone_does_not_trigger_stockholm_place_extraction(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Generic Stockholm overview",
                    "url": "https://example.test/overview",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "General overview.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "web_evidence",
                            "value": "Stockholm is a city with islands and neighbourhoods.",
                            "evidence": "Stockholm is a city with islands and neighbourhoods.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = RaisingSemanticExtractor()
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert extractor.requests == []
    assert telemetry["semantic_extraction_calls"] == 0


def test_semantic_batching_preserves_source_provenance(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                        }
                    ],
                },
                {
                    "title": "Visit Sweden",
                    "url": "https://visitsweden.com/skansen",
                    "domain": "visitsweden.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? Cultural attractions",
                            "evidence": "Visit Sweden destaca Skansen para uma visita cultural.",
                        }
                    ],
                },
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Skansen", "attraction", "Skansen", 0.9, source_ids=("https://visitsweden.com/skansen",)),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    entries = task_knowledge(engine._work_session, engine._work_session.active_task.id)
    skansen = next(entry for entry in entries if entry.predicate == "activities" and entry.value == "Skansen")

    assert skansen.source.reference == "https://visitsweden.com/skansen"


def test_semantic_batching_early_stops_after_sufficient_places(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("assistant.task_research.SEMANTIC_BATCH_SOURCE_CAP", 2)
    results = [
        {
            "title": f"Visit Stockholm source {index}",
            "url": f"https://www.visitstockholm.com/early-{index}",
            "domain": "visitstockholm.com",
            "source_quality": "official",
            "snippet": "Things to do in Stockholm attractions museums.",
            "claims": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "What to do in Stockholm? The 15 Best Attractions",
                    "evidence": "Visit Stockholm recomenda Djurgården, Vasa Museum, ABBA Museum, Skansen, Gröna Lund e Nationalmuseum.",
                }
            ],
        }
        for index in range(8)
    ]
    registry, _queries = _registry_with_results({"Estocolmo official tourism things to do attractions": results})
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Djurgården", "place", "Djurgården", 0.9),
            SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9),
            SemanticCandidate("ABBA Museum", "attraction", "ABBA Museum", 0.9),
            SemanticCandidate("Skansen", "attraction", "Skansen", 0.9),
            SemanticCandidate("Gröna Lund", "attraction", "Gröna Lund", 0.9),
            SemanticCandidate("Nationalmuseum", "attraction", "Nationalmuseum", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert len(extractor.requests) == 1
    assert telemetry["semantic_extraction_calls"] == 1
    assert telemetry["semantic_candidates_after_validation"] == 6


def test_weak_social_transport_snippet_does_not_enter_task_knowledge(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official public transport airport city pass": [
                {
                    "title": "Facebook comment",
                    "url": "https://facebook.com/comment",
                    "domain": "facebook.com",
                    "source_quality": "social",
                    "snippet": "cristinamadriaza Hermosa cada estación...",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "transport",
                            "value": "cristinamadriaza Hermosa cada estación...",
                            "evidence": "cristinamadriaza Hermosa cada estación...",
                        }
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry, semantic_extractor=RaisingSemanticExtractor())
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa transportes")
    entries = task_knowledge(engine._work_session, engine._work_session.active_task.id)

    assert not any("cristinamadriaza" in entry.value for entry in entries)


def test_stockholm_semantic_quality_fixture_still_yields_grounded_places(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/things-to-do",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Djurgården, Vasa Museum, ABBA Museum, Skansen and Gröna Lund are named Stockholm attractions.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Djurgården", "place", "Djurgården", 0.9),
            SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9),
            SemanticCandidate("ABBA Museum", "attraction", "ABBA Museum", 0.9),
            SemanticCandidate("Skansen", "attraction", "Skansen", 0.9),
            SemanticCandidate("Gröna Lund", "attraction", "Gröna Lund", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    entries = task_knowledge(engine._work_session, engine._work_session.active_task.id)
    values = {entry.value for entry in entries if entry.predicate == "activities"}

    assert {"Djurgården", "Vasa Museum", "ABBA Museum", "Skansen", "Gröna Lund"}.issubset(values)
    assert len(extractor.requests) == 1


def test_successful_semantic_batch_clears_stale_failure_reason(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Generic overview",
                    "url": "https://example.test/overview",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "General overview.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "web_evidence",
                            "value": "Stockholm has islands and general culture.",
                            "evidence": "Stockholm has islands and general culture.",
                        }
                    ],
                },
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "Visit Stockholm recomenda o Vasa Museum como atração principal.",
                        }
                    ],
                },
            ]
        }
    )
    extractor = StubSemanticExtractor([SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9)])
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()

    assert telemetry["semantic_extraction_status"] == "success"
    assert telemetry["semantic_extraction_failure_reason"] == ""


def test_compact_semantic_payload_excludes_price_transport_and_noise(tmp_path: Path) -> None:
    class CapturingExtractor(StubSemanticExtractor):
        def extract(self, request):
            self.requests.append(request)
            return [SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9)]

    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Stockholm",
                    "url": "https://www.visitstockholm.com/vasa",
                    "domain": "visitstockholm.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? The 15 Best Attractions",
                            "evidence": "navigation footer " * 60 + "Visit Stockholm recomenda o Vasa Museum como atração principal. " + "cookie banner " * 60,
                        },
                        {
                            "subject": "Estocolmo",
                            "predicate": "price",
                            "value": "Vasa Museum costs money.",
                            "evidence": "Nytorget 6 and Meatballs for the People appear in price examples.",
                        },
                        {
                            "subject": "Estocolmo",
                            "predicate": "transport",
                            "value": "SL metro bus tram commuter rail.",
                            "evidence": "SL public transport covers metro, bus, tram and commuter rail.",
                        },
                    ],
                }
            ]
        }
    )
    extractor = CapturingExtractor([])
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()
    payload = extractor.requests[0].evidence

    assert "Vasa Museum" in payload
    assert "Nytorget" not in payload
    assert "Meatballs for the People" not in payload
    assert "commuter rail" not in payload
    assert "cookie banner" not in payload
    assert telemetry["semantic_extraction_input_chars"] == len(payload)
    assert telemetry["semantic_extraction_input_chars"] < 1200


def test_strong_sources_are_selected_before_weak_sources_for_semantic_batch(tmp_path: Path) -> None:
    results = [
        {
            "title": "Facebook attractions post",
            "url": "https://facebook.com/stockholm-attractions",
            "domain": "facebook.com",
            "source_quality": "social",
            "snippet": "Things to do in Stockholm.",
            "claims": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "What to do in Stockholm? The 15 Best Attractions",
                    "evidence": "Random social post mentions a vague fun place.",
                }
            ],
        }
    ]
    results.extend(
        {
            "title": f"Visit Stockholm strong source {index}",
            "url": f"https://www.visitstockholm.com/strong-{index}",
            "domain": "visitstockholm.com",
            "source_quality": "official",
            "snippet": "Things to do in Stockholm attractions museums.",
            "claims": [
                {
                    "subject": "Estocolmo",
                    "predicate": "activities",
                    "value": "What to do in Stockholm? The 15 Best Attractions",
                    "evidence": "Visit Stockholm recomenda Djurgården, Vasa Museum, ABBA Museum, Skansen, Gröna Lund e Gamla Stan.",
                }
            ],
        }
        for index in range(7)
    )
    registry, _queries = _registry_with_results({"Estocolmo official tourism things to do attractions": results})
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate("Djurgården", "place", "Djurgården", 0.9),
            SemanticCandidate("Vasa Museum", "attraction", "Vasa Museum", 0.9),
            SemanticCandidate("ABBA Museum", "attraction", "ABBA Museum", 0.9),
            SemanticCandidate("Skansen", "attraction", "Skansen", 0.9),
            SemanticCandidate("Gröna Lund", "attraction", "Gröna Lund", 0.9),
            SemanticCandidate("Gamla Stan", "place", "Gamla Stan", 0.9),
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    telemetry = engine.get_last_turn_telemetry()
    payload = extractor.requests[0].evidence

    assert "facebook.com" not in payload
    assert telemetry["semantic_extraction_sources_used"] == 5
    assert telemetry["semantic_extraction_calls"] == 1


def test_original_evidence_remains_available_for_grounding_after_compaction(tmp_path: Path) -> None:
    long_prefix = "navigation footer unrelated text. " * 80
    hidden_fragment = "Skansen"
    registry, _queries = _registry_with_results(
        {
            "Estocolmo official tourism things to do attractions": [
                {
                    "title": "Visit Sweden",
                    "url": "https://visitsweden.com/skansen",
                    "domain": "visitsweden.com",
                    "source_quality": "official",
                    "snippet": "Things to do in Stockholm.",
                    "claims": [
                        {
                            "subject": "Estocolmo",
                            "predicate": "activities",
                            "value": "What to do in Stockholm? Cultural attractions",
                            "evidence": long_prefix + "Visit Sweden destaca Skansen para uma visita cultural.",
                        }
                    ],
                }
            ]
        }
    )
    extractor = StubSemanticExtractor(
        [
            SemanticCandidate(
                "Skansen",
                "attraction",
                hidden_fragment,
                0.9,
                source_ids=("https://visitsweden.com/skansen",),
            )
        ]
    )
    engine = _engine(tmp_path, registry, semantic_extractor=extractor)
    engine.respond("vou a Estocolmo de 10 a 15 de outubro")

    engine.respond("pesquisa o que vale a pena fazer")
    entries = task_knowledge(engine._work_session, engine._work_session.active_task.id)

    skansen_entry = next(entry for entry in entries if entry.predicate == "activities" and entry.value == "Skansen")
    assert "Visit Sweden destaca Skansen" in skansen_entry.source.original_text


def test_dpreview_mixed_model_page_accepts_exact_iphone_claim_and_rejects_pro_claim(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "Best camera settings for iPhone 15 and iPhone 15 Pro",
                    "url": "https://www.dpreview.com/articles/iphone-15-camera-settings",
                    "domain": "dpreview.com",
                    "source_quality": "web",
                    "snippet": "The article discusses iPhone 15 and iPhone 15 Pro separately.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "No iPhone 15 base, a câmara principal tem bom detalhe em pouca luz."},
                        {"subject": "iPhone 15 Pro", "predicate": "camera_quality", "value": "better", "evidence": "No iPhone 15 Pro, o teleobjetivo oferece mais alcance."},
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")

    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task
    entries = task_knowledge(engine._work_session, task.id)
    sources = engine._work_session.knowledge.get("research_sources", [])

    assert any(entry.subject == "iPhone 15" and entry.metadata.get("applicability") == "exact" for entry in entries if entry.source.source_type == "web")
    assert not any(entry.subject == "iPhone 15 Pro" and entry.status == "active" for entry in entries)
    assert any(source["status"] == "accepted" and source["source_strength_for_claim"] == "strong" for source in sources)


def test_stronger_exact_source_precedes_social_source_in_visible_order(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 low light TikTok",
                    "url": "https://www.tiktok.com/@creator/video/2",
                    "domain": "tiktok.com",
                    "source_quality": "web",
                    "snippet": "Creator says the iPhone 15 video looks good in low light.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "O vídeo do criador diz que a câmara parece boa em pouca luz."}
                    ],
                },
                {
                    "title": "iPhone 15 camera lab test",
                    "url": "https://www.dxomark.com/iphone-15-camera-test/",
                    "domain": "dxomark.com",
                    "source_quality": "web",
                    "snippet": "Lab camera test for iPhone 15.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "better", "evidence": "O teste técnico mediu vídeo estável, boa exposição e bom detalhe em pouca luz."}
                    ],
                },
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")

    response = engine.respond("lê a primeira fonte")

    assert "dxomark.com" in response
    assert "TikTok" not in response


def test_natural_source_presentations_hide_internal_labels_and_metadata(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 camera review",
                    "url": "https://example.test/iphone-camera-natural",
                    "domain": "example.test",
                    "source_quality": "independent_review",
                    "snippet": "Description 340K subscribers 1200 likes",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "Description\n340K subscribers\n1200 likes\nO teste mostra bom vídeo em pouca luz no iPhone 15."}
                    ],
                }
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")

    brief = engine.respond("lê a primeira fonte")
    detail = engine.respond("mais detalhe")
    summary = engine.respond("resume")

    combined = "\n".join([brief, detail, summary])
    for forbidden in ("camera_quality", "family_related", "accepted", "rejected", "= good", "= mixed", "subscribers", "likes", "Description"):
        assert forbidden not in combined
    assert "bom desempenho" in combined or "bom vídeo" in combined
    assert len(detail) > len(brief)
    assert len(summary) <= len(detail)


def test_task_surface_synthesizes_good_and_mixed_without_internal_values(tmp_path: Path) -> None:
    registry, _queries = _registry_with_results(
        {
            "iPhone 15": [
                {
                    "title": "iPhone 15 review positive",
                    "url": "https://www.dxomark.com/iphone-15-positive",
                    "domain": "dxomark.com",
                    "source_quality": "independent_test",
                    "snippet": "Positive camera test.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "good", "evidence": "A análise técnica mostra boa câmara em pouca luz."}
                    ],
                },
                {
                    "title": "iPhone 15 review mixed",
                    "url": "https://www.dpreview.com/iphone-15-mixed",
                    "domain": "dpreview.com",
                    "source_quality": "independent_review",
                    "snippet": "Mixed camera review.",
                    "claims": [
                        {"subject": "iPhone 15", "predicate": "camera_quality", "value": "mixed", "evidence": "A análise encontra boa câmara, mas resultados variáveis em alguns cenários."}
                    ],
                },
            ]
        }
    )
    engine = _engine(tmp_path, registry)
    engine.respond("vamos comparar Samsung Galaxy S24 e iPhone 15")
    engine.respond("pesquisa mais sobre a câmara do iPhone")
    task = engine._work_session.active_task

    view = cognitive_task_view_model(engine._work_session, task)
    text = str(view)

    assert "camera_quality" not in text
    assert "câmara: good" not in text.lower()
    assert "câmara: mixed" not in text.lower()
    assert "resultados positivos, mas com alguma incerteza" in text
