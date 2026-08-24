from __future__ import annotations

from pathlib import Path
import json

from assistant.task_intelligence import (
    answer_task_query,
    cognitive_task_view_model,
    infer_task_stage,
    task_knowledge,
    task_reasoning,
    update_task_knowledge,
)
from assistant.task_intelligence import _upsert_entry
from tests.test_automatic_routing_regressions import make_engine


def _options(engine, task):
    return {
        obj.title: obj
        for obj in engine._work_session.objects.values()
        if obj.kind == "option" and obj.task_id == task.id
    }


def test_task_knowledge_adds_options_with_user_provenance(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    task = engine._work_session.active_task
    assert task is not None

    entries = task_knowledge(engine._work_session, task.id)
    option_entries = [entry for entry in entries if entry.kind == "option"]

    assert {entry.subject for entry in option_entries} == {"Samsung", "Apple"}
    assert all(entry.source.source_type == "user" for entry in option_entries)
    assert all("Samsung" in entry.source.original_text or "Apple" in entry.source.original_text for entry in option_entries)
    assert ollama.calls == []


def test_mixed_sentence_extracts_multiple_entity_facts_and_preferences(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("não quero gastar mais de 850 euros")
    engine.respond("para mim a câmara é mais importante do que a bateria")
    engine.respond("o Samsung custa 799 e o Apple 899")
    engine.respond("o Samsung tem melhor bateria, mas acho que o Apple tem melhor câmara")
    task = engine._work_session.active_task
    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)

    assert _options(engine, task)["Samsung"].task_id == task.id
    assert any(entry.kind == "constraint" and entry.predicate == "budget_max" and entry.value == "850" for entry in entries)
    assert any(entry.kind == "preference" and entry.predicate == "camera_priority" and entry.value == "high" for entry in entries)
    assert any(entry.subject == "Samsung" and entry.predicate == "price" and entry.value == "799" for entry in entries)
    assert any(entry.subject == "Apple" and entry.predicate == "price" and entry.value == "899" for entry in entries)
    assert any(entry.subject == "Samsung" and entry.predicate == "battery" for entry in entries)
    assert any(entry.subject == "Apple" and entry.predicate == "camera_quality" and entry.kind == "opinion" for entry in entries)
    assert ollama.calls == []


def test_real_phone_wording_resolves_iphone_and_updates_budget(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("não queria gastar mais de 850€")
    engine.respond("para mim a câmara é mais importante, mas também valorizo bastante a bateria")
    engine.respond("o samsung está a 799 e o iphone a 899")
    engine.respond("acho que o iphone tem melhor câmara mas o samsung aguenta mais tempo")
    first = engine.respond("qual parece melhor neste momento?")
    engine.respond("afinal consigo ir até aos 950€ e já não dou tanta importância à bateria")
    second = engine.respond("e agora?")
    task = engine._work_session.active_task
    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)

    assert any(entry.subject == "Samsung" and entry.predicate == "price" and entry.value == "799" for entry in entries)
    assert any(entry.subject == "Apple" and entry.predicate == "price" and entry.value == "899" for entry in entries)
    assert any(entry.subject == "Apple" and entry.predicate == "camera_quality" for entry in entries)
    assert any(entry.subject == "Samsung" and entry.predicate == "battery" for entry in entries)
    assert any(entry.predicate == "budget_max" and entry.value == "850" and entry.status == "superseded" for entry in entries)
    assert any(entry.predicate == "budget_max" and entry.value == "950" and entry.status == "active" for entry in entries)
    assert "Samsung" in first
    assert "Apple" in second
    assert ollama.calls == []


def test_messy_typo_normalizes_without_duplicate_and_ambiguous_reference_clarifies(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("o samasung parece me melhor na bateria mas n sei se vale a pena pelos 799")
    ambiguous = engine.respond("acho que esse é melhor")
    task = engine._work_session.active_task
    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)

    assert {entry.subject for entry in entries if entry.kind == "option"} == {"Samsung", "Apple"}
    assert any(entry.subject == "Samsung" and entry.predicate == "battery" for entry in entries)
    assert "Samsung" in ambiguous and "Apple" in ambiguous
    assert ollama.calls == []


def test_correction_supersedes_previous_active_fact(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("o Samsung custa 850")
    engine.respond("afinal não, o Samsung custa 799")
    task = engine._work_session.active_task
    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)

    active_prices = [entry for entry in entries if entry.subject == "Samsung" and entry.predicate == "price" and entry.status == "active"]
    old_prices = [entry for entry in entries if entry.subject == "Samsung" and entry.predicate == "price" and entry.status == "superseded"]
    assert [entry.value for entry in active_prices] == ["799"]
    assert [entry.value for entry in old_prices] == ["850"]
    assert active_prices[0].replaces == [old_prices[0].id]


def test_conflicting_sources_can_be_reported_when_preserved(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    task = engine._work_session.active_task
    assert task is not None
    update_task_knowledge(engine._work_session, task, "o Samsung custa 850", turn_id="a")
    _upsert_entry(engine._work_session, task.id, "fact", "Samsung", "price", "799", "fonte externa", "web-1", replace_active=False)

    response = answer_task_query(engine._work_session, task, "há informação contraditória?")

    assert "Samsung / price" in response
    assert "850" in response and "799" in response


def test_recommendation_is_provisional_and_recomputed_after_changed_constraints(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("não quero gastar mais de 850 euros")
    engine.respond("para mim a câmara é mais importante do que a bateria")
    engine.respond("o Samsung custa 799 e o Apple 899")
    engine.respond("o Samsung tem melhor bateria, mas acho que o Apple tem melhor câmara")

    first = engine.respond("qual parece melhor neste momento?")
    engine.respond("afinal posso gastar até 950 e já não me importo tanto com bateria")
    second = engine.respond("e agora qual escolherias?")
    reasoning = task_reasoning(engine._work_session, engine._work_session.active_task.id)

    assert "Samsung" in first
    assert "provisória" in first
    assert "Apple" in second
    assert reasoning.current_recommendation == "Apple"
    assert ollama.calls == []


def test_travel_discovery_and_known_destination_stages(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero planear umas férias mas não faço ideia para onde")
    discovery = engine._work_session.active_task
    assert discovery is not None
    assert discovery.task_type == "plan_trip"
    assert infer_task_stage(discovery, task_knowledge(engine._work_session, discovery.id)) == "explore"

    engine.clear_conversation()
    engine.respond("vou a Estocolmo de 10 a 15 de outubro e quero ajuda a planear")
    known = engine._work_session.active_task
    assert known is not None
    entries = task_knowledge(engine._work_session, known.id)

    assert infer_task_stage(known, entries) == "plan"
    assert any(entry.predicate == "destination" and entry.value == "Estocolmo" for entry in entries)
    assert any(entry.predicate == "date_range" and entry.value == "10 a 15 de outubro" for entry in entries)
    assert ollama.calls == []


def test_travel_updates_duration_and_known_query_does_not_pollute_notes(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero planear umas férias mas não faço ideia para onde")
    engine.respond("tenho uns 7 dias e queria gastar no máximo 1200€")
    response = engine.respond("o que já sabes sobre estas férias?")
    task = engine._work_session.active_task
    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)

    assert any(entry.predicate == "duration" and entry.value == "7 dias" for entry in entries)
    assert any(entry.predicate == "budget_max" and entry.value == "1200" for entry in entries)
    assert "7 dias" in response
    assert "1200" in response
    assert "o que já sabes" not in "; ".join(task.collected_facts)
    assert ollama.calls == []


def test_travel_dialogue_reflects_updated_state_and_does_not_repeat_answered_questions(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    start = engine.respond("quero planear umas férias mas não sei para onde")
    duration_budget = engine.respond("tenho 7 dias e até 1200 euros")
    style = engine.respond("quero descansar mas também ter coisas para ver")
    flights = engine.respond("não quero voos muito longos")

    assert "descanso" in start and "equilíbrio" in start
    assert "7 dias" in duration_budget and "1200" in duration_budget
    assert "estilo da viagem" in duration_budget
    assert "descanso" in style and "coisas para ver" in style
    assert "voos" in style.lower()
    assert "descanso, conhecer sítios" not in style
    assert "voos curtos" in flights
    assert "destinos adequados" in flights
    assert "descanso, conhecer sítios" not in flights
    assert ollama.calls == []


def test_known_destination_dialogue_stays_in_planning_stage(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    start = engine.respond("vou a Estocolmo de 10 a 15 de outubro e quero ajuda a planear")
    budget = engine.respond("quero gastar até 1000 euros sem contar os voos")
    organize = engine.respond("o que temos de organizar?")

    assert "Estocolmo" in start
    assert "10 a 15 de outubro" in start
    assert "alojamento" in start and "transportes" in start
    assert "tipo de destino" not in start.lower()
    assert "1000" in budget
    assert "sem contar" in budget
    assert "alojamento" in budget and "zona" in budget
    assert "alojamento" in organize and "itinerário" in organize
    assert ollama.calls == []


def test_travel_destination_suggestion_query_stays_deterministic_when_context_is_missing(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero planear umas férias mas não faço ideia para onde")
    engine.respond("tenho uns 7 dias e queria gastar no máximo 1200€")
    response = engine.respond("já consegues sugerir destinos?")

    assert "Ainda não sugeria destinos" in response
    assert "destino" in response
    assert ollama.calls == []


def test_comparison_dialogue_acknowledges_budget_prices_tradeoff_and_re_evaluation(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    budget = engine.respond("não quero gastar mais de 850 euros")
    preference = engine.respond("para mim a câmara é mais importante que a bateria")
    prices = engine.respond("o Samsung custa 799 e o Apple 899")
    tradeoff = engine.respond("acho que o Apple tem melhor câmara mas o Samsung aguenta mais")
    changed = engine.respond("afinal posso gastar até 950 e já não dou tanta importância à bateria")
    now = engine.respond("e agora?")

    assert "850" in budget
    assert "câmara" in preference and "bateria" in preference
    assert "Samsung 799" in prices and "Apple 899" in prices
    assert "Apple fica 49 € acima" in prices
    assert "tradeoff" in tradeoff.lower()
    assert "Samsung" in tradeoff and "Apple" in tradeoff
    assert "950" in changed
    assert "bateria passa a pesar menos" in changed
    assert "Apple" in now and "provisória" in now
    assert ollama.calls == []


def test_task_query_dialogue_intents_are_distinct_and_grounded(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("não quero gastar mais de 850 euros")
    engine.respond("para mim a câmara é mais importante que a bateria")
    engine.respond("o Samsung custa 799 e o Apple 899")

    known = engine.respond("o que sabemos até agora?")
    important = engine.respond("resume-me só o importante")
    unknowns = engine.respond("o que ainda falta saber?")
    decisions = engine.respond("o que falta decidir?")
    next_step = engine.respond("qual é o próximo passo?")

    assert known != important
    assert "Estamos a comparar" in known
    assert "O mais importante" in important
    assert "Ainda falta" in unknowns
    assert "decidir" in decisions
    assert "próximo passo" in next_step.lower() or "fechar" in next_step.lower()
    assert ollama.calls == []


def test_dialogue_next_question_matches_active_unknowns_and_presentation(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero planear umas férias mas não sei para onde")
    engine.respond("tenho 7 dias e até 1200 euros")
    response = engine.respond("quero descansar mas também ter coisas para ver")
    task = engine._work_session.active_task
    assert task is not None
    view = cognitive_task_view_model(engine._work_session, task)
    unknown_text = str([section for section in view["sections"] if section["role"] == "unknowns"])

    assert "voos" in response.lower()
    assert "ritmo da viagem" not in unknown_text
    assert "orçamento" not in unknown_text
    assert "datas ou duração" not in unknown_text
    assert "descanso, conhecer sítios" not in response
    assert ollama.calls == []


def test_generic_supplier_scenario_uses_same_comparison_architecture(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero escolher entre três propostas de fornecedores: Alpha, Beta e Gamma")
    task = engine._work_session.active_task

    assert task is not None
    assert task.task_type == "generic_comparison"
    assert {obj.title for obj in _options(engine, task).values()} == {"Alpha", "Beta", "Gamma"}
    assert all(obj.task_id == task.id for obj in _options(engine, task).values())
    assert ollama.calls == []


def test_generic_supplier_prices_support_and_recommendation(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero escolher entre três propostas: Alpha, Beta e Gamma")
    engine.respond("a Alpha custa 12000, a Beta 10500 e a Gamma 9800")
    engine.respond("prefiro pagar um pouco mais se tiver melhor suporte")
    engine.respond("a Alpha inclui suporte 24 horas e a Gamma só em horário laboral")
    recommendation = engine.respond("qual parece melhor?")
    task = engine._work_session.active_task
    assert task is not None
    entries = task_knowledge(engine._work_session, task.id)

    assert any(entry.subject == "Alpha" and entry.predicate == "price" and entry.value == "12000" for entry in entries)
    assert any(entry.subject == "Beta" and entry.predicate == "price" and entry.value == "10500" for entry in entries)
    assert any(entry.subject == "Gamma" and entry.predicate == "price" and entry.value == "9800" for entry in entries)
    assert any(entry.predicate == "support_priority" and entry.value == "high" for entry in entries)
    assert any(entry.subject == "Alpha" and entry.predicate == "support_quality" and entry.value == "better" for entry in entries)
    assert "Alpha" in recommendation
    assert ollama.calls == []


def test_generic_supplier_dialogue_uses_same_state_based_planner(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero escolher entre Alpha, Beta e Gamma")
    prices = engine.respond("a Alpha custa 12000, a Beta 10500 e a Gamma 10200")
    preference = engine.respond("prefiro pagar mais se tiver melhor suporte")
    support = engine.respond("a Alpha tem suporte 24h e a Gamma só em horário laboral")
    recommendation = engine.respond("qual parece melhor?")

    assert "Gamma 10200" in prices
    assert "mais barata" in prices or "preços" in prices
    assert "suporte passa a pesar mais" in preference
    assert "Alpha" in support and "Gamma" in support
    assert "Alpha" in recommendation
    assert "provisória" in recommendation
    assert ollama.calls == []


def test_resume_comparison_by_inferred_category_preserves_task_knowledge(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("o samsung está a 799 e o iphone a 899")
    phone_task = engine._work_session.active_task
    assert phone_task is not None
    assert phone_task.parameters.get("category") == "telemóveis"

    engine.respond("quero escolher entre três propostas: Alpha, Beta e Gamma")
    proposals_task = engine._work_session.active_task
    assert proposals_task is not None
    assert proposals_task.id != phone_task.id

    response = engine.respond("volta aos telemóveis")

    assert engine._work_session.active_task.id == phone_task.id
    assert "comparação" in response.lower() or "temos" in response.lower()
    assert any(entry.subject == "Samsung" and entry.predicate == "price" for entry in task_knowledge(engine._work_session, phone_task.id))
    assert ollama.calls == []


def test_task_queries_report_unknowns_sources_and_next_step(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("o Samsung custa 799")
    task = engine._work_session.active_task
    assert task is not None

    known = engine.respond("o que sabemos até agora?")
    unknowns = engine.respond("o que ainda falta saber?")
    source = engine.respond("de onde veio esse preço?")
    next_step = engine.respond("qual é o próximo passo?")

    assert "Samsung" in known and "799" in known
    assert "Apple" in unknowns or "câmara" in unknowns or "autonomia" in unknowns
    assert "o Samsung custa 799" in source
    assert "próximo" in next_step or "falta" in next_step
    assert ollama.calls == []


def test_decision_and_derived_insight_are_generic_relations(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    task = engine._work_session.active_task
    assert task is not None
    _upsert_entry(engine._work_session, task.id, "decision", "decision", "selected", "Samsung", "vou escolher Samsung", "turn-decision")
    _upsert_entry(engine._work_session, task.id, "derived_insight", "Samsung", "within_budget", "true", "derivado do preço e orçamento", "derived-1", confidence=0.82)

    entries = task_knowledge(engine._work_session, task.id)

    assert any(entry.kind == "decision" and entry.predicate == "selected" for entry in entries)
    assert any(entry.kind == "derived_insight" and entry.subject == "Samsung" for entry in entries)


def test_minimize_and_restore_preserve_task_knowledge(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("não quero gastar mais de 850 euros")
    task = engine._work_session.active_task
    assert task is not None
    before = [(entry.subject, entry.predicate, entry.value, entry.status) for entry in task_knowledge(engine._work_session, task.id)]

    engine.respond("minimiza a comparação")
    engine.respond("abre a comparação")
    after = [(entry.subject, entry.predicate, entry.value, entry.status) for entry in task_knowledge(engine._work_session, task.id)]

    assert after == before
    assert ollama.calls == []


def test_no_task_knowledge_leaks_between_tasks(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    phone_task = engine._work_session.active_task
    assert phone_task is not None
    engine.respond("o Samsung custa 799")

    engine.respond("quero escolher entre três propostas de fornecedores: Alpha, Beta e Gamma")
    supplier_task = engine._work_session.active_task
    assert supplier_task is not None
    assert supplier_task.id != phone_task.id

    supplier_entries = task_knowledge(engine._work_session, supplier_task.id)
    assert all(entry.subject not in {"Samsung", "Apple"} for entry in supplier_entries)
    assert ollama.calls == []


def test_cognitive_projection_groups_comparison_without_raw_duplication(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("não quero gastar mais de 850 euros")
    engine.respond("para mim a câmara é mais importante que a bateria")
    engine.respond("o Samsung custa 799 e o Apple 899")
    engine.respond("acho que o Apple tem melhor câmara mas o Samsung aguenta mais")
    task = engine._work_session.active_task
    assert task is not None

    view = cognitive_task_view_model(engine._work_session, task, title="Comparação: telemóveis")
    sections = {section["role"]: section for section in view["sections"]}

    assert view["stage"] == "narrow"
    assert {item["label"] for item in sections["options"]["items"]} == {"Samsung", "Apple"}
    assert any("799" in " ".join(item["details"]) for item in sections["options"]["items"] if item["label"] == "Samsung")
    assert any("899" in " ".join(item["details"]) for item in sections["options"]["items"] if item["label"] == "Apple")
    assert any("850" in item["label"] for item in sections["constraints"]["items"])
    assert "priorities" in sections
    assert "recommendation" in sections
    assert "key_information" not in sections
    assert "Notas" not in str(view)
    assert ollama.calls == []


def test_cognitive_projection_hides_superseded_values_and_surfaces_conflict(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    engine.respond("o Samsung custa 850")
    engine.respond("afinal não, o Samsung custa 799")
    task = engine._work_session.active_task
    assert task is not None
    _upsert_entry(engine._work_session, task.id, "fact", "Apple", "price", "899", "fonte A", "a", replace_active=False)
    _upsert_entry(engine._work_session, task.id, "fact", "Apple", "price", "879", "fonte B", "b", replace_active=False)

    view = cognitive_task_view_model(engine._work_session, task)
    text = str(view)
    sections = {section["role"]: section for section in view["sections"]}

    assert "799" in text
    assert "850" not in text
    assert "conflicts" in sections
    assert "879" in str(sections["conflicts"]) and "899" in str(sections["conflicts"])


def test_cognitive_projection_is_stage_aware_for_travel_discovery_and_plan(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("quero planear umas férias mas não sei para onde")
    engine.respond("tenho 7 dias e até 1200 euros")
    engine.respond("quero descansar mas também ter coisas para ver")
    engine.respond("não quero voos muito longos")
    discovery = engine._work_session.active_task
    assert discovery is not None
    discovery_view = cognitive_task_view_model(engine._work_session, discovery)
    discovery_sections = {section["role"]: section for section in discovery_view["sections"]}

    assert discovery_view["stage"] == "explore"
    assert "constraints" in discovery_sections
    assert "priorities" in discovery_sections
    assert "unknowns" in discovery_sections
    assert "plan_areas" not in discovery_sections
    assert "tipo de destino" in str(discovery_sections["unknowns"])

    engine.clear_conversation()
    engine.respond("vou a Estocolmo de 10 a 15 de outubro e quero ajuda a planear")
    engine.respond("quero gastar até 1000 euros sem contar os voos")
    plan = engine._work_session.active_task
    assert plan is not None
    plan_view = cognitive_task_view_model(engine._work_session, plan)
    plan_sections = {section["role"]: section for section in plan_view["sections"]}

    assert plan_view["stage"] == "plan"
    assert "Estocolmo" in str(plan_sections["overview"])
    assert "10 a 15 de outubro" in str(plan_sections["overview"])
    assert "plan_areas" in plan_sections
    assert {"Alojamento", "Transportes", "Itinerário", "Custos"} <= {item["label"] for item in plan_sections["plan_areas"]["items"]}
    assert ollama.calls == []


def test_task_artifact_event_carries_cognitive_projection_without_new_truth_source(tmp_path: Path) -> None:
    engine, _llm, ollama, _anthropic = make_engine(tmp_path, mode="local")

    engine.respond("vamos comparar Samsung e Apple")
    events = [json.loads(event) for event in engine.consume_ui_events()]
    artifact_events = [event for event in events if event["type"] == "task_artifact_updated"]
    semantic_events = [event for event in events if event["type"] == "semantic_object_upserted"]

    assert artifact_events
    presentation = artifact_events[-1]["presentation"]
    assert presentation["task_type"] == "generic_comparison"
    assert any(section["role"] == "options" for section in presentation["sections"])
    assert {event["title"] for event in semantic_events if event["kind"] == "option"} == {"Samsung", "Apple"}
    assert "presentation" not in engine._work_session.knowledge
    assert ollama.calls == []
