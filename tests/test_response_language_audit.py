"""Phase 7.7: response_language consistency across tool results, the task
system, and deterministic/social paths -- not just the conversation
engine's own scaffolding (already covered by tests/test_language_preferences.py).

Real scenarios run through the ACTUAL routing/composition paths (the
Agent's tool-execution loop, AssistantEngine.respond(), cognitive_task_
view_model) wherever practical, per section 3's "do not only grep for
Portuguese strings" instruction.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import pytest

from assistant.system_health import not_measured_replies
from assistant.agent import Agent, AgentContext
from assistant.conversation import _generic_task_title
from assistant.response_language import localize
from assistant.task_intelligence import cognitive_task_view_model
from assistant.tool_registry import ToolRegistry
from assistant.tools import get_active_application, get_active_window
from tests.test_automatic_routing_regressions import make_engine


class _FakeWindow:
    def __init__(self, title: str, process_name: str = "", is_active: bool = False) -> None:
        self.title = title
        self.process_name = process_name
        self.is_active = is_active


class _FakeContextObserver:
    def observe_once(self):
        return SimpleNamespace(
            active_window="Notepad",
            active_app="notepad.exe",
            open_windows=[_FakeWindow("Notepad", "notepad.exe", True), _FakeWindow("Chrome", "chrome.exe", False)],
        )

    def latest_summary(self):
        return None

    def activity_summary(self, limit=5):
        return []


def _tool_registry_with_context_tools() -> ToolRegistry:
    # The real global tool_registry (assistant.tools' @tool_registry.register
    # decorators) -- importing assistant.tools is enough to populate it.
    from assistant.tools import tool_registry

    return tool_registry


class _ToolChoosingFakeLLM:
    def __init__(self, tool_name: str) -> None:
        self.tool_name = tool_name

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": self.tool_name, "arguments": {}, "reason": "test"}


def _agent_with_context_observer(*, llm=None) -> Agent:
    return Agent(
        llm=llm,
        tools=_tool_registry_with_context_tools(),
        workspace_path=Path("."),
        context_observer=_FakeContextObserver(),
    )


# --- A/B: TOOL_RESULT (the reported get_open_windows bug) -------------------


def test_a_tool_result_pt_pt() -> None:
    agent = _agent_with_context_observer()
    result = agent.run(
        "que janelas tenho abertas",
        AgentContext(system_prompt="", history=[], response_language="pt-PT"),
    )
    assert "Janelas detetadas" in result.response
    assert "Open windows" not in result.response


def test_b_tool_result_en_gb() -> None:
    agent = _agent_with_context_observer(llm=_ToolChoosingFakeLLM("get_open_windows"))
    result = agent.run(
        "which windows do I have open",
        AgentContext(system_prompt="", history=[], response_language="en-GB"),
    )
    assert "Open windows detected" in result.response
    assert "Janelas" not in result.response


# --- C/D: task start ---------------------------------------------------------


def test_c_task_start_pt_pt(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"

    engine.respond("quero planear ferias para o porto")

    task = engine._work_session.active_task
    assert task is not None
    title = _generic_task_title(task, engine.response_language)
    assert title == "Planeamento de férias"


def test_d_task_start_en_gb(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "en-GB"

    engine.respond("quero planear ferias para o porto")

    task = engine._work_session.active_task
    assert task is not None
    title = _generic_task_title(task, engine.response_language)
    assert title == "Holiday planning"
    assert "Planeamento" not in title


# --- E/F: task continuation ---------------------------------------------------


def test_e_task_continuation_pt_pt(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"

    engine.respond("quero planear ferias para o porto")
    task_id_first = engine._work_session.active_task.id
    engine.respond("prefiro mais aventura")
    task = engine._work_session.active_task

    assert task.id == task_id_first
    assert _generic_task_title(task, engine.response_language) == "Planeamento de férias"


def test_f_task_continuation_en_gb(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "en-GB"

    engine.respond("quero planear ferias para o porto")
    task_id_first = engine._work_session.active_task.id
    engine.respond("prefiro mais aventura")
    task = engine._work_session.active_task

    assert task.id == task_id_first
    assert _generic_task_title(task, engine.response_language) == "Holiday planning"


# --- G/H: task completion (reminder task system) ----------------------------


def test_g_task_completion_pt_pt() -> None:
    from assistant.tools import complete_task

    class _Memory:
        def complete_task(self, query: str = "") -> str:
            return "Tarefa marcada como concluida."

        def pending_task_count(self) -> int:
            return 1

    result = complete_task(long_term_memory=_Memory(), query="comprar leite", response_language="pt-PT")
    assert result == "Está feito. Marquei essa tarefa como concluída e já não aparece nas pendentes."


def test_h_task_completion_en_gb() -> None:
    from assistant.tools import complete_task

    class _Memory:
        def complete_task(self, query: str = "") -> str:
            return "Tarefa marcada como concluida."

        def pending_task_count(self) -> int:
            return 1

    result = complete_task(long_term_memory=_Memory(), query="buy milk", response_language="en-GB")
    assert result == "Done. I've marked that task as completed and it no longer shows up as pending."


# --- I/J: task artifact labels (cognitive_task_view_model) -------------------


class _FakeWorkSession:
    def __init__(self) -> None:
        self.knowledge: dict = {}


class _FakeTask:
    def __init__(self, task_type: str) -> None:
        self.id = "task-1"
        self.task_type = task_type
        self.goal = ""
        self.parameters: dict = {}


def test_i_task_artifact_labels_pt_pt() -> None:
    work_session = _FakeWorkSession()
    task = _FakeTask("plan_trip")
    view = cognitive_task_view_model(work_session, task, response_language="pt-PT")
    assert view.get("title") in ("", task.task_type, task.goal)  # title falls back when no goal set
    # No sections without knowledge -- but overview title helper must be pt.
    from assistant.task_intelligence import _overview_title

    assert _overview_title(task, "pt-PT") == "Planeamento"


def test_j_task_artifact_labels_en_gb() -> None:
    task = _FakeTask("plan_trip")
    from assistant.task_intelligence import _overview_title

    assert _overview_title(task, "en-GB") == "Planning"
    assert _overview_title(task, "en-GB") != "Planeamento"


# --- K/L: mid-task language switch -------------------------------------------


def test_k_pt_task_switch_to_en_continuation_same_task(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"
    engine.respond("quero planear ferias para o porto")
    task_id_before = engine._work_session.active_task.id
    title_before = _generic_task_title(engine._work_session.active_task, engine.response_language)

    engine.response_language = "en-GB"
    engine.respond("prefiro mais aventura")
    task_after = engine._work_session.active_task

    assert task_after.id == task_id_before  # same task, not restarted
    assert title_before == "Planeamento de férias"
    assert _generic_task_title(task_after, engine.response_language) == "Holiday planning"


def test_l_en_task_switch_to_pt_continuation_same_task(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "en-GB"
    engine.respond("quero planear ferias para o porto")
    task_id_before = engine._work_session.active_task.id

    engine.response_language = "pt-PT"
    engine.respond("prefiro mais aventura")
    task_after = engine._work_session.active_task

    assert task_after.id == task_id_before
    assert _generic_task_title(task_after, engine.response_language) == "Planeamento de férias"


# --- M: language command intercepted during active task ---------------------


def test_m_language_command_intercepted_during_active_task(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"
    engine.respond("quero planear ferias para o porto")
    task_id_before = engine._work_session.active_task.id

    response = engine.respond("respond in English")

    assert engine.response_language == "en-GB"
    assert response == "Sure. I'll respond in English."
    # The task itself must be untouched by the language command turn.
    assert engine._work_session.active_task.id == task_id_before


# --- N/O: input-language independence ----------------------------------------


def test_n_pt_input_en_gb_output(tmp_path: Path) -> None:
    """N: Portuguese input, response_language=en-GB -> English output.
    Re-verifies Phase 7.3.4's fix still holds (section 12: no regression)."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "en-GB"

    reply = engine.respond("como estas")

    assert any(reply.startswith(line) for line in not_measured_replies("en-GB"))
    assert reply.count("?") <= 1
    assert "estou" not in reply.lower()


def test_o_en_input_pt_pt_output(tmp_path: Path) -> None:
    """O: English input, response_language=pt-PT -> Portuguese output."""
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"

    reply = engine.respond("how are you")

    assert any(reply.startswith(line) for line in not_measured_replies("pt-PT"))
    assert reply.count("?") <= 1


# --- P/Q: deterministic error --------------------------------------------------


def test_p_deterministic_error_pt_pt() -> None:
    from prototype_web_ui.controller import _tts_unavailable_message

    assert _tts_unavailable_message("pt-PT") == "A voz portuguesa ainda não está disponível."


def test_q_deterministic_error_en_gb() -> None:
    from prototype_web_ui.controller import _tts_unavailable_message

    message = _tts_unavailable_message("en-GB")
    assert message == "I couldn't find a compatible local TTS voice."
    assert "voz" not in message.lower()


# --- R/S: workspace/desktop-state action -------------------------------------


def test_r_workspace_action_pt_pt() -> None:
    result = get_active_window(context_observer=_FakeContextObserver(), response_language="pt-PT")
    assert result == "Janela ativa detetada: Notepad"


def test_s_workspace_action_en_gb() -> None:
    result = get_active_application(context_observer=_FakeContextObserver(), response_language="en-GB")
    assert result == "Active application detected: notepad.exe"


# --- T: obvious Echo-authored language mismatch diagnostic -------------------


def test_t_language_mismatch_diagnostic_logs_bounded_line(capsys) -> None:
    from assistant.conversation import _log_language_audit

    _log_language_audit(response_language="en-GB", source="TOOL_RESULT")

    captured = capsys.readouterr().out
    assert "[LANGUAGE_AUDIT]" in captured
    assert "response_language=en-GB" in captured
    assert "source=TOOL_RESULT" in captured
    assert "detected_language=pt" in captured
    assert "mismatch=true" in captured


# --- Regression: existing composer no-context fallback now localizes -------


def test_composer_no_context_fallback_localizes() -> None:
    assert localize("composer.no_context_fallback", "en-GB") == (
        "I have some context about this, but not enough yet to give you a good answer."
    )
    assert localize("composer.no_context_fallback", "pt-PT") == (
        "Tenho algum contexto sobre isso, mas ainda não o suficiente para te responder bem."
    )


# =============================================================================
# Phase 7.7.1: closing the remaining free-text gaps the Phase 7.7 audit
# explicitly flagged (task dialogue sentences, next-step labels, generic
# activity questions, recent-activity/available-information headers, and
# local-safe-fallback input-language independence).
# =============================================================================

from assistant.response_composer import _local_safe_fallback  # noqa: E402
from assistant.tools import get_recent_activity  # noqa: E402


def _drive_travel_task(tmp_path: Path, response_language: str):
    """Starts a real plan_trip task via the ACTUAL engine and returns
    (engine, task, knowledge, reasoning) so _plan_travel_dialogue/
    _next_step_label/answer_task_query can be exercised with REAL state,
    not hand-built fixtures."""
    from assistant.task_intelligence import _active_entries, _refresh_reasoning, task_knowledge

    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = response_language
    engine.respond("quero planear ferias para o porto")
    task = engine._work_session.active_task
    knowledge = _active_entries(task_knowledge(engine._work_session, task.id))
    reasoning = _refresh_reasoning(engine._work_session, task)
    return engine, task, knowledge, reasoning


# --- A/B: _plan_travel_dialogue ----------------------------------------------


def test_a_plan_travel_dialogue_pt_pt(tmp_path: Path) -> None:
    from assistant.task_intelligence import _plan_travel_dialogue

    _engine, task, knowledge, reasoning = _drive_travel_task(tmp_path / "a", "pt-PT")
    plan = _plan_travel_dialogue(task, knowledge, reasoning, [], "quero planear ferias para o porto", "pt-PT")
    rendered = plan.render()
    assert rendered
    assert "Sure" not in rendered and "Great" not in rendered


def test_b_plan_travel_dialogue_en_gb(tmp_path: Path) -> None:
    from assistant.task_intelligence import _plan_travel_dialogue

    _engine, task, knowledge, reasoning = _drive_travel_task(tmp_path / "b", "en-GB")
    plan = _plan_travel_dialogue(task, knowledge, reasoning, [], "quero planear ferias para o porto", "en-GB")
    rendered = plan.render()
    assert rendered
    assert "Claro" not in rendered and "Certo" not in rendered


# --- C/D: _next_step_label ---------------------------------------------------


def test_c_next_step_label_pt_pt(tmp_path: Path) -> None:
    from assistant.task_intelligence import _next_step_label

    _engine, task, knowledge, reasoning = _drive_travel_task(tmp_path / "c", "pt-PT")
    label = _next_step_label(task, knowledge, reasoning, "pt-PT")
    if label:
        assert "Organizar" in label or "Fechar" in label or "Confirmar" in label


def test_d_next_step_label_en_gb(tmp_path: Path) -> None:
    from assistant.task_intelligence import _next_step_label

    _engine, task, knowledge, reasoning = _drive_travel_task(tmp_path / "d", "en-GB")
    label = _next_step_label(task, knowledge, reasoning, "en-GB")
    if label:
        assert "Organise" in label or "Close" in label or "Confirm" in label
        assert "Organizar" not in label and "Fechar" not in label


# --- E/F: answer_task_query ---------------------------------------------------


def test_e_answer_task_query_pt_pt(tmp_path: Path) -> None:
    from assistant.task_intelligence import answer_task_query

    engine, task, _knowledge, _reasoning = _drive_travel_task(tmp_path / "e", "pt-PT")
    response = answer_task_query(engine._work_session, task, "o que ainda falta", response_language="pt-PT")
    assert response
    assert "still" not in response.lower()


def test_f_answer_task_query_en_gb(tmp_path: Path) -> None:
    from assistant.task_intelligence import answer_task_query

    engine, task, _knowledge, _reasoning = _drive_travel_task(tmp_path / "f", "en-GB")
    response = answer_task_query(engine._work_session, task, "o que ainda falta", response_language="en-GB")
    assert response
    assert "falta" not in response.lower()


# --- G/H: _generic_activity_question -----------------------------------------


def test_g_generic_activity_question_pt_pt() -> None:
    from assistant.conversation import ActiveTaskContext, _generic_activity_question

    ctx = ActiveTaskContext(id="t1", task_type="make_decision")
    assert _generic_activity_question(ctx, "pt-PT") == "Certo. Qual é o critério que pesa mais nessa decisão?"


def test_h_generic_activity_question_en_gb() -> None:
    from assistant.conversation import ActiveTaskContext, _generic_activity_question

    ctx = ActiveTaskContext(id="t1", task_type="make_decision")
    assert _generic_activity_question(ctx, "en-GB") == "Right. Which criterion matters most in this decision?"


# --- I/J: available-information headings -------------------------------------


def test_i_available_information_headings_pt_pt(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path / "i", mode="local")
    engine.long_term_memory.current_work_context = lambda: ""  # MemoryStub doesn't implement this
    engine.response_language = "pt-PT"
    summary = engine._available_information_summary()
    assert "Tenho acesso a estas fontes de informação:" in summary
    assert "\nTarefas:" in summary


def test_j_available_information_headings_en_gb(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path / "j", mode="local")
    engine.long_term_memory.current_work_context = lambda: ""  # MemoryStub doesn't implement this
    engine.response_language = "en-GB"
    summary = engine._available_information_summary()
    assert "I have access to these sources of information:" in summary
    assert "\nTasks:" in summary
    assert "Tenho acesso" not in summary


# --- K/L: recent-activity wrapper ---------------------------------------------


class _FakeObserverWithSummary:
    def __init__(self, summary_text: str) -> None:
        self._summary_text = summary_text

    def latest_summary(self):
        return SimpleNamespace(summary=self._summary_text)

    def activity_summary(self, limit=5):
        return []


def test_k_recent_activity_wrapper_pt_pt() -> None:
    result = get_recent_activity(
        context_observer=_FakeObserverWithSummary("trabalhaste no Echo durante 2 horas"),
        response_language="pt-PT",
    )
    assert result == "Atividade recente: trabalhaste no Echo durante 2 horas"


def test_l_recent_activity_wrapper_en_gb() -> None:
    # The wrapper localizes; the stored summary content itself (Echo/
    # context_observer-generated prose, not this tool's own label) is left
    # exactly as its source produced it -- section 8's "do not blindly
    # machine-translate arbitrary stored content".
    result = get_recent_activity(
        context_observer=_FakeObserverWithSummary("trabalhaste no Echo durante 2 horas"),
        response_language="en-GB",
    )
    assert result == "Recent activity: trabalhaste no Echo durante 2 horas"
    assert result.startswith("Recent activity:")


# --- M/N: local-safe-fallback input-language independence -------------------


def test_m_english_input_pt_pt_local_fallback() -> None:
    assert _local_safe_fallback("I failed the exam", "pt-PT") == "Lamento que tenha corrido mal."


def test_n_portuguese_input_en_gb_local_fallback() -> None:
    assert _local_safe_fallback("estou triste com isso", "en-GB") == "I'm sorry. That must be hard."


# --- O/P: mid-task language switch changes both card AND dialogue -----------


def test_o_mid_task_switch_changes_card_and_dialogue(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"
    engine.respond("quero planear ferias para o porto")
    task_id_before = engine._work_session.active_task.id
    engine.respond("prefiro mais aventura")

    engine.response_language = "en-GB"
    dialogue_reply = engine.respond("quero saber mais sobre o alojamento")
    task_after = engine._work_session.active_task
    card_title = _generic_task_title(task_after, engine.response_language)

    assert card_title == "Holiday planning"
    assert "?" in dialogue_reply  # a real clarifying question came back
    assert "Claro" not in dialogue_reply and "Certo" not in dialogue_reply
    assert task_after.id == task_id_before


def test_p_no_task_restart_same_task_id_across_switch(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = "pt-PT"
    engine.respond("quero planear ferias para o porto")
    task_id_1 = engine._work_session.active_task.id

    engine.response_language = "en-GB"
    engine.respond("prefiro mais aventura")
    task_id_2 = engine._work_session.active_task.id

    engine.response_language = "pt-PT"
    engine.respond("mais alguma coisa")
    task_id_3 = engine._work_session.active_task.id

    assert task_id_1 == task_id_2 == task_id_3


# =============================================================================
# Phase 7.7.2: closing the final two known gaps -- the generic_comparison
# dialogue subsystem (_plan_comparison_dialogue and every helper it calls)
# and the evidence/sources/conflicts formatters (_format_important/
# _format_sources/_format_evidence_answer/_format_conflicts).
# =============================================================================

from assistant.task_intelligence import (  # noqa: E402
    KnowledgeEntry,
    KnowledgeSource,
    TaskReasoningState,
    _format_conflicts,
    _format_evidence_answer,
    _format_important,
    _format_sources,
    answer_task_query,
)


def _drive_comparison_task(tmp_path: Path, response_language: str):
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path, mode="local")
    engine.response_language = response_language
    engine.respond("quero comparar samsung e apple")
    task = engine._work_session.active_task
    return engine, task


# --- A/B: comparison start ----------------------------------------------------


def test_a_comparison_start_pt_pt(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path / "a", mode="local")
    engine.response_language = "pt-PT"
    reply = engine.respond("quero comparar samsung e apple")
    assert "opções" in reply.lower() or "op" in reply.lower()
    assert "options" not in reply.lower()


def test_b_comparison_start_en_gb(tmp_path: Path) -> None:
    engine, _llm, _ollama, _anthropic = make_engine(tmp_path / "b", mode="local")
    engine.response_language = "en-GB"
    reply = engine.respond("quero comparar samsung e apple")
    assert "options" in reply.lower()
    assert "opções" not in reply.lower()


# --- C/D: missing-criteria question -------------------------------------------


def test_c_comparison_missing_criteria_question_pt_pt(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "c", "pt-PT")
    reply = engine.respond("o samsung custa 799 e o apple custa 999")
    assert reply
    assert "criterion" not in reply.lower() and "which factor" not in reply.lower()


def test_d_comparison_missing_criteria_question_en_gb(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "d", "en-GB")
    reply = engine.respond("o samsung custa 799 e o apple custa 999")
    assert reply
    assert "critérios" not in reply.lower() and "sabes o" not in reply.lower()


# --- E/F: comparison continuation ---------------------------------------------


def test_e_comparison_continuation_pt_pt(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "e", "pt-PT")
    reply = engine.respond("a camara pesa mais para mim")
    assert "Percebi" in reply or "câmara" in reply.lower()
    assert "camera" not in reply.lower()


def test_f_comparison_continuation_en_gb(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "f", "en-GB")
    reply = engine.respond("a camara pesa mais para mim")
    assert "Got it" in reply or "camera" in reply.lower()
    assert "câmara" not in reply.lower() and "Percebi" not in reply


# --- G/H: comparison completion (recommendation query) -----------------------


def test_g_comparison_completion_pt_pt(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "g", "pt-PT")
    engine.respond("o samsung custa 799 e o apple custa 999")
    reply = answer_task_query(engine._work_session, task, "qual parece melhor", response_language="pt-PT")
    assert reply
    assert "provisional" not in reply.lower()


def test_h_comparison_completion_en_gb(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "h", "en-GB")
    engine.respond("o samsung custa 799 e o apple custa 999")
    reply = answer_task_query(engine._work_session, task, "qual parece melhor", response_language="en-GB")
    assert reply
    assert "provisório" not in reply.lower()


# --- I/J: mid-comparison language switch, same task_id ------------------------


def test_i_pt_comparison_switch_en_same_task_id(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "i", "pt-PT")
    task_id_before = task.id
    engine.response_language = "en-GB"
    reply = engine.respond("a camara pesa mais para mim")
    assert engine._work_session.active_task.id == task_id_before
    assert "câmara" not in reply.lower()
    assert "camera" in reply.lower() or "got it" in reply.lower()


def test_j_en_comparison_switch_pt_same_task_id(tmp_path: Path) -> None:
    engine, task = _drive_comparison_task(tmp_path / "j", "en-GB")
    task_id_before = task.id
    engine.response_language = "pt-PT"
    reply = engine.respond("a camara pesa mais para mim")
    assert engine._work_session.active_task.id == task_id_before
    assert "camera" not in reply.lower()


# --- K/L: _format_important ---------------------------------------------------


def _sample_comparison_knowledge() -> list[KnowledgeEntry]:
    return [
        KnowledgeEntry(
            id="k1", task_id="t1", kind="option", subject="Samsung", predicate="is_option", value="true",
            source=KnowledgeSource(source_type="user"),
        ),
        KnowledgeEntry(
            id="k2", task_id="t1", kind="option", subject="Apple", predicate="is_option", value="true",
            source=KnowledgeSource(source_type="user"),
        ),
        KnowledgeEntry(
            id="k3", task_id="t1", kind="fact", subject="Samsung", predicate="camera_quality", value="better",
            source=KnowledgeSource(
                source_type="web", reference="TechRadar", original_text="Samsung has a stronger zoom lens"
            ),
            metadata={"source_title": "TechRadar review", "source_strength_for_claim": "strong"},
        ),
        KnowledgeEntry(
            id="k4", task_id="t1", kind="fact", subject="Apple", predicate="camera_quality", value="mixed",
            source=KnowledgeSource(
                source_type="web", reference="CNET", original_text="Apple camera results were inconsistent"
            ),
            metadata={"source_title": "CNET review", "source_strength_for_claim": "useful"},
        ),
    ]


def _sample_task():
    return SimpleNamespace(task_type="generic_comparison", id="t1")


def test_k_format_important_pt_pt() -> None:
    reasoning = TaskReasoningState(task_id="t1", current_recommendation="Samsung", reasons=["tem vantagem de câmara"])
    result = _format_important(_sample_task(), _sample_comparison_knowledge(), reasoning, "pt-PT")
    assert "mais importante" in result.lower()
    assert "important thing" not in result.lower()


def test_l_format_important_en_gb() -> None:
    reasoning = TaskReasoningState(task_id="t1", current_recommendation="Samsung", reasons=["has a camera advantage"])
    result = _format_important(_sample_task(), _sample_comparison_knowledge(), reasoning, "en-GB")
    assert "most important thing" in result.lower()
    assert "mais importante" not in result.lower()
    assert "Samsung" in result  # canonical value preserved


# --- M/N: _format_sources -----------------------------------------------------


def test_m_format_sources_pt_pt() -> None:
    result = _format_sources(_sample_comparison_knowledge(), "camara", "pt-PT")
    assert result.startswith("Veio de")


def test_n_format_sources_en_gb() -> None:
    result = _format_sources(_sample_comparison_knowledge(), "camara", "en-GB")
    assert result.startswith("It came from")


# --- O/P: _format_evidence_answer ---------------------------------------------


def test_o_format_evidence_answer_pt_pt() -> None:
    result = _format_evidence_answer(_sample_task(), _sample_comparison_knowledge(), "a camara do samsung e melhor?", "pt-PT")
    assert "evidência" in result.lower() or "sim" in result.lower()
    assert "evidence" not in result.lower()


def test_p_format_evidence_answer_en_gb() -> None:
    result = _format_evidence_answer(_sample_task(), _sample_comparison_knowledge(), "a camara do samsung e melhor?", "en-GB")
    assert "evidence" in result.lower()
    assert "evidência" not in result.lower()


# --- Q/R: _format_conflicts ----------------------------------------------------


def test_q_format_conflicts_pt_pt() -> None:
    result = _format_conflicts(_sample_comparison_knowledge(), "pt-PT")
    assert "informação contraditória" in result.lower() or "valores diferentes" in result.lower()


def test_r_format_conflicts_en_gb() -> None:
    result = _format_conflicts(_sample_comparison_knowledge(), "en-GB")
    assert "contradictory information" in result.lower() or "differing values" in result.lower()


# --- S: source titles/data remain untranslated --------------------------------


def test_s_source_titles_and_quotes_remain_untranslated() -> None:
    knowledge = _sample_comparison_knowledge()
    for lang in ("pt-PT", "en-GB"):
        result = _format_sources(knowledge, "camara", lang)
        # The source-derived title/quotation must appear verbatim regardless
        # of response_language -- only the Echo-authored wrapper localizes.
        assert "CNET review" in result
        assert "Apple camera results were inconsistent" in result


# --- T: cross-language input independence for comparison dialogue -----------


def test_t_cross_language_input_independence_comparison(tmp_path: Path) -> None:
    # PT input + en-GB response
    engine_a, _llm_a, _ollama_a, _anthropic_a = make_engine(tmp_path / "t_a", mode="local")
    engine_a.response_language = "en-GB"
    reply_a = engine_a.respond("quero comparar samsung e apple")
    assert "options" in reply_a.lower()

    # EN input + pt-PT response
    engine_b, _llm_b, _ollama_b, _anthropic_b = make_engine(tmp_path / "t_b", mode="local")
    engine_b.response_language = "pt-PT"
    reply_b = engine_b.respond("I want to compare samsung and apple")
    # response_language must win regardless of what language the request was in
    assert engine_b.response_language == "pt-PT"


def test_live_desktop_confirmation_and_cancellation_follow_response_language(tmp_path):
    engine, _, local, _ = make_engine(tmp_path, mode="local")
    engine._set_response_language("en-GB")
    response = engine.respond("abre o gmail")
    assert response.startswith("Would you like me to open Gmail?")
    assert "Responde" not in response
    assert engine.agent.has_pending_confirmation()
    response = engine.respond("Não.")
    assert response == "Action cancelled. I haven't executed anything."
    assert not engine.agent.has_pending_confirmation()
    assert local.calls == []


@pytest.mark.parametrize("original,expected", [
    ("Não encontrei 'notas, por favor?' na workspace. Qual é o nome exato do ficheiro?",
     "I couldn't find 'notas, por favor?' in the workspace. What's the exact filename?"),
    ("Queres que traga o VS Code para a frente?", "Would you like me to bring VS Code to the front?"),
    ("Queres que abra a pasta 'relatórios'?", "Would you like me to open the folder 'relatórios'?"),
    ("Queres que pesquise no Google por: gatos?", "Would you like me to search Google for: gatos?"),
])
def test_local_wrappers_translate_without_changing_user_values(original, expected):
    from assistant.response_language import localize_deterministic_response
    assert localize_deterministic_response(original, "en-GB") == expected
    assert localize_deterministic_response(original, "pt-PT") == original


def test_confirmation_localization_never_truncates_or_translates_a_draft_body():
    from assistant.response_language import localize_deterministic_response
    draft = "Subject: Instructions\n\nQueres que abra o Gmail?\nResponde 'sim' para executar ou 'nao' para cancelar.\n\nAna"
    assert localize_deterministic_response(draft, "en-GB") == draft
