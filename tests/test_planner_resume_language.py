"""Voice test of 11/11: "Vamos continuar a tarefa do email." with no email
task fell to the agent, whose planner answered with a fixed Portuguese text
(no accents) about resuming the AssistenteIA project -- in an en-GB session.
The planner's resume text now goes through localize(), and a resume request
about an email, with no email task, says there is none and offers to start
one (an active or paused email task is handled by the task routing first)."""
from __future__ import annotations

from pathlib import Path

from assistant.planner import plan_user_request
from tests.test_automatic_routing_regressions import make_engine


def _respond(tmp_path: Path, message: str, language: str) -> str:
    engine, *_ = make_engine(tmp_path, mode="local")
    engine.response_language = language
    return engine.respond(message)


def test_resume_email_without_a_task_says_there_is_none_in_english(tmp_path: Path) -> None:
    response = _respond(tmp_path, "Vamos continuar a tarefa do email.", "en-GB")
    assert "no email in progress" in response.lower()
    assert "start one" in response.lower()
    assert "AssistenteIA" not in response


def test_resume_email_without_a_task_says_there_is_none_in_portuguese(tmp_path: Path) -> None:
    response = _respond(tmp_path, "Vamos continuar a tarefa do email.", "pt-PT")
    assert "não há nenhum email em curso" in response.lower()
    assert "AssistenteIA" not in response


def test_resume_project_text_is_localized() -> None:
    english = plan_user_request("vamos continuar o projeto", response_language="en-GB").direct_response
    portuguese = plan_user_request("vamos continuar o projeto", response_language="pt-PT").direct_response
    assert english.startswith("Good. We can pick up the AssistenteIA project.")
    assert portuguese.startswith("Boa. Podemos retomar o projeto AssistenteIA.")
    assert "começaria" in portuguese and "próximo" in portuguese
