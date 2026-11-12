"""Voice test of 11/11: "Queria escrever um mail contigo, ajudas-me?" went to
GENERAL_CONVERSATION instead of opening the email task. The writing-request
detector only knew preciso/quero/ajuda/podes/consegues -- not the
conditional "queria", the 2nd person "ajudas", or English."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.test_automatic_routing_regressions import make_engine


def _route(tmp_path: Path, message: str):
    engine, *_ = make_engine(tmp_path, mode="local")
    engine.response_language = "en-GB"
    engine.respond(message)
    telemetry = engine.get_last_turn_telemetry() or {}
    return telemetry.get("selected_path"), engine._pending_writing_task is not None


@pytest.mark.parametrize(
    "message",
    [
        "Queria escrever um mail contigo, ajudas-me?",
        "queria escrever um mail contigo",
        "ajudas-me a escrever um email",
        "preciso de mandar um mail",
        "can you help me write an email",
        "Boa, boa. Queria escrever o mail contigo. Ajudas-me?",
        "Boa. Queria escrever o email contigo.",
        "Olá! Gostava de escrever o mail contigo.",
        "Queria redigir o email.",
    ],
)
def test_email_request_opens_the_email_task(tmp_path: Path, message: str) -> None:
    assert _route(tmp_path, message) == ("WRITING_TASK", True)


@pytest.mark.parametrize(
    "message",
    [
        "recebi um mail",
        "o mail do João é engraçado",
        # A request verb about an email, but not to write one.
        "I need to check my email",
        "help me find an email from João",
        "I want to read my emails",
        "preciso de ler o mail do João",
        "queria ver os mails de hoje",
        "Boa, boa. Não quero escrever o mail.",
        "Não queria escrever um email.",
        "Queria ler o email contigo.",
        "Não me ajudes a escrever o email.",
    ],
)
def test_talking_about_an_email_is_not_a_request(tmp_path: Path, message: str) -> None:
    # Other routes may answer (a read can go to the workspace search); the
    # email-writing task must not open.
    path, pending = _route(tmp_path, message)
    assert path != "WRITING_TASK"
    assert not pending
