"""R3 conversation eval: the case set, the property checks and the runner.

The runner is exercised against the real AssistantEngine with a fake LLM
provider only: no Ollama, no cloud.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from evals.conversation import runner
from evals.conversation.checks import FAIL, PASS, PROPERTIES, REVIEW, evaluate_turn

REPO_ROOT = Path(__file__).resolve().parents[1]


def _statuses(response: str, language: str = "en-GB", turn: dict | None = None, prior: list[str] | None = None) -> dict:
    outcomes = evaluate_turn(response, language, turn or {"user": "x"}, prior or [])
    assert [o.property for o in outcomes] == list(PROPERTIES)
    return {o.property: o.status for o in outcomes}


@pytest.fixture
def isolated_eval_env(tmp_path, monkeypatch):
    monkeypatch.setenv("ECHO_ENV", "test")
    monkeypatch.setenv("ECHO_TEST_DATA_DIR", str(tmp_path / "eval-root"))


# --- case set ---------------------------------------------------------------


def test_case_set_covers_what_r3_asked_for() -> None:
    cases = runner.load_cases()
    users = [turn["user"] for case in cases for turn in case["turns"]]

    assert len(cases) >= 30
    assert {case["language"] for case in cases} == {"pt-PT", "en-GB"}
    for phrase in (
        "Bom dia, como é que estás?",
        "Porquê que dizes sempre parâmetros normais? Sempre que te pergunto como é que estás?",
        "Querias que eu ver um mal contigo, mas jostas-me.",
    ):
        assert phrase in users
    assert any(len(case["turns"]) >= 4 for case in cases)
    assert {"greeting", "garbled", "help_request", "personal_share", "about_echo", "conversation"} <= {
        case["category"] for case in cases
    }


def test_load_cases_rejects_an_unknown_property(tmp_path) -> None:
    path = tmp_path / "cases.json"
    case = {"id": "a", "language": "pt-PT", "turns": [{"user": "Olá", "review": ["politeness"]}]}
    path.write_text(json.dumps({"cases": [case]}), encoding="utf-8")

    with pytest.raises(ValueError, match="politeness"):
        runner.load_cases(path)


# --- checks: the replies from the real log must fail ------------------------


def test_real_tic_reply_fails_only_the_tic_property() -> None:
    statuses = _statuses("Good morning, I'm functioning within normal parameters.")

    assert statuses["no_tic_phrases"] == FAIL
    assert statuses["language"] == PASS
    assert statuses["not_rude"] == PASS


def test_real_reply_to_garbled_input_is_rude_and_does_not_ask_to_repeat() -> None:
    statuses = _statuses(
        "It seems you're a bit confused. I'm here to assist you, not to argue or play tricks on you.",
        turn={"user": "Querias que eu ver um mal contigo, mas jostas-me.", "garbled": True},
    )

    assert statuses["not_rude"] == FAIL
    assert statuses["answers_question"] == FAIL
    assert statuses["no_needless_follow_up"] == FAIL


def test_polite_request_to_repeat_passes_a_garbled_turn() -> None:
    turn = {"user": "Eco, as luz do tem que jantar.", "garbled": True}

    assert set(_statuses("Sorry, I didn't catch that. Could you say it again?", turn=turn).values()) == {PASS}
    assert set(_statuses("Desculpa, não percebi. Podes repetir?", "pt-PT", turn).values()) == {PASS}


def test_real_third_person_reply_fails_second_person_but_vocative_passes() -> None:
    real = "What's the main point you'd like to discuss or remind Alexandre about?"

    assert _statuses(real)["second_person"] == FAIL
    assert _statuses("The user wants an email.")["second_person"] == FAIL
    assert _statuses("O Alexandre quer escrever um email.", "pt-PT")["second_person"] == FAIL
    assert _statuses("Good morning, Alexandre.")["second_person"] == PASS
    assert _statuses("You, Alexandre.")["second_person"] == PASS


def test_real_portuguese_reply_fails_language_under_en_gb() -> None:
    real = "Boa. Podemos retomar o projeto AssistenteIA. Eu comecaria por rever onde ficamos."

    statuses = _statuses(real)
    assert statuses["language"] == FAIL
    assert statuses["no_invented_state"] == FAIL  # "onde ficamos": invented continuity
    assert _statuses(real, "pt-PT")["language"] == PASS
    assert _statuses("Good morning. How can I help?", "pt-PT")["language"] == FAIL


def test_quoting_the_users_words_does_not_count_as_the_wrong_language() -> None:
    assert _statuses('You said "não possamos decidir nada, só estava curioso", so I will leave it there.')["language"] == PASS


def test_invented_activity_and_memory_fail() -> None:
    assert _statuses("I've been working on your files while you were away.")["no_invented_state"] == FAIL
    assert _statuses("Tenho estado a organizar as tuas notas.", "pt-PT")["no_invented_state"] == FAIL
    assert _statuses("I don't remember anything from before this conversation.")["no_invented_state"] == PASS


def test_follow_up_modes() -> None:
    assert _statuses("Glad to hear it.", turn={"user": "x", "follow_up": "none"})["no_needless_follow_up"] == PASS
    assert _statuses("Anything else?", turn={"user": "x", "follow_up": "none"})["no_needless_follow_up"] == FAIL
    assert _statuses("With what?", turn={"user": "x", "follow_up": "required"})["no_needless_follow_up"] == PASS
    assert _statuses("Sure.", turn={"user": "x", "follow_up": "required"})["no_needless_follow_up"] == FAIL
    assert _statuses("What is it? When is it due?")["no_needless_follow_up"] == FAIL


def test_answer_any_review_soft_and_forbid() -> None:
    answer = {"user": "x", "answer_any": [["90", "noventa"]]}
    assert _statuses("São noventa minutos.", "pt-PT", answer)["answers_question"] == PASS
    assert _statuses("Não sei.", "pt-PT", answer)["answers_question"] == FAIL
    assert _statuses("Fine.")["answers_question"] == REVIEW

    assert _statuses("Fine.", turn={"user": "x", "review": ["not_rude"]})["not_rude"] == REVIEW
    soft = {"user": "x", "soft": ["no_tic_phrases"]}
    assert _statuses("I said I was functioning normally.", turn=soft)["no_tic_phrases"] == REVIEW
    forbid = {"user": "x", "forbid": {"no_invented_state": ["assistenteia"]}}
    assert _statuses("Let's resume AssistenteIA.", turn=forbid)["no_invented_state"] == FAIL


def test_repeating_an_earlier_reply_is_a_tic_and_empty_reply_fails_everything() -> None:
    reply = "I am here and ready to help."

    assert _statuses(reply, prior=[reply])["no_tic_phrases"] == FAIL
    assert _statuses(reply, prior=["Something else entirely was said."])["no_tic_phrases"] == PASS
    assert set(_statuses("  ").values()) == {FAIL}


# --- runner: real engine, fake LLM ------------------------------------------


def test_runner_drives_the_real_engine_in_the_case_language(isolated_eval_env) -> None:
    seen_system_prompts: list[str] = []

    def reply(messages: list[dict]) -> str:
        seen_system_prompts.append(messages[0]["content"])
        return "Good morning. How can I help you today?"

    cases = [
        {"id": "en", "language": "en-GB", "turns": [{"user": "Estou um bocado stressado com o trabalho."}]},
        {"id": "pt", "language": "pt-PT", "turns": [{"user": "Hoje estou mesmo cansado."}, {"user": "Pois é."}]},
    ]
    factory = runner.make_llm_factory("fake", fake_reply=reply)

    report = runner.run_cases(cases, factory, provider="fake")

    assert seen_system_prompts, "the fake provider was never called: the engine did not reach the LLM"
    assert [len(case["turns"]) for case in report["cases"]] == [1, 2]
    assert all(not turn["error"] and turn["response"] for case in report["cases"] for turn in case["turns"])
    summary = report["summary"]
    assert summary["cases"] == 2 and summary["turns"] == 3 and summary["errors"] == 0
    for row in summary["properties"].values():
        assert row["pass"] + row["fail"] + row["review"] == 3
    # The English fake reply is the wrong language for the pt-PT case only.
    assert summary["by_language"]["en-GB"]["language"]["pass_rate"] == 1.0
    assert summary["by_language"]["pt-PT"]["language"]["pass_rate"] == 0.0


def test_summary_rates_and_markdown_report(tmp_path) -> None:
    def turn(status_by_property: dict) -> dict:
        outcomes = [
            {"property": name, "status": status_by_property.get(name, PASS), "detail": "why"} for name in PROPERTIES
        ]
        return {"user": "Olá", "response": "This sentence shows up in every case.", "error": "", "outcomes": outcomes}

    results = [
        {"id": f"case_{index}", "language": "en-GB", "turns": [turn(statuses)]}
        for index, statuses in enumerate(
            ({"no_tic_phrases": FAIL}, {"no_tic_phrases": REVIEW}, {}, {"answers_question": REVIEW})
        )
    ]
    report = {"provider": "fake", "model": "", "summary": runner.summarize(results), "cases": results}

    tic = report["summary"]["properties"]["no_tic_phrases"]
    assert tic == {"pass": 2, "fail": 1, "review": 1, "pass_rate": 0.667}
    assert report["summary"]["by_language"]["pt-PT"]["language"]["pass_rate"] is None
    assert report["summary"]["repeated_sentences"] == [{"sentence": "this sentence shows up in every case", "cases": 4}]

    json_path, md_path = runner.write_report(report, tmp_path / "out")
    markdown = md_path.read_text(encoding="utf-8")
    assert json.loads(json_path.read_text(encoding="utf-8"))["summary"] == report["summary"]
    assert "| no_tic_phrases | 67% | 2 | 1 | 1 | n/a | 67% |" in markdown
    assert "**case_0** turn 1" in markdown.split("## Human review")[0]
    assert "**case_3** turn 1" in markdown.split("## Human review")[1]


def test_claude_needs_explicit_opt_in() -> None:
    with pytest.raises(PermissionError):
        runner.make_llm_factory("anthropic")
    with pytest.raises(ValueError):
        runner.make_llm_factory("openai")


def test_cli_runs_selected_cases_with_the_fake_provider(isolated_eval_env, tmp_path, capsys) -> None:
    spec = importlib.util.spec_from_file_location("eval_conversation", REPO_ROOT / "scripts" / "eval_conversation.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out_dir = tmp_path / "report"

    exit_code = module.main(
        ["--provider", "fake", "--case", "log_greeting_how_are_you_en", "--case", "garbled_short_pt", "--out", str(out_dir)]
    )

    assert exit_code == 0
    report = json.loads((out_dir / "report.json").read_text(encoding="utf-8"))
    assert [case["id"] for case in report["cases"]] == ["log_greeting_how_are_you_en", "garbled_short_pt"]
    assert (out_dir / "report.md").exists()
    printed = capsys.readouterr().out
    assert "no_tic_phrases" in printed and "[LIFECYCLE]" not in printed
    with pytest.raises(SystemExit):
        module.main(["--provider", "anthropic", "--out", str(out_dir)])
