"""R3 step 3: Alexandre's core profile is always present in the composer.

Uses the fictitious profile in tests/fixtures/; the real file
(data/profile/core_profile.md) is private and must never be read here.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from assistant import core_profile
from assistant.response_composer import ComposerRequest, ResponseComposer, _SYSTEM_PROMPT, _SYSTEM_PROMPT_EN
from assistant.response_language import response_language_instruction

FIXTURE = Path(__file__).parent / "fixtures" / "core_profile.md"
PROFILE_FACT = "Toca violoncelo desde os dez anos."


class _CapturingLLM:
    def __init__(self) -> None:
        self.system_prompts: list[str] = []

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.system_prompts.append(system_prompt or "")
        return "Paris."


def _compose(language: str) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).compose(
        ComposerRequest(intent="conversa", user_message="Qual é a capital de França?", response_language=language)
    )
    return llm.system_prompts[-1]


def _regenerate(language: str) -> str:
    llm = _CapturingLLM()
    ResponseComposer(llm).regenerate("Qual é a capital de França?", [], "emotion_inversion", response_language=language)
    return llm.system_prompts[-1]


@pytest.fixture
def profile_file(tmp_path, monkeypatch) -> Path:
    path = tmp_path / "core_profile.md"
    path.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(path))
    return path


def _touch_later(path: Path) -> None:
    stat = path.stat()
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))


# --- composer payload ---------------------------------------------------------


@pytest.mark.parametrize("build", [_compose, _regenerate])
def test_profile_is_in_the_english_system_prompt(profile_file, build) -> None:
    prompt = build("en-GB")

    assert "About Alexandre" in prompt
    assert PROFILE_FACT in prompt
    assert "Use these notes silently" in prompt
    assert "only when it is relevant" in prompt
    assert "do not authorise inventing" in prompt
    assert "Sobre o Alexandre" not in prompt
    # The profile sits between the base rules and the language instruction, which stays last.
    assert prompt.startswith(_SYSTEM_PROMPT_EN)
    assert prompt.endswith(response_language_instruction("en-GB"))


@pytest.mark.parametrize("build", [_compose, _regenerate])
def test_profile_is_in_the_portuguese_system_prompt(profile_file, build) -> None:
    prompt = build("pt-PT")

    assert "Sobre o Alexandre" in prompt
    assert PROFILE_FACT in prompt
    assert "Usa estas notas em silêncio" in prompt
    assert "só quando for relevante" in prompt
    assert "não te autorizam a inventar" in prompt
    assert "About Alexandre" not in prompt
    assert prompt.startswith(_SYSTEM_PROMPT)
    assert prompt.endswith(response_language_instruction("pt-PT"))


@pytest.mark.parametrize("language,base", [("pt-PT", _SYSTEM_PROMPT), ("en-GB", _SYSTEM_PROMPT_EN)])
def test_missing_profile_leaves_the_payload_as_it_is_today(language, base, capsys) -> None:
    expected = f"{base}\n\n{response_language_instruction(language)}"

    assert _compose(language) == expected
    assert _regenerate(language) == expected
    assert "[CORE_PROFILE] status=missing" in capsys.readouterr().out


# --- loader -------------------------------------------------------------------


def test_profile_is_personal_and_logged_once(profile_file, capsys) -> None:
    first = core_profile.load_core_profile()
    second = core_profile.load_core_profile()

    assert first is not None and PROFILE_FACT in first.text
    assert first.privacy_class == core_profile.PRIVACY_CLASS == "PERSONAL"
    assert second is first
    out = capsys.readouterr().out
    assert out.count("[CORE_PROFILE] status=loaded") == 1
    assert "violoncelo" not in out  # the content never goes to the log


def test_profile_reloads_when_the_file_changes(profile_file) -> None:
    assert "Toca violoncelo" in _compose("pt-PT")

    profile_file.write_text("- Toca oboé.\n", encoding="utf-8")
    _touch_later(profile_file)

    prompt = _compose("pt-PT")
    assert "Toca oboé." in prompt
    assert "violoncelo" not in prompt


def test_profile_appears_and_disappears_without_a_restart(profile_file) -> None:
    profile_file.unlink()
    assert "Sobre o Alexandre" not in _compose("pt-PT")

    profile_file.write_text("- Toca oboé.\n", encoding="utf-8")
    assert "Toca oboé." in _compose("pt-PT")


def test_size_limit_keeps_only_the_start_and_warns(profile_file, capsys) -> None:
    profile_file.write_text("INICIO " + "a" * 5000 + " FIM", encoding="utf-8")

    profile = core_profile.load_core_profile()

    assert profile is not None
    assert len(profile.text) <= core_profile.MAX_CHARS == 3000
    assert profile.text.startswith("INICIO")
    assert "FIM" not in profile.text
    assert "[CORE_PROFILE] status=truncated" in capsys.readouterr().out


@pytest.mark.parametrize("content", [b"", b"   \n\n", b"\xff\xfe\x00 not utf-8 \xff"])
def test_empty_or_unreadable_profile_never_crashes(profile_file, content, capsys) -> None:
    profile_file.write_bytes(content)

    assert core_profile.load_core_profile() is None
    assert "Sobre o Alexandre" not in _compose("pt-PT")
    out = capsys.readouterr().out
    assert "[CORE_PROFILE] status=missing" in out or "[CORE_PROFILE] status=error" in out


def test_a_directory_in_place_of_the_file_is_an_error_not_a_crash(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(tmp_path))

    assert core_profile.load_core_profile() is None
    assert "[CORE_PROFILE] status=error" in capsys.readouterr().out


# --- the real file is never read in tests --------------------------------------


def test_default_path_is_the_private_file_and_tests_never_use_it(monkeypatch) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    assert core_profile.DEFAULT_PATH == repo_root / "data" / "profile" / "core_profile.md"

    # tests/conftest.py points every test away from it.
    assert core_profile.profile_path() != core_profile.DEFAULT_PATH
    assert not core_profile.profile_path().exists()

    monkeypatch.delenv("ECHO_CORE_PROFILE_PATH")
    assert core_profile.profile_path() == core_profile.DEFAULT_PATH


def test_real_profile_folder_is_ignored_by_git() -> None:
    gitignore = (Path(__file__).resolve().parents[1] / ".gitignore").read_text(encoding="utf-8")
    assert "data/profile/" in gitignore.splitlines()
