"""R3 step 4, point 3: a question about Alexandre's own life whose answer is
in neither the profile, the conversation nor the supplied memory context gets
"I don't know", never a guess. Real case: "Quantos irmãos tenho?" was answered
"Tens um irmão." with a profile that says nothing about siblings.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from assistant.personal_facts import missing_personal_fact_reply, unknown_reply
from assistant.response_composer import ComposerRequest, ResponseComposer

FIXTURE = Path(__file__).parent / "fixtures" / "core_profile.md"
PROFILE = FIXTURE.read_text(encoding="utf-8")


class _CountingLLM:
    def __init__(self, reply: str = "Tens um irmão.") -> None:
        self.calls = 0
        self.system_prompts: list[str] = []
        self.reply = reply

    def chat(self, user_message, history=None, system_prompt=None, **kwargs):
        self.calls += 1
        self.system_prompts.append(system_prompt or "")
        return self.reply


@pytest.mark.parametrize("message", [
    "Quantos irmãos tenho?", "Tenho irmãs?", "Qual é a minha morada?", "Quando é o meu aniversário?", "Quando faço anos?",
    "Como se chama o meu treinador?", "Quem é o meu treinador?", "Como se chama a minha mãe?", "Qual é o meu número de telefone?",
    "How many siblings do I have?", "What's my address?", "When is my birthday?", "What is my coach's name?", "Do I have a dog?",
])
def test_absent_personal_fact_is_not_guessed(message: str) -> None:
    assert missing_personal_fact_reply(message, [PROFILE], "en-GB") == unknown_reply("en-GB")
    assert missing_personal_fact_reply(message, [PROFILE], "pt-PT") == unknown_reply("pt-PT")
    assert missing_personal_fact_reply(message, [], "pt-PT") == unknown_reply("pt-PT")


def test_the_reply_wording() -> None:
    assert unknown_reply("en-GB") == "I'm afraid I don't know that — tell me and I'll remember it, if you like."
    assert unknown_reply("pt-PT") == "Receio não saber isso — diz-me e eu guardo, se quiseres."


@pytest.mark.parametrize("message,evidence", [
    ("Quantos irmãos tenho?", "- Tem dois irmãos mais novos."),
    ("Quantos irmãos tenho?", "Tenho uma irmã chamada Rita."),
    ("Como se chama o meu treinador?", "O treinador de voleibol é o Paulo."),
    ("When is my birthday?", "Born on 3 March."),
    ("Qual é a minha morada?", "Morada: Rua das Flores, 12."),
])
def test_the_model_answers_when_the_evidence_mentions_the_topic(message: str, evidence: str) -> None:
    assert missing_personal_fact_reply(message, [PROFILE, evidence], "pt-PT") is None


# A fictitious profile in which the user IS a coach, a brother and a dog trainer's
# neighbour, but none of those lines says who his coach, siblings or pets are.
ROLE_PROFILE = """
- É treinador de voleibol dos juniores e presidente da secção.
- Presidente, treinador e atleta da equipa sénior.
- Works as a coach at the club on Saturdays.
- Gosta de cães, mas vive num apartamento pequeno.
- Trata os atletas mais novos como irmãos.
- Vive em Aveiro, perto da Rua Direita.
- Organiza a festa de aniversário do clube todos os anos.
- Atende o telefone do clube às segundas.
"""


@pytest.mark.parametrize("message", [
    "Como se chama o meu treinador?", "Quem é o meu treinador?", "What is my coach's name?", "Who is my coach?",
    "Quantos irmãos tenho?", "Do I have a dog?", "Qual é a minha morada?", "Quando é o meu aniversário?",
    "Qual é o meu número de telefone?",
])
def test_a_word_used_in_another_sense_is_not_evidence(message: str) -> None:
    """Real case: the profile said Alexandre is a coach, so "treinador" was found
    and the model was left to invent his coach's name."""
    assert missing_personal_fact_reply(message, [ROLE_PROFILE], "pt-PT") == unknown_reply("pt-PT")


@pytest.mark.parametrize("message,evidence", [
    ("Como se chama o meu treinador?", "- O treinador dele é o Paulo Matos."),
    ("Como se chama o meu treinador?", "O meu treinador chama-se Paulo."),
    ("Como se chama o meu treinador?", "Treinador: Paulo Matos"),
    ("Como se chama o meu treinador?", "É treinado pelo Paulo Matos."),
    ("What is my coach's name?", "My coach is Paul."),
    ("What is my coach's name?", "He is coached by Paul."),
    ("Quantos irmãos tenho?", "Irmãos: Rui e Marta."),
    ("Quantos irmãos tenho?", "A irmã dele chama-se Marta."),
    ("Do I have a dog?", "Tem um cão chamado Bolt."),
    ("Qual é a minha morada?", "Mora na Rua das Flores, 12."),
    ("Quando é o meu aniversário?", "Nasceu a 3 de março de 2003."),
    ("Quando é o meu aniversário?", "Faz anos a 3 de março."),
    ("Qual é o meu número de telefone?", "Telemóvel: 912 345 678"),
])
def test_a_concrete_fact_is_evidence(message: str, evidence: str) -> None:
    assert missing_personal_fact_reply(message, [ROLE_PROFILE, evidence], "pt-PT") is None


@pytest.mark.parametrize("message", [
    "O que é que eu estudo?", "O que faço na AAC?", "Que instrumento toco?",   # answered by the profile
    "O meu irmão vem cá jantar amanhã.", "Tenho de ligar ao meu treinador.",    # statements, not questions
    "Achas que devo ligar à minha mãe?", "Can you help me write to my coach?",  # advice and requests
    "Quantos irmãos tem o João?", "Qual é a capital de Itália?",                # not about Alexandre
    "O que é que sabes sobre mim?",
])
def test_everything_else_is_left_alone(message: str) -> None:
    assert missing_personal_fact_reply(message, [PROFILE], "pt-PT") is None


# --- through the composer -------------------------------------------------------


@pytest.fixture
def profile(tmp_path, monkeypatch) -> None:
    path = tmp_path / "core_profile.md"
    path.write_text(PROFILE, encoding="utf-8")
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(path))


@pytest.mark.parametrize("language", ["en-GB", "pt-PT"])
def test_composer_does_not_call_the_model_for_an_absent_fact(profile, language) -> None:
    llm = _CountingLLM()

    reply = ResponseComposer(llm).compose(
        ComposerRequest(intent="conversa", user_message="Quantos irmãos tenho?", response_language=language)
    )

    assert reply == unknown_reply(language)
    assert llm.calls == 0


def test_composer_uses_what_the_user_said_earlier_but_not_what_echo_said(profile) -> None:
    told = [{"role": "user", "content": "Tenho dois irmãos, o Rui e a Marta."}, {"role": "assistant", "content": "Entendido."}]
    invented = [{"role": "user", "content": "Olá"}, {"role": "assistant", "content": "Tens um irmão."}]

    llm = _CountingLLM("Tens dois irmãos.")
    request = ComposerRequest(intent="conversa", user_message="Quantos irmãos tenho?", history=told, response_language="pt-PT")
    assert ResponseComposer(llm).compose(request) == "Tens dois irmãos."
    assert llm.calls == 1

    llm = _CountingLLM()
    request = ComposerRequest(intent="conversa", user_message="Quantos irmãos tenho?", history=invented, response_language="pt-PT")
    assert ResponseComposer(llm).compose(request) == unknown_reply("pt-PT")
    assert llm.calls == 0


def test_asking_twice_does_not_turn_the_question_into_evidence(profile) -> None:
    asked_before = [
        {"role": "user", "content": "Como se chama o meu treinador?"},
        {"role": "assistant", "content": unknown_reply("pt-PT")},
    ]
    llm = _CountingLLM("Chama-se Ricardo.")

    request = ComposerRequest(intent="conversa", user_message="Quem é o meu treinador?", history=asked_before, response_language="pt-PT")

    assert ResponseComposer(llm).compose(request) == unknown_reply("pt-PT")
    assert llm.calls == 0


def test_composer_uses_memory_handed_in_as_facts_or_context(profile) -> None:
    llm = _CountingLLM("O teu treinador é o Paulo.")

    reply = ResponseComposer(llm).compose(ComposerRequest(
        intent="conversa", user_message="Como se chama o meu treinador?", facts=["Treinador: Paulo."], response_language="pt-PT",
    ))

    assert reply == "O teu treinador é o Paulo."


@pytest.mark.parametrize("language,rule", [
    ("en-GB", "say that you do not know and offer to remember it; never guess"),
    ("pt-PT", "diz que não sabes e oferece-te para o guardar; nunca adivinhes"),
])
def test_prompt_tells_the_model_not_to_guess_other_personal_facts(profile, language, rule) -> None:
    llm = _CountingLLM("Estudas Arquitetura Paisagista.")

    ResponseComposer(llm).compose(ComposerRequest(intent="conversa", user_message="O que é que eu estudo?", response_language=language))

    assert llm.calls == 1
    assert rule in llm.system_prompts[-1]
