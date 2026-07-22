from assistant.response_composer import ComposerRequest, ResponseComposer


class FakeLLM:
    def __init__(self, replies: str | list[str]) -> None:
        self.replies = [replies] if isinstance(replies, str) else list(replies)
        self.calls: list[tuple[str, str | None]] = []

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.calls.append((user_message, system_prompt))
        if not self.replies:
            return ""
        return self.replies.pop(0)


def test_default_response_is_directed_by_prompt_not_cut_blindly() -> None:
    reply = (
        "Acho que há aqui um padrão importante. "
        "Costumas querer perceber primeiro a arquitetura antes de começares a mexer no código."
    )
    llm = FakeLLM(reply)
    composer = ResponseComposer(llm)

    answer = composer.compose(
        ComposerRequest(
            intent="personal_model",
            user_message="o que sabes sobre mim?",
            facts=["O Alexandre gosta de estruturar projetos antes de programar."],
        )
    )

    assert answer == reply
    assert len(llm.calls) == 1


def test_voice_critic_is_not_run_inside_response_composer() -> None:
    llm = FakeLLM(
        [
            "Voc? pode acessar seus arquivos nesta tela quando quiser.",
            "Podes aceder aos teus ficheiros neste ecr? quando quiseres.",
        ]
    )
    composer = ResponseComposer(llm)

    answer = composer.compose(
        ComposerRequest(
            intent="general",
            user_message="podes explicar?",
            facts=["Pedido simples."],
        )
    )

    assert answer == "Voc? pode acessar seus arquivos nesta tela quando quiser."
    assert len(llm.calls) == 1


def test_clarifying_question_fallback_uses_only_available_questions() -> None:
    composer = ResponseComposer(FakeLLM(""))

    answer = composer.compose(
        ComposerRequest(
            intent="clarifying_question",
            user_message="Ajuda-me a estudar.",
            facts=[
                "Quando é o exame?",
                "Já tens material para estudar?",
                "Quanto tempo tens?",
            ],
        )
    )

    assert answer == "Quando é o exame? Já tens material para estudar?"


def test_social_conversation_is_short_and_natural_without_llm() -> None:
    llm = FakeLLM("irrelevante")
    composer = ResponseComposer(llm)

    answer = composer.compose(
        ComposerRequest(
            intent="social_conversation",
            user_message="Olá",
            facts=["Interação social simples."],
        )
    )

    assert answer == "Olá! Como estás?"
    assert len(llm.calls) == 0


def test_social_greeting_with_relevant_content_is_not_consumed_by_greeting() -> None:
    llm = FakeLLM("Isso parece estar a pesar-te. O que aconteceu?")
    composer = ResponseComposer(llm)

    answer = composer.compose(
        ComposerRequest(
            intent="social_conversation",
            user_message="Olá, estou preocupado com uma coisa.",
            facts=["O Alexandre está preocupado."],
        )
    )

    assert answer == "Isso parece estar a pesar-te. O que aconteceu?"
    assert len(llm.calls) == 1


def test_emotional_response_is_not_prefixed_with_generic_validation() -> None:
    llm = FakeLLM(
        "Parece-me que o que te assusta não é uma função em particular. "
        "É a ideia de voltares a carregar tudo ao mesmo tempo."
    )
    composer = ResponseComposer(llm)

    answer = composer.compose(
        ComposerRequest(
            intent="support",
            user_message="Tenho medo de não conseguir lidar com tudo no próximo ano.",
            facts=["Tem várias responsabilidades acumuladas."],
        )
    )

    assert "É normal" not in answer
    assert "carregar tudo ao mesmo tempo" in answer


def test_response_composer_returns_raw_reply_before_final_voice_critic() -> None:
    llm = FakeLLM(
        [
            "Claro que sim! Estou aqui para ajudar. Qual ? a coisa que precisa de aten??o?",
            "Claro. O que se passa?",
        ]
    )
    composer = ResponseComposer(llm)

    answer = composer.compose(
        ComposerRequest(
            intent="general",
            user_message="Preciso de ajuda numa coisa.",
            facts=["Pedido simples."],
        )
    )

    assert answer == "Claro que sim! Estou aqui para ajudar. Qual ? a coisa que precisa de aten??o?"
    assert len(llm.calls) == 1

