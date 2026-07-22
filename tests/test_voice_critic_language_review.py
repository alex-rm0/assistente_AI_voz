from __future__ import annotations

from assistant.response_composer import ComposerRequest
from assistant.voice_critic import _review_trigger, _revision_is_faithful


def request(message: str = "teste") -> ComposerRequest:
    return ComposerRequest(intent="general_conversation", user_message=message)


# Teste A - correção linguística legítima (idiom brasileiro -> europeu)
def test_teste_a_idiom_correction_is_accepted() -> None:
    accepted, reason = _revision_is_faithful(
        "O que achas que te deu errado?",
        "O que achas que correu mal?",
    )
    assert accepted, reason


# Teste B - colocação pronominal
def test_teste_b_pronoun_placement_correction_is_accepted() -> None:
    accepted, reason = _revision_is_faithful(
        "Podes me explicar isso?",
        "Podes explicar-me isso?",
    )
    assert accepted, reason


# Teste C - tratamento (você -> tu) com mudança de conjugação/sinónimo
def test_teste_c_formality_correction_is_accepted() -> None:
    accepted, reason = _revision_is_faithful(
        "Entendo que se sinta assim.",
        "Percebo que te sintas assim.",
    )
    assert accepted, reason


# Teste D - fidelidade: não pode acrescentar conselho/plano
def test_teste_d_new_advice_is_rejected() -> None:
    accepted, reason = _revision_is_faithful(
        "Parece que estudaste bastante.",
        "Parece que estudaste bastante, por isso deves tentar novamente amanhã.",
    )
    assert not accepted
    assert "motiva" in reason or "conselho" in reason


# Teste E - não pode acrescentar emoção nova
def test_teste_e_new_emotion_is_rejected() -> None:
    accepted, reason = _revision_is_faithful(
        "Isso foi difícil.",
        "Isso foi difícil e deixou-te muito triste.",
    )
    assert not accepted
    assert "emo" in reason


# Teste F - não pode acrescentar pergunta nova
def test_teste_f_new_question_is_rejected() -> None:
    accepted, reason = _revision_is_faithful(
        "Isso deve ter sido frustrante.",
        "Isso deve ter sido frustrante. Queres falar sobre isso?",
    )
    assert not accepted
    assert "pergunta" in reason


# Teste G - troca de sujeito (Echo passa a falar da própria experiência)
def test_teste_g_subject_swap_is_rejected() -> None:
    accepted, reason = _revision_is_faithful(
        "Deve ter sido difícil para ti.",
        "Fiquei muito triste com o resultado.",
    )
    assert not accepted
    assert reason == "mudou a experiência do utilizador para o Echo"


# Teste H - resposta já correta não deve ser alterada / não precisa de crítico
def test_teste_h_already_correct_reply_has_no_trigger() -> None:
    trigger = _review_trigger("Percebo que estejas frustrado.", request())
    assert trigger == ""

    accepted, reason = _revision_is_faithful(
        "Percebo que estejas frustrado.",
        "Percebo que estejas frustrado.",
    )
    assert accepted, reason


# Gatilhos linguísticos objetivos que devem sobreviver à simplificação do pipeline.
def test_new_language_markers_trigger_review() -> None:
    for reply in (
        "Você acha isso normal?",
        "Entendo que se sinta assim.",
        "O que te deu errado?",
        "Isso está relacionado ao exame.",
        "Podes me dizer mais sobre isso?",
        "Me diga o que se passou.",
        "Me conta o que aconteceu.",
        "Vai te ajudar a perceber melhor.",
    ):
        trigger = _review_trigger(reply, request())
        assert trigger, f"esperava gatilho para: {reply}"
