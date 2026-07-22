from __future__ import annotations

from assistant.response_composer import ComposerRequest
from assistant.voice_critic import (
    VoiceCritic,
    _has_too_many_questions_for_casual,
    _review_trigger,
    _revision_is_faithful,
    detect_semantic_conflict,
    detect_subject_swap,
)


class CriticLLM:
    def __init__(self, reviewed: str) -> None:
        self.reviewed = reviewed
        self.chat_calls = 0
        self.last_user_message = ""
        self.last_history = []

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        self.last_user_message = user_message
        self.last_history = list(history or [])
        return self.reviewed


def request(message: str = "teste") -> ComposerRequest:
    return ComposerRequest(intent="conversa_normal", user_message=message)


def test_voice_critic_accepts_small_form_review() -> None:
    critic = VoiceCritic(CriticLLM("Estás cansado?"))

    trace = critic.review_with_trace("Você está cansado?", request())

    assert trace.final_response == "Estás cansado?"
    assert trace.changed
    assert trace.accepted


def test_correct_tu_form_does_not_trigger_review() -> None:
    # "Tu estás cansado?" is correct European Portuguese; a redundant subject
    # pronoun is not an objective error and must not call the Voice Critic.
    critic = VoiceCritic(CriticLLM("nao devia ser chamado"))

    trace = critic.review_with_trace("Tu estás cansado?", request())

    assert trace.final_response == "Tu estás cansado?"
    assert not trace.changed


def test_voice_critic_keeps_equivalent_question() -> None:
    critic = VoiceCritic(CriticLLM("Queres falar sobre isso?"))

    trace = critic.review_with_trace("Queres falar sobre isso?", request())

    assert trace.final_response == "Queres falar sobre isso?"
    assert trace.accepted


def test_voice_critic_rejects_new_event_and_person() -> None:
    accepted, reason = _revision_is_faithful("Foi um dia mau?", "O dia foi mau porque encontraste o teu antigo chefe.")

    assert not accepted


def test_voice_critic_does_not_duplicate_history_in_user_prompt() -> None:
    llm = CriticLLM("Podes aceder aos teus ficheiros neste ecrã quando quiseres.")
    critic = VoiceCritic(llm)
    history = [{"role": "user", "content": "conteudo unico do historico"}]

    trace = critic.review_with_trace(
        "Você pode acessar seus arquivos nesta tela quando quiser.",
        request(),
        history=history,
    )

    assert trace.final_response == "Podes aceder aos teus ficheiros neste ecrã quando quiseres."
    assert llm.last_history == history
    assert "conteudo unico do historico" not in llm.last_user_message


def test_voice_critic_telemetry_separates_rejected_review_from_final_change() -> None:
    critic = VoiceCritic(CriticLLM("Podes aceder aos ficheiros com o professor Carlos."))

    trace = critic.review_with_trace("Você pode acessar seus arquivos.", request())

    assert trace.review_changed is True
    assert trace.review_accepted is False
    assert trace.final_response_changed is False


def test_sentence_initial_words_are_not_proper_names() -> None:
    accepted, reason = _revision_is_faithful(
        "Acho que isso pesa.",
        "Sim, acho que isso pesa.",
    )

    assert accepted, reason

    accepted, reason = _revision_is_faithful(
        "Podes falar disso aos poucos.",
        "Entendo. Podes falar disso aos poucos.",
    )

    assert accepted, reason


def test_real_new_name_is_still_rejected() -> None:
    accepted, reason = _revision_is_faithful(
        "Isso parece pesado.",
        "Isso parece pesado por causa do professor Carlos.",
    )

    assert not accepted
    assert "Carlos" in reason
    assert "introduziu" in reason


def test_voice_critic_rejects_new_subject() -> None:
    accepted, _reason = _revision_is_faithful("Não percebi bem.", "Não percebi bem o problema com o trabalho.")

    assert not accepted


def test_voice_critic_rejects_echo_lived_experience() -> None:
    accepted, _reason = _revision_is_faithful(
        "Estás preocupado com o exame?",
        "Eu vivi uma situação parecida e fiquei preocupado com o exame.",
    )

    assert not accepted


# --- Section 1: novos gatilhos ---------------------------------------------


def test_new_semantic_markers_trigger_review() -> None:
    # Only objective cases remain: emotional inversion and explicit Brazilian
    # vocabulary/pronoun placement. Softer stylistic phrasing (therapeutic
    # language, "talvez possamos", generic follow-ups) is intentionally
    # tolerated and no longer calls the Voice Critic.
    negative_request = request("Estou mais ou menos, chumbei a um exame importante.")
    for reply in (
        "Que alívio! Fico contente por teres partilhado isso.",
        "Deves estar muito estressado com isso.",
        "Podes me dizer o que aconteceu?",
    ):
        trigger = _review_trigger(reply, negative_request)
        assert trigger, f"esperava gatilho para: {reply}"


def test_stylistic_phrasing_no_longer_triggers_review() -> None:
    negative_request = request("Estou mais ou menos, chumbei a um exame importante.")
    for reply in (
        "Sinto muito saber que isso aconteceu.",
        "Isto pode ser um ponto de inflexão para ti.",
        "Vamos encontrar um ponto de partida juntos.",
        "Como te sentiu com o resultado?",
        "Que foi que se passou ao certo?",
        "Após o exame, como te sentes?",
        "Foi um choque para ti, não foi?",
        "Talvez possamos falar sobre isso amanhã.",
    ):
        trigger = _review_trigger(reply, negative_request)
        assert not trigger, f"não esperava gatilho para: {reply}"


# --- Section 2: conflito semântico (inversão emocional) --------------------


def test_detect_semantic_conflict_flags_inverted_emotion() -> None:
    reason = detect_semantic_conflict(
        "Estou mais ou menos, chumbei a um exame importante.",
        "Que alívio! Estou contente por teres partilhado isso.",
    )

    assert reason


def test_detect_semantic_conflict_is_silent_without_negative_message() -> None:
    reason = detect_semantic_conflict("Correu tudo bem no fim de semana.", "Que bom! Fico contente.")

    assert reason == ""


def test_review_trigger_labels_semantic_conflict() -> None:
    trigger = _review_trigger(
        "Que alívio! Estou contente por ter sido convidado para o teu apoio nesse momento.",
        ComposerRequest(intent="general_conversation", user_message="Estou mais ou menos, chumbei a um exame importante."),
    )

    assert trigger.startswith("conflito_semantico")


# --- Section 3: troca de sujeito -------------------------------------------


def test_detect_subject_swap_flags_echo_speaking_as_user() -> None:
    for reply in (
        "Fiquei um pouco surpreendido com o resultado.",
        "Estou mais assustado do que chateado.",
        "Espero poder melhorar para a próxima.",
        "Estou contente por ter sido convidado para o teu apoio nesse momento.",
        "Passei por isso também.",
        "Quando fiz o exame, senti-me da mesma forma.",
    ):
        assert detect_subject_swap(reply), f"esperava troca de sujeito para: {reply}"


def test_revision_rejected_for_subject_swap_has_expected_reason() -> None:
    accepted, reason = _revision_is_faithful(
        "Estás preocupado com o resultado?",
        "Fiquei um pouco surpreendido com o resultado, mas percebo que te custe.",
    )

    assert not accepted
    assert reason == "mudou a experiência do utilizador para o Echo"


# --- Section 4: uma única pergunta -----------------------------------------


def test_general_conversation_reply_with_two_questions_triggers_review() -> None:
    reply = "Como foi o exame? Foi difícil ou estavas preparado?"
    triggered = _has_too_many_questions_for_casual(
        reply, ComposerRequest(intent="general_conversation", user_message="Foi difícil, mas correu mal.")
    )

    assert triggered


def test_revision_keeping_two_questions_is_rejected() -> None:
    accepted, reason = _revision_is_faithful(
        "Como foi o exame? Foi difícil ou estavas preparado?",
        "Como correu o exame? Sentiste-te preparado? Foi mais difícil do que esperavas?",
    )

    assert not accepted
    assert "pergunta" in reason
