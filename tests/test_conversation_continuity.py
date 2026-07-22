from __future__ import annotations

from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.personal_model import PersonalModel
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import ToolRegistry


class ContinuityLLM:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.histories: list[list[dict[str, str]]] = []
        self.user_messages: list[str] = []

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        self.histories.append(list(history or []))
        self.user_messages.append(user_message)
        if "Estou mais ou menos" in user_message:
            return "Lamento. Só tens mais alguma oportunidade?"
        if "Foi mau" in user_message:
            assert any("chumbei a um exame importante" in item["content"] for item in history or [])
            return "Percebo. Foi mau no exame, então."
        if "Foi dif" in user_message:
            return "O exame correu mal e isso pesa. Faz sentido que esteja a custar."
        if "Só tenho mais uma oportunidade" in user_message:
            assert any("Chumbei a um exame" in item["content"] for item in history or [])
            return "Então essa oportunidade é mesmo para repetir o exame."
        if "Já leste?" in user_message:
            assert any("Leste a minha mensagem" in item["content"] for item in history or [])
            return "Já. Tinha lido mal a tua mensagem anterior."
        return "Não percebi bem. Falta-me contexto."

    def embed(self, text: str):
        return None


class VoiceReviewLLM:
    def __init__(self) -> None:
        self.chat_calls = 0
        self.sources: list[str] = []

    def choose_tool(self, user_message, tools_description, profile_name=None, active_contexts=None):
        return {"tool": None, "arguments": {}, "reason": "sem ferramenta"}

    def chat(self, user_message, history=None, system_prompt=None, response_format=None):
        self.chat_calls += 1
        self.sources.append(getattr(self, "_next_call_source", "OTHER"))
        if len(self.sources) == 1:
            return "Você pode acessar seus arquivos nesta tela quando quiser."
        return "Podes aceder aos teus ficheiros neste ecrã quando quiseres."

    def embed(self, text: str):
        return None


def make_engine(tmp_path: Path) -> tuple[AssistantEngine, ContinuityLLM]:
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    llm = ContinuityLLM()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        personal_model=PersonalModel(data),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    return engine, llm


def make_engine_with_llm(tmp_path: Path, llm):
    data = tmp_path / "data"
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    engine = AssistantEngine(
        llm=llm,
        memory=ConversationMemory(data, "history.json", 20),
        long_term_memory=LongTermMemory(data, embedder=llm),
        personal_model=PersonalModel(data),
        tools=ToolRegistry(),
        workspace_path=workspace,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
    )
    return engine


def test_short_followup_keeps_exam_context(tmp_path: Path) -> None:
    engine, llm = make_engine(tmp_path)

    first = engine.respond("Estou mais ou menos, chumbei a um exame importante.")
    second = engine.respond("Foi mau.")

    assert first == "Lamento. Só tens mais alguma oportunidade?"
    assert second == "Percebo. Foi mau no exame, então."
    assert any("chumbei a um exame importante" in item["content"] for item in llm.histories[-1])
    forbidden = ("chefe", "reencontro", "trabalho")
    assert not any(word in second.lower() for word in forbidden)


def test_voice_critic_is_not_used_for_normal_style_cleanup(tmp_path: Path) -> None:
    llm = VoiceReviewLLM()
    engine = make_engine_with_llm(tmp_path, llm)

    answer = engine.respond("podes explicar?")

    assert "tu" in answer or "teus" in answer
    assert "Você" not in answer
    assert llm.sources == ["RESPONSE_COMPOSER"]


def test_simple_factual_question_uses_one_llm_call(tmp_path: Path) -> None:
    class OneReplyLLM(ContinuityLLM):
        def chat(self, user_message, history=None, system_prompt=None, response_format=None):
            self.chat_calls += 1
            self.histories.append(list(history or []))
            self.user_messages.append(user_message)
            return (
                "Estratégias Algorítmicas é uma área dedicada à conceção e análise "
                "de métodos para resolver problemas de forma eficiente."
            )

    llm = OneReplyLLM()
    engine = make_engine_with_llm(tmp_path, llm)

    answer = engine.respond("O que sabes sobre Estratégias Algorítmicas?")

    assert "Estratégias Algorítmicas" in answer
    assert llm.chat_calls == 1


def test_topic_shift_limits_history_sent_to_llm(tmp_path: Path) -> None:
    class TopicLLM(ContinuityLLM):
        def chat(self, user_message, history=None, system_prompt=None, response_format=None):
            self.chat_calls += 1
            self.histories.append(list(history or []))
            self.user_messages.append(user_message)
            assert "Picasso" not in " ".join(item["content"] for item in history or [])
            return "Estratégias Algorítmicas estuda formas eficientes de resolver problemas."

    llm = TopicLLM()
    engine = make_engine_with_llm(tmp_path, llm)
    engine.memory.append_pair("Pesquisa sobre Picasso.", "Ainda não tenho uma ferramenta de pesquisa ligada.")

    shift = engine.respond("Queria falar sobre outro tema.")
    answer = engine.respond("O que sabes sobre Estratégias Algorítmicas?")

    assert "Que tema" in shift
    assert "Estratégias Algorítmicas" in answer


def test_como_assim_explains_previous_assistant_phrase(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)
    engine.memory.append_pair("Teste", "A frase anterior estava confusa.")

    answer = engine.respond("Como assim?")

    assert "frase anterior estava confusa" in answer


def test_one_more_opportunity_refers_to_exam(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)

    engine.respond("Chumbei a um exame.")
    answer = engine.respond("Só tenho mais uma oportunidade.")

    assert "repetir o exame" in answer


def test_reference_to_reading_previous_message_is_kept(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)

    first = engine.respond("Leste a minha mensagem?")
    second = engine.respond("Já leste?")

    assert "Li mal" in first or "li mal" in first
    assert "mensagem anterior" in second


def test_short_bad_result_without_history_does_not_invent(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)

    answer = engine.respond("Foi mau.")

    assert (
        "Falta-me contexto" in answer
        or "Não percebi bem" in answer
        or "Diz-me um pouco melhor" in answer
    )
    assert "chefe" not in answer.lower()
    assert "reencontro" not in answer.lower()


def test_exam_negative_result_stays_short_and_pt_pt(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)

    answer = engine.respond("Foi difícil, o exame correu mal.")

    forbidden = ("estressado", "puxa", "te saiu", "compartilhar", "sientas", "prova")
    assert "exame correu mal" in answer
    assert answer.count("?") <= 1
    assert not any(word in answer.lower() for word in forbidden)


def test_does_not_ask_user_to_rewrite_echo_phrase(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)
    engine.memory.append_pair("Teste", "A frase do Echo estava pouco clara.")

    answer = engine.respond("Não percebi o que querias dizer.")

    assert "frase do Echo estava pouco clara" in answer
    assert "reescrev" not in answer.lower()


def test_exam_typo_falar_is_clarified_as_falhar(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)
    engine.respond("Estou mais ou menos, chumbei a um exame importante.")

    answer = engine.respond("Agora só tenho mais uma oportunidade e tenho medo de falar.")

    assert "medo de falhar" in answer
    assert "última oportunidade" in answer or "ultima oportunidade" in answer


def test_no_much_to_talk_about_does_not_insist(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)

    answer = engine.respond("Não sei se há muito a falar...")

    assert "Talvez não haja" in answer
    assert answer.count("?") == 0


def test_failure_feeling_is_not_turned_into_two_questions(tmp_path: Path) -> None:
    engine, _llm = make_engine(tmp_path)

    answer = engine.respond("Sinto que fui um fracasso, estudei para falhar.")

    assert "não transforma todo o trabalho num fracasso" in answer
    assert answer.count("?") == 0
