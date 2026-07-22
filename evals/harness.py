"""Isolated AssistantEngine construction for evals.

Every case gets its own fresh engine backed by a throwaway directory — never
the user's real `data/` folder. See section 2.2 of the eval spec: isolation
is via ECHO_ENV=test / ECHO_TEST_DATA_DIR, not by mocking the memory layer.
"""

from __future__ import annotations

import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from assistant.conversation import AssistantEngine
from assistant.long_term_memory import LongTermMemory
from assistant.memory import ConversationMemory
from assistant.presence_manager import PresenceManager
from assistant.prompts import get_base_system_prompt
from assistant.tool_registry import tool_registry

# Registers the built-in tools onto tool_registry as a side effect. Imported
# here (not left to the caller) so every eval engine sees the same tool set
# app.py wires up — a case that types "pesquisa sobre Picasso" should react
# exactly like the real app would.
import assistant.tools  # noqa: F401


@dataclass
class ProviderConfig:
    provider: str = "ollama"
    model: str = ""
    base_url: str = "http://127.0.0.1:11434"


def build_llm(config: ProviderConfig):
    """Single seam that turns a --provider/--model choice into an llm object.

    Goes through assistant.model_provider.ModelProvider (Part 3) rather than
    assistant.llm.OllamaClient directly, so evals actually exercise the same
    provider abstraction future providers (Anthropic, OpenAI) would slot
    into — adding one only ever means adding a branch here.
    """
    if config.provider != "ollama":
        raise ValueError(
            f"Provider '{config.provider}' ainda não está implementado. "
            "Só 'ollama' está disponível nesta tarefa (ver Parte 3 do pedido: "
            "não integrar Anthropic/OpenAI ainda)."
        )
    from assistant.llm import OLLAMA_MODEL
    from assistant.model_provider import OllamaProvider, ProviderBackedLLM

    provider = OllamaProvider(model=config.model or OLLAMA_MODEL, base_url=config.base_url)
    return ProviderBackedLLM(provider, system_prompt=get_base_system_prompt())


class EvalRun:
    """Owns the run-level temp root; each case gets its own subdirectory under it."""

    def __init__(self, keep_data: bool = False) -> None:
        explicit = os.environ.get("ECHO_TEST_DATA_DIR")
        self._owns_root = explicit is None
        self.root = Path(explicit) if explicit else Path(tempfile.mkdtemp(prefix="echo_evals_"))
        self.root.mkdir(parents=True, exist_ok=True)
        self.keep_data = keep_data
        os.environ["ECHO_ENV"] = "test"
        os.environ.setdefault("ECHO_TEST_DATA_DIR", str(self.root))

    def case_dir(self, case_id: str) -> Path:
        safe_id = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in case_id)
        path = self.root / safe_id
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def cleanup(self) -> None:
        if self.keep_data:
            return
        if self._owns_root:
            shutil.rmtree(self.root, ignore_errors=True)


def build_engine(case_dir: Path, config: ProviderConfig) -> AssistantEngine:
    data_path = case_dir / "data"
    workspace_path = case_dir / "workspace"
    workspace_path.mkdir(parents=True, exist_ok=True)
    llm = build_llm(config)
    memory = ConversationMemory(data_path, "history.json", 20)
    long_term_memory = LongTermMemory(data_path, embedder=llm)
    engine = AssistantEngine(
        llm=llm,
        memory=memory,
        long_term_memory=long_term_memory,
        tools=tool_registry,
        workspace_path=workspace_path,
        base_system_prompt=get_base_system_prompt(),
        presence_manager=PresenceManager(),
        known_projects={"AssistenteIA": "."},
        desktop_config={},
    )
    # Always on for evals regardless of the provider/model — this is what
    # drives AssistantEngine.get_last_turn_telemetry() (see conversation.py);
    # it never touches the user's real config/settings.json.
    engine.debug_ollama_payload = True
    return engine


def apply_setup_steps(engine: AssistantEngine, setup: tuple[dict, ...]) -> None:
    """Seeds conversation/memory state before the graded turns run.

    Two step kinds:
    - {"say": "..."}: a real conversational turn (drives passive extraction,
      history, etc.) whose result is discarded — not graded.
    - {"fact_type": "...", "fields": {...}}: writes a structured fact
      directly via long_term_memory, bypassing the LLM entirely — for
      seeding memory deterministically without depending on extraction
      regexes picking up the setup phrasing.
    """
    for step in setup:
        if "say" in step:
            engine.respond(str(step["say"]))
        elif "fact_type" in step:
            engine.long_term_memory.remember_structured_fact(
                str(step["fact_type"]),
                dict(step.get("fields", {})),
                confidence=float(step.get("confidence", 0.9)),
            )


def timed_respond(engine: AssistantEngine, user_message: str) -> tuple[str, float]:
    started = time.perf_counter()
    response = engine.respond(user_message)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return response, elapsed_ms
