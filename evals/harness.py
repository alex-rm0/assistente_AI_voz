"""Isolated AssistantEngine construction for evals.

Every case gets its own fresh engine backed by a throwaway directory, never
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
# here so every eval engine sees the same tool set app.py wires up.
import assistant.tools  # noqa: F401


@dataclass
class ProviderConfig:
    provider: str = "ollama"
    model: str = ""
    base_url: str = "http://127.0.0.1:11434"
    model_source: str = "provider:ollama"


def build_llm(config: ProviderConfig):
    """Turn a --provider/--model choice into the LLM object used by evals."""
    from assistant.anthropic_provider import AnthropicProvider
    from assistant.model_provider import DEFAULT_ANTHROPIC_MODEL, DEFAULT_OLLAMA_MODEL, OllamaProvider, ProviderBackedLLM

    if config.provider == "ollama":
        provider = OllamaProvider(model=config.model or DEFAULT_OLLAMA_MODEL, base_url=config.base_url)
    elif config.provider == "anthropic":
        provider = AnthropicProvider(model=config.model or DEFAULT_ANTHROPIC_MODEL)
    else:
        raise ValueError(f"Provider '{config.provider}' nao esta implementado.")
    return ProviderBackedLLM(provider, system_prompt=get_base_system_prompt(), model_source=config.model_source)


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
    # Always on for evals. This drives AssistantEngine.get_last_turn_telemetry()
    # and never touches the user's real config/settings.json.
    engine.debug_ollama_payload = True
    return engine


def apply_setup_steps(engine: AssistantEngine, setup: tuple[dict, ...]) -> None:
    """Seed conversation/memory state before the graded turns run."""
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
