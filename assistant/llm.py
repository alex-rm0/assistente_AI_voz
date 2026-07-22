from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import requests

from assistant.text_encoding import debug_text_encoding, has_mojibake_markers


# Single default model for the whole textual pipeline (composer, regeneration,
# and voice critic when it runs). Overridable via the OLLAMA_MODEL environment
# variable, read in app.py.
OLLAMA_MODEL = "llama3.1:8b"


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    model_source: str = "unknown"
    timeout_seconds: int = 120
    debug_performance: bool = False
    debug_ollama_payload: bool = False


class OllamaClient:
    def __init__(self, settings: OllamaSettings, system_prompt: str) -> None:
        self.settings = settings
        self.system_prompt = system_prompt
        self._chat_call_count = 0
        self._next_call_source = ""
        self._chat_call_sources: list[str] = []
        self._chat_call_tokens: list[dict[str, Any]] = []

    def mark_next_call_source(self, source: str) -> None:
        self._next_call_source = str(source or "").strip()

    @property
    def chat_call_count(self) -> int:
        return self._chat_call_count

    @property
    def chat_call_sources(self) -> list[str]:
        """Source tag recorded for each .chat() call so far, in order.

        Used by evals telemetry (evals/schemas.py TurnResult.llm_call_sources)
        to attribute LLM calls to the code path that made them without
        parsing debug stdout.
        """
        return list(self._chat_call_sources)

    @property
    def chat_call_tokens(self) -> list[dict[str, Any]]:
        """Per-call {input_tokens, output_tokens, latency_ms}, same order as chat_call_sources."""
        return list(self._chat_call_tokens)

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
        source: str | None = None,
    ) -> str:
        call_source = self._consume_call_source(source)
        system_content = system_prompt or self.system_prompt
        history_messages = history or []
        messages = [{"role": "system", "content": system_content}]
        messages.extend(history_messages)
        messages.append({"role": "user", "content": user_message})
        self._debug_chat_payload(
            source=call_source,
            system_prompt=system_content,
            history=history_messages,
            user_message=user_message,
            messages=messages,
            response_format=response_format,
        )

        return self._chat_messages(messages, response_format=response_format, source=call_source)

    def choose_tool(
        self,
        user_message: str,
        tools_description: str,
        profile_name: str | None = None,
        active_contexts: list[str] | None = None,
    ) -> dict[str, Any]:
        context_label = ", ".join(active_contexts or []) or profile_name or "desconhecido"
        system_prompt = (
            "Es um seletor de ferramentas para o AssistenteIA.\n"
            "A tua unica tarefa e decidir se a mensagem precisa de uma ferramenta de ficheiros.\n"
            f"Contextos ativos da conversa: {context_label}.\n"
            "Responde apenas em JSON valido, sem markdown.\n\n"
            "REGRAS OBRIGATORIAS:\n"
            "- So usa uma ferramenta se a mensagem pedir CLARAMENTE uma acao sobre ficheiros na workspace.\n"
            "- Perguntas sobre o assistente, sobre perfis, sobre capacidades, saudacoes, conversas gerais "
            "ou qualquer topico que nao envolva um ficheiro especifico -> SEMPRE {\"tool\": null}.\n"
            "- Perguntas sobre janelas, aplicacoes, programas, atividade do computador ou monitorizacao "
            "devem usar uma ferramenta do Context Observer quando disponivel; nunca inventar esta informacao.\n"
            "- 'list_workspace_files': so usa se o utilizador pede para listar, mostrar ou ver os ficheiros "
            "da pasta workspace.\n"
            "- 'read_workspace_file': so usa se o utilizador menciona ou implica claramente um nome de "
            "ficheiro especifico para ler ou resumir.\n"
            "- 'create_workspace_file': so usa se o utilizador pede explicitamente para criar ou guardar "
            "um ficheiro com conteudo.\n"
            "- Em caso de duvida, responde sempre com {\"tool\": null}.\n\n"
            "Formato quando usar ferramenta:\n"
            '{"tool": "nome_da_ferramenta", "arguments": {"chave": "valor"}, "reason": "motivo curto"}\n\n'
            "Formato quando nao usar ferramenta:\n"
            '{"tool": null, "arguments": {}, "reason": "motivo curto"}\n\n'
            "Ferramentas disponiveis:\n"
            f"{tools_description}\n"
        )
        raw_response = self.chat(
            user_message,
            history=[],
            system_prompt=system_prompt,
            response_format="json",
            source="TOOL_SELECTOR",
        )

        try:
            decision = json.loads(raw_response)
        except json.JSONDecodeError:
            return {"tool": None, "arguments": {}}

        if not isinstance(decision, dict):
            return {"tool": None, "arguments": {}}

        tool_name = decision.get("tool")
        arguments = decision.get("arguments", {})
        reason = decision.get("reason", "")
        if tool_name is not None and not isinstance(tool_name, str):
            tool_name = None
        if not isinstance(arguments, dict):
            arguments = {}
        if not isinstance(reason, str):
            reason = ""

        return {"tool": tool_name, "arguments": arguments, "reason": reason}

    def _chat_messages(
        self,
        messages: list[dict[str, str]],
        response_format: str | None = None,
        source: str = "OTHER",
    ) -> str:
        self._chat_call_count += 1
        started_at = time.perf_counter()
        url = f"{self.settings.base_url.rstrip('/')}/api/chat"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
        }
        if response_format is not None:
            payload["format"] = response_format

        try:
            response = requests.post(
                url,
                json=payload,
                headers={"Content-Type": "application/json; charset=utf-8"},
                timeout=self.settings.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            raise RuntimeError(
                "Nao consegui ligar ao Ollama. Confirma que o Ollama esta aberto "
                f"e que o modelo '{self.settings.model}' esta instalado."
            ) from exc
        except ValueError as exc:
            raise RuntimeError("O Ollama devolveu uma resposta invalida.") from exc

        message = data.get("message", {})
        content = message.get("content")
        elapsed_ms = (time.perf_counter() - started_at) * 1000
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("O Ollama nao devolveu texto para esta mensagem.")
        if self.settings.debug_ollama_payload and has_mojibake_markers(content):
            debug_text_encoding("ollama_raw_response", content)

        self._chat_call_sources.append(source)
        self._chat_call_tokens.append(
            {
                "input_tokens": data.get("prompt_eval_count"),
                "output_tokens": data.get("eval_count"),
                "latency_ms": elapsed_ms,
            }
        )

        if self.settings.debug_performance:
            print(f"[AssistenteIA PERF] chamada Ollama /api/chat ({source}): {elapsed_ms:.1f} ms")

        return content.strip()

    def summarize_text(self, filename: str, content: str) -> str:
        prompt = (
            "Resume o seguinte ficheiro em portugues de Portugal.\n"
            "O resumo deve ser claro, curto e fiel ao conteudo. "
            "Se fizer sentido, usa pontos principais.\n\n"
            f"Ficheiro: {filename}\n\n"
            f"Conteudo:\n{content}"
        )
        return self.chat(prompt, history=[], source="MEMORY_SUMMARY")

    def embed(self, text: str) -> list[float] | None:
        started_at = time.perf_counter()
        url = f"{self.settings.base_url.rstrip('/')}/api/embeddings"
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "prompt": text,
        }

        try:
            response = requests.post(url, json=payload, timeout=self.settings.timeout_seconds)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError):
            return None

        if self.settings.debug_performance:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            print(f"[AssistenteIA PERF] chamada Ollama /api/embeddings: {elapsed_ms:.1f} ms")

        embedding = data.get("embedding")
        if not isinstance(embedding, list):
            return None

        return [float(item) for item in embedding if isinstance(item, (int, float))]

    def _consume_call_source(self, source: str | None) -> str:
        value = str(source or self._next_call_source or "OTHER").strip() or "OTHER"
        self._next_call_source = ""
        return value

    def _debug_chat_payload(
        self,
        *,
        source: str,
        system_prompt: str,
        history: list[dict[str, str]],
        user_message: str,
        messages: list[dict[str, str]],
        response_format: str | None,
    ) -> None:
        if not self.settings.debug_ollama_payload:
            return

        payload_preview: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": False,
        }
        if response_format is not None:
            payload_preview["format"] = response_format
        payload_text = json.dumps(payload_preview, ensure_ascii=False)

        print("\n[OLLAMA PAYLOAD DEBUG]")
        print(f"timestamp={datetime.now().isoformat(timespec='seconds')}")
        print(f"source={source}")
        print(f"model={self.settings.model}")
        print(f"model_source={self.settings.model_source}")
        print(f"response_format={response_format}")
        print("\n--- system prompt completo ---")
        print(system_prompt)
        debug_text_encoding("ollama_payload_system_prompt", system_prompt)
        print("\n--- histórico completo ---")
        print(json.dumps(history, ensure_ascii=False, indent=2))
        print("\n--- mensagem atual ---")
        print(user_message)
        debug_text_encoding("ollama_payload_user_message", user_message)
        print("\n--- lista final de mensagens enviada para /api/chat ---")
        print(json.dumps(messages, ensure_ascii=False, indent=2))
        print("\n--- tamanhos ---")
        for index, message in enumerate(messages):
            role = message.get("role", "")
            content = message.get("content", "")
            print(f"messages[{index}] role={role} chars={len(content)}")
        print(f"payload_total_chars_aprox={len(payload_text)}")
        print("[/OLLAMA PAYLOAD DEBUG]\n")
