from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any

import requests


@dataclass(frozen=True)
class OllamaSettings:
    base_url: str
    model: str
    timeout_seconds: int = 120
    debug_performance: bool = False


class OllamaClient:
    def __init__(self, settings: OllamaSettings, system_prompt: str) -> None:
        self.settings = settings
        self.system_prompt = system_prompt

    def chat(
        self,
        user_message: str,
        history: list[dict[str, str]] | None = None,
        system_prompt: str | None = None,
        response_format: str | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt or self.system_prompt}]
        messages.extend(history or [])
        messages.append({"role": "user", "content": user_message})

        return self._chat_messages(messages, response_format=response_format)

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

    def _chat_messages(self, messages: list[dict[str, str]], response_format: str | None = None) -> str:
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
        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("O Ollama nao devolveu texto para esta mensagem.")

        if self.settings.debug_performance:
            elapsed_ms = (time.perf_counter() - started_at) * 1000
            print(f"[AssistenteIA PERF] chamada Ollama /api/chat: {elapsed_ms:.1f} ms")

        return content.strip()

    def summarize_text(self, filename: str, content: str) -> str:
        prompt = (
            "Resume o seguinte ficheiro em portugues de Portugal.\n"
            "O resumo deve ser claro, curto e fiel ao conteudo. "
            "Se fizer sentido, usa pontos principais.\n\n"
            f"Ficheiro: {filename}\n\n"
            f"Conteudo:\n{content}"
        )
        return self.chat(prompt, history=[])

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
