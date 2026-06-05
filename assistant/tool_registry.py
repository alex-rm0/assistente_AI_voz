from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


ToolFunction = Callable[..., str]


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    function: ToolFunction
    permissions: tuple[str, ...]
    remember_result: bool = True

    def run(self, arguments: dict[str, Any]) -> str:
        # Ignore extra arguments returned by the LLM instead of passing unknown
        # values into a tool function.
        signature = inspect.signature(self.function)
        accepted_arguments = {
            name: value
            for name, value in arguments.items()
            if name in signature.parameters
        }
        try:
            return self.function(**accepted_arguments)
        except TypeError as exc:
            return f"Nao consegui executar a ferramenta '{self.name}': argumentos invalidos ({exc})."

    def describe(self) -> str:
        permissions = ", ".join(self.permissions)
        return f"- {self.name}: {self.description} Permissoes: {permissions}."


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(
        self,
        name: str,
        description: str,
        permissions: tuple[str, ...],
        remember_result: bool = True,
    ) -> Callable[[ToolFunction], ToolFunction]:
        def decorator(function: ToolFunction) -> ToolFunction:
            # Importing a module with decorated functions is enough to register
            # its tools in the global registry.
            self._tools[name] = Tool(
                name=name,
                description=description,
                function=function,
                permissions=permissions,
                remember_result=remember_result,
            )
            return function

        return decorator

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def list(self) -> list[Tool]:
        return list(self._tools.values())

    def describe(self) -> str:
        if not self._tools:
            return "Nao ha ferramentas disponiveis."

        return "\n".join(tool.describe() for tool in self.list())


tool_registry = ToolRegistry()
