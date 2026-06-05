from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkspaceFileContent:
    filename: str
    content: str
    error: str | None = None


class WorkspaceGuard:
    """Small safety helper that restricts file operations to the workspace folder."""

    def __init__(self, workspace_path: Path, create: bool = True) -> None:
        self.workspace_path = workspace_path.resolve()
        if create:
            self.workspace_path.mkdir(parents=True, exist_ok=True)

    def resolve(self, relative_path: str = "") -> Path:
        target = (self.workspace_path / relative_path).resolve()
        if target != self.workspace_path and self.workspace_path not in target.parents:
            raise ValueError("Access outside the workspace folder is not allowed.")
        return target
