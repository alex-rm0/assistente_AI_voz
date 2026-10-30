"""Stabilization battery scenario 13 latency: get_active_window took ~35 s,
~27 s of it in _git_repositories, because both git scans walked the whole
project tree (~199k entries, almost all .venv*/.uv_* caches) and found a
fake repo inside .uv_cache. Virtualenvs and caches must be pruned.
"""
from __future__ import annotations

from pathlib import Path

from assistant.context_observer import _git_repositories


def _repo(path: Path, branch: str = "main") -> None:
    (path / ".git").mkdir(parents=True)
    (path / ".git" / "HEAD").write_text(f"ref: refs/heads/{branch}\n", encoding="utf-8")


def _file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")


def _tree(tmp_path: Path) -> Path:
    root = tmp_path / "assistenteIA"
    _repo(root)
    _file(root / "assistant" / "agent.py")
    _repo(root / "plugins" / "inner", branch="dev")
    _file(root / "plugins" / "inner" / "inner.py")
    for ignored in (
        ".venv/Lib/site-packages/pkg/mod.py",
        ".venv_tts/Lib/site-packages/tts.py",
        ".venv.broken-msstore-20260722/Lib/x.py",
        ".uv_cache/archive/y.py",
        ".uv_python/cpython/z.py",
        ".hf_cache/models/w.bin",
        "node_modules/pkg/index.js",
        "assistant/__pycache__/agent.cpython-312.pyc",
        "data/history.json",
    ):
        _file(root / ignored)
    _repo(root / ".uv_cache" / "sdists-v9")
    _repo(root / ".venv" / "src" / "vendored")
    return root


def test_git_scan_skips_repos_inside_virtualenvs_and_caches(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    repos = {repo.path: repo.branch for repo in _git_repositories(root)}

    assert repos == {"assistenteIA": "main", "inner": "dev"}


def test_git_modified_files_skip_virtualenvs_and_caches(tmp_path: Path) -> None:
    root = _tree(tmp_path)

    repos = {repo.path: repo for repo in _git_repositories(root)}
    modified = repos["assistenteIA"].modified_files

    assert "assistant/agent.py" in modified
    ignored_roots = (".venv", ".uv_", ".hf_cache", "node_modules", "data/", ".git/")
    assert not [name for name in modified if name.startswith(ignored_roots)], modified
    assert not [name for name in modified if "__pycache__" in name], modified
    assert repos["inner"].modified_files == ("inner.py",)
