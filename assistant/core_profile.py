"""Alexandre's core profile: a short private file that is always in the composer prompt.

North Star §5 (small core profile, always present) and §6 (privacy classes).
The file lives outside git (data/profile/core_profile.md). It is read again
whenever it changes on disk, so it can be edited without restarting Echo. A
missing or unreadable file is not an error for the caller: Echo answers as it
did before the profile existed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from assistant.response_language import ENGLISH_GB, normalize_response_language

ENV_VAR = "ECHO_CORE_PROFILE_PATH"
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "profile" / "core_profile.md"
MAX_CHARS = 3000
# North Star §6. The model router does not check classes yet; this is the
# class it will see for anything derived from the profile.
PRIVACY_CLASS = "PERSONAL"


@dataclass(frozen=True)
class CoreProfile:
    text: str
    privacy_class: str = PRIVACY_CLASS


_cached_key: tuple[str, int, int] | None = None
_cached_profile: CoreProfile | None = None
_last_logged: tuple[str, str] | None = None


def profile_path() -> Path:
    override = os.environ.get(ENV_VAR, "").strip()
    return Path(override) if override else DEFAULT_PATH


def _log(path: Path, status: str, detail: str = "") -> None:
    # Once per path and status, not once per turn. Never the content.
    global _last_logged
    if _last_logged == (str(path), status):
        return
    _last_logged = (str(path), status)
    print(f"[CORE_PROFILE] status={status}" + (f" {detail}" if detail else ""))


def load_core_profile() -> CoreProfile | None:
    """The current profile, or None when there is none to use. Never raises."""
    global _cached_key, _cached_profile
    path = profile_path()
    try:
        stat = path.stat()
    except FileNotFoundError:
        _cached_key = _cached_profile = None
        _log(path, "missing")
        return None
    except OSError as exc:
        _cached_key = _cached_profile = None
        _log(path, "error", f"type={type(exc).__name__}")
        return None

    key = (str(path), stat.st_mtime_ns, stat.st_size)
    if key == _cached_key:
        return _cached_profile

    try:
        text = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        _cached_key, _cached_profile = key, None
        _log(path, "error", f"type={type(exc).__name__}")
        return None

    if not text:
        _cached_key, _cached_profile = key, None
        _log(path, "missing", "reason=empty")
        return None

    if len(text) > MAX_CHARS:
        _log(path, "truncated", f"chars={len(text)} kept={MAX_CHARS}")
        text = text[:MAX_CHARS].rstrip()
    else:
        _log(path, "loaded", f"chars={len(text)} privacy_class={PRIVACY_CLASS}")
    _cached_key, _cached_profile = key, CoreProfile(text=text)
    return _cached_profile


_SECTION_EN = """
About Alexandre (private notes, only for you):
{profile}

Use these notes silently to answer better. Never list or recite them unless Alexandre asks for that.
Mention a fact from these notes only when it is relevant to what he asked.
These notes do not authorise inventing anything beyond what is written in them.
The notes are written in Portuguese. When you use a fact from them, say it in English: translate degrees, subjects, roles and descriptions, and never copy Portuguese wording into your reply. Names of people, places and institutions may stay as written.
""".strip()

_SECTION_PT = """
Sobre o Alexandre (notas privadas, só para ti):
{profile}

Usa estas notas em silêncio para responderes melhor. Nunca as listes nem as recites sem o Alexandre pedir.
Menciona um facto destas notas só quando for relevante para o que ele perguntou.
Estas notas não te autorizam a inventar nada para além do que lá está escrito.
""".strip()


def about_alexandre_section(response_language: str) -> str:
    """The system-prompt section for the composer; "" when there is no profile."""
    profile = load_core_profile()
    if profile is None:
        return ""
    template = _SECTION_EN if normalize_response_language(response_language) == ENGLISH_GB else _SECTION_PT
    return template.format(profile=profile.text)
