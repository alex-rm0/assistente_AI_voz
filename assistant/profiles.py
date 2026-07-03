from __future__ import annotations

from dataclasses import dataclass

from assistant.prompts import DEFAULT_PROFILE_NAME, get_system_prompt, profile_names


@dataclass(frozen=True)
class Profile:
    name: str
    system_prompt: str


def get_profile(name: str) -> Profile:
    profile_name = name if name in profile_names() else DEFAULT_PROFILE_NAME
    return Profile(name=profile_name, system_prompt=get_system_prompt(profile_name))
