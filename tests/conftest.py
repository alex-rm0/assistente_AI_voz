from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _no_real_core_profile(tmp_path, monkeypatch):
    """The real data/profile/core_profile.md is private and changes every
    composer payload. No test may read it: point the loader at a path that
    does not exist. Tests that need a profile set the variable themselves."""
    monkeypatch.setenv("ECHO_CORE_PROFILE_PATH", str(tmp_path / "no_core_profile.md"))
