from __future__ import annotations

from assistant.fast_router import route_fast_command


def test_fast_router_opens_known_url_aliases() -> None:
    route = route_fast_command("abre o youtube")

    assert route is not None
    assert route.kind == "tool"
    assert route.tool_name == "open_url"
    assert route.arguments == {"url": "https://www.youtube.com"}


def test_fast_router_normalizes_bare_domains() -> None:
    route = route_fast_command("abre www.youtube.com")

    assert route is not None
    assert route.arguments == {"url": "https://www.youtube.com"}


def test_fast_router_keeps_explicit_https_url() -> None:
    route = route_fast_command("abre https://www.google.com")

    assert route is not None
    assert route.arguments == {"url": "https://www.google.com"}


def test_fast_router_handles_clear_conversation() -> None:
    route = route_fast_command("limpar conversa")

    assert route is not None
    assert route.kind == "clear_conversation"


def test_fast_router_ignores_dangerous_shell_request() -> None:
    assert route_fast_command("executa powershell") is None


def test_fast_router_blocks_dangerous_open_targets() -> None:
    messages = (
        "abre file:///C:/Windows/System32/cmd.exe",
        "abre javascript:alert(1)",
        "abre data:text/html,<script>alert(1)</script>",
        "abre C:\\Windows\\System32",
        "abre powershell",
        "abre cmd",
        "abre terminal",
        "abre programa.exe",
        "abre script.ps1",
        "abre ../ficheiro",
    )

    for message in messages:
        route = route_fast_command(message)
        assert route is not None
        assert route.kind == "denied"
        assert route.tool_name is None
        assert route.arguments is None


def test_fast_router_still_allows_safe_url_targets() -> None:
    examples = (
        ("abre youtube", "https://www.youtube.com"),
        ("abre o google", "https://www.google.com"),
        ("abre www.youtube.com", "https://www.youtube.com"),
        ("abre https://www.google.com", "https://www.google.com"),
    )

    for message, url in examples:
        route = route_fast_command(message)
        assert route is not None
        assert route.kind == "tool"
        assert route.tool_name == "open_url"
        assert route.arguments == {"url": url}
