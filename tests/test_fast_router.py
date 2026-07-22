from __future__ import annotations

import json

from assistant.fast_router import load_quick_sites, route_fast_command


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
        ("abrir youtube", "https://www.youtube.com"),
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


def test_fast_router_loads_quick_sites_from_config_file(tmp_path) -> None:
    config = tmp_path / "quick_sites.json"
    config.write_text(
        json.dumps({"quick_sites": {"docs": "https://docs.python.org"}}),
        encoding="utf-8",
    )

    sites = load_quick_sites(config)
    route = route_fast_command("abre docs", quick_sites=sites)

    assert route is not None
    assert route.kind == "tool"
    assert route.arguments == {"url": "https://docs.python.org"}


def test_fast_router_blocks_dangerous_configured_site_name() -> None:
    route = route_fast_command("abre cmd", quick_sites={"cmd": "https://example.com"})

    assert route is not None
    assert route.kind == "denied"
    assert route.arguments is None


def test_fast_router_blocks_dangerous_configured_site_url() -> None:
    route = route_fast_command("abre seguro", quick_sites={"seguro": "javascript:alert(1)"})

    assert route is not None
    assert route.kind == "denied"
    assert route.arguments is None


def test_fast_router_builds_google_search_url() -> None:
    route = route_fast_command("pesquisa no google por gatos british shorthair")

    assert route is not None
    assert route.kind == "tool"
    assert route.tool_name == "open_url"
    assert route.arguments == {
        "url": "https://www.google.com/search?q=gatos+british+shorthair",
        "search_engine": "Google",
        "search_query": "gatos british shorthair",
    }


def test_fast_router_builds_youtube_search_url() -> None:
    route = route_fast_command("pesquisa no youtube por tutorial python tkinter")

    assert route is not None
    assert route.kind == "tool"
    assert route.tool_name == "open_url"
    assert route.arguments == {
        "url": "https://www.youtube.com/results?search_query=tutorial+python+tkinter",
        "search_engine": "YouTube",
        "search_query": "tutorial python tkinter",
    }


def test_fast_router_treats_dangerous_search_query_as_text() -> None:
    route = route_fast_command("pesquisa no google por file:///C:/Windows/System32/cmd.exe")

    assert route is not None
    assert route.kind == "tool"
    assert route.arguments == {
        "url": "https://www.google.com/search?q=file%3A%2F%2F%2FC%3A%2FWindows%2FSystem32%2Fcmd.exe",
        "search_engine": "Google",
        "search_query": "file:///C:/Windows/System32/cmd.exe",
    }
