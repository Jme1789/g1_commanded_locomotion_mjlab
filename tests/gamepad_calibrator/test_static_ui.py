from __future__ import annotations

from html.parser import HTMLParser
from pathlib import Path

from fastapi.testclient import TestClient

from src.gamepad_calibrator.app import create_app

REQUIRED_IDS = {
    "app-header",
    "process-interlock",
    "connection-status",
    "device-details",
    "mapping-grid",
    "mapping-guidance",
    "raw-monitor",
    "raw-axes",
    "raw-buttons",
    "raw-events",
    "logical-preview",
    "logical-levels",
    "edge-events",
    "combo-preview",
    "save-profile",
    "profile-list",
    "profile-import",
    "profile-export",
    "profile-activate",
    "restart-banner",
    "status-announcements",
    "confirmation-dialog",
}

FORBIDDEN_IDS = {
    "template-select",
    "neutral-step",
    "candidate-panel",
    "candidate-choices",
    "manual-editor",
    "manual-form",
    "preview-checklist",
    "preview-confirmations",
}


class ShellParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.html_lang: str | None = None
        self.ids: set[str] = set()
        self.assets: list[str] = []
        self.script_sources: list[str] = []
        self.input_types: list[str] = []
        self.tags: list[str] = []
        self.buttons: list[dict[str, str | None]] = []
        self.inline_handlers: list[str] = []
        self.live_regions: list[str] = []
        self.inline_scripts = 0
        self._script_src: str | None = None

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        self.tags.append(tag)
        if tag == "html":
            self.html_lang = attributes.get("lang")
        if element_id := attributes.get("id"):
            self.ids.add(element_id)
        if tag == "script":
            self._script_src = attributes.get("src")
            if self._script_src:
                self.assets.append(self._script_src)
                self.script_sources.append(self._script_src)
            else:
                self.inline_scripts += 1
        if (
            tag == "link"
            and attributes.get("rel") == "stylesheet"
            and (href := attributes.get("href"))
        ):
            self.assets.append(href)
        if tag == "button":
            self.buttons.append(attributes)
        if tag == "input":
            self.input_types.append(attributes.get("type") or "text")
        self.inline_handlers.extend(
            name for name in attributes if name.lower().startswith("on")
        )
        if aria_live := attributes.get("aria-live"):
            self.live_regions.append(aria_live)


def _assert_security_headers(response) -> None:
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["x-frame-options"] == "DENY"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-cache"
    csp = response.headers["content-security-policy"]
    assert "default-src 'self'" in csp
    assert "connect-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp


def test_served_shell_is_local_chinese_semantic_and_accessible() -> None:
    with TestClient(create_app()) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "http://" not in response.text
    assert "https://" not in response.text

    parser = ShellParser()
    parser.feed(response.text)
    assert parser.html_lang == "zh-CN"
    assert REQUIRED_IDS <= parser.ids
    assert parser.script_sources == ["/static/quick_app.js"]
    assert "/static/app.js" not in parser.assets
    assert "select" not in parser.tags
    assert "radio" not in parser.input_types
    assert parser.ids.isdisjoint(FORBIDDEN_IDS)
    assert "只观察和映射输入" in response.text
    assert "不会控制 MuJoCo、g1_ctrl 或机器人" in response.text
    assert parser.assets
    assert all(asset.startswith("/static/") for asset in parser.assets)
    assert parser.inline_scripts == 0
    assert parser.inline_handlers == []
    assert parser.buttons
    assert all(button.get("type") in {"button", "submit"} for button in parser.buttons)
    assert parser.live_regions


def test_every_declared_local_asset_is_served_with_security_headers() -> None:
    with TestClient(create_app()) as client:
        shell = client.get("/")
        parser = ShellParser()
        parser.feed(shell.text)
        responses = [shell, *(client.get(asset) for asset in parser.assets)]

    assert len(responses) >= 3
    assert all(response.status_code == 200 for response in responses)
    for response in responses:
        _assert_security_headers(response)


def test_advanced_calibration_artifacts_remain_served_but_unlinked() -> None:
    with TestClient(create_app()) as client:
        legacy_shell = client.get("/static/legacy_index.html")
        legacy_app = client.get("/static/app.js")

    assert legacy_shell.status_code == 200
    assert legacy_app.status_code == 200
    _assert_security_headers(legacy_shell)
    _assert_security_headers(legacy_app)

    legacy_node_suite = Path("tests/gamepad_calibrator/ui_behavior_test.mjs")
    assert legacy_node_suite.is_file()
    assert '../../src/gamepad_calibrator/static/app.js' in legacy_node_suite.read_text()


def test_api_keeps_security_headers_and_generated_docs_disabled() -> None:
    with TestClient(create_app()) as client:
        health = client.get("/api/v1/health")
        docs = [client.get(path) for path in ("/docs", "/redoc", "/openapi.json")]

    assert health.status_code == 200
    _assert_security_headers(health)
    assert [response.status_code for response in docs] == [404, 404, 404]
