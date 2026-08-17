from __future__ import annotations

import pytest

import scripts.gamepad_calibrator as launcher
from scripts.gamepad_calibrator import (
    CalibratorLaunchConfig,
    CalibratorLaunchError,
    parse_args,
    run_calibrator,
    start_browser_when_ready,
)


def test_defaults_are_fixed_to_loopback_calibrator_port() -> None:
    assert parse_args([]) == CalibratorLaunchConfig(port=8766, no_browser=False)
    assert launcher.LOOPBACK_HOST == "127.0.0.1"


@pytest.mark.parametrize("port", [0, -1, 65536, 99999])
def test_port_must_be_in_tcp_range(port: int) -> None:
    with pytest.raises(SystemExit):
        parse_args(["--port", str(port)])


def test_only_port_and_no_browser_are_accepted() -> None:
    assert parse_args(["--port", "9001", "--no-browser"]) == CalibratorLaunchConfig(
        port=9001,
        no_browser=True,
    )
    with pytest.raises(SystemExit):
        parse_args(["--host", "0.0.0.0"])
    with pytest.raises(SystemExit):
        parse_args(["unexpected-positional"])


def test_uvicorn_runs_factory_on_fixed_loopback(monkeypatch, capsys) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    monkeypatch.setattr(launcher, "port_is_available", lambda host, port: True)
    monkeypatch.setattr(
        launcher.uvicorn,
        "run",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )

    run_calibrator(CalibratorLaunchConfig(port=9124, no_browser=True))

    assert calls == [
        (
            ("src.gamepad_calibrator.app:create_app",),
            {"factory": True, "host": "127.0.0.1", "port": 9124},
        )
    ]
    assert "http://127.0.0.1:9124" in capsys.readouterr().out


def test_occupied_port_is_rejected_before_uvicorn(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(launcher, "port_is_available", lambda host, port: False)
    monkeypatch.setattr(launcher.uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))

    with pytest.raises(CalibratorLaunchError) as raised:
        run_calibrator(CalibratorLaunchConfig(port=8766, no_browser=True))

    assert raised.value.code == "calibrator_port_in_use"
    assert calls == []


def test_port_probe_uses_only_requested_loopback_address(monkeypatch) -> None:
    calls = []

    class FakeSocket:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def setsockopt(self, level, option, value):
            calls.append(("setsockopt", level, option, value))

        def bind(self, address):
            calls.append(("bind", address))

    monkeypatch.setattr(launcher.socket, "socket", lambda *args: FakeSocket())

    assert launcher.port_is_available("127.0.0.1", 8766) is True
    assert calls[-1] == ("bind", ("127.0.0.1", 8766))


def test_browser_opens_only_after_health_is_ready() -> None:
    probes = iter([False, False, True])
    browser_calls = []
    sleeps = []

    ready = start_browser_when_ready(
        "http://127.0.0.1:8766",
        health_probe=lambda url: next(probes),
        browser_open=lambda url: browser_calls.append(url) or True,
        sleep=lambda seconds: sleeps.append(seconds),
        timeout=1.0,
        monotonic=iter([0.0, 0.1, 0.2, 0.3]).__next__,
    )

    assert ready is True
    assert browser_calls == ["http://127.0.0.1:8766"]
    assert len(sleeps) == 2


@pytest.mark.parametrize("browser_result", [False, RuntimeError("no desktop")])
def test_browser_failure_prints_manual_url_without_raising(browser_result) -> None:
    messages = []

    def browser_open(url: str) -> bool:
        if isinstance(browser_result, Exception):
            raise browser_result
        return browser_result

    ready = start_browser_when_ready(
        "http://127.0.0.1:8766",
        health_probe=lambda url: True,
        browser_open=browser_open,
        printer=messages.append,
    )

    assert ready is True
    assert any("手动访问" in message and "http://127.0.0.1:8766" in message for message in messages)
