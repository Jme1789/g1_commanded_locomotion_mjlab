from __future__ import annotations

import argparse
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
import webbrowser
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import uvicorn

LOOPBACK_HOST = "127.0.0.1"
DEFAULT_PORT = 8766

_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))


class CalibratorLaunchError(RuntimeError):
    def __init__(self, *, code: str, message_zh: str) -> None:
        super().__init__(message_zh)
        self.code = code
        self.message_zh = message_zh


@dataclass(frozen=True, slots=True)
class CalibratorLaunchConfig:
    port: int = DEFAULT_PORT
    no_browser: bool = False

    def __post_init__(self) -> None:
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise ValueError("port must be in the range 1..65535")
        if type(self.no_browser) is not bool:
            raise TypeError("no_browser must be a boolean")


def _port_argument(raw_value: str) -> int:
    try:
        port = int(raw_value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("port must be an integer") from exc
    if not 1 <= port <= 65535:
        raise argparse.ArgumentTypeError("port must be in the range 1..65535")
    return port


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="启动仅限本机访问的 G1 游戏手柄校准器。",
    )
    parser.add_argument("--port", type=_port_argument, default=DEFAULT_PORT)
    parser.add_argument("--no-browser", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> CalibratorLaunchConfig:
    namespace = _argument_parser().parse_args(argv)
    return CalibratorLaunchConfig(
        port=namespace.port,
        no_browser=namespace.no_browser,
    )


def port_is_available(host: str, port: int) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind((host, port))
    except OSError:
        return False
    return True


def calibrator_health_is_ready(base_url: str) -> bool:
    try:
        with urllib.request.urlopen(
            f"{base_url}/api/v1/health",
            timeout=0.5,
        ) as response:
            return response.status == 200
    except (OSError, TimeoutError, urllib.error.URLError):
        return False


def start_browser_when_ready(
    url: str,
    *,
    health_probe: Callable[[str], bool] = calibrator_health_is_ready,
    browser_open: Callable[[str], bool] = webbrowser.open,
    printer: Callable[[str], None] = print,
    timeout: float = 15.0,
    interval: float = 0.1,
    sleep: Callable[[float], None] = time.sleep,
    monotonic: Callable[[], float] = time.monotonic,
) -> bool:
    deadline = monotonic() + timeout
    while True:
        if health_probe(url):
            try:
                opened = browser_open(url)
            except Exception as exc:  # noqa: BLE001 - desktop integration boundary
                printer(f"浏览器打开失败，请手动访问 {url}（{exc}）")
            else:
                if not opened:
                    printer(f"浏览器未打开，请手动访问 {url}")
            return True
        if monotonic() >= deadline:
            printer(f"校准器健康检查超时，请稍后手动访问 {url}")
            return False
        sleep(interval)


def run_calibrator(config: CalibratorLaunchConfig) -> None:
    if not port_is_available(LOOPBACK_HOST, config.port):
        raise CalibratorLaunchError(
            code="calibrator_port_in_use",
            message_zh=f"本地端口 {config.port} 已被占用。",
        )
    url = f"http://{LOOPBACK_HOST}:{config.port}"
    print(f"G1 游戏手柄校准器：{url}")
    print("此工具不会启动、停止或重启 MuJoCo 与控制器进程。")
    if not config.no_browser:
        threading.Thread(
            target=start_browser_when_ready,
            args=(url,),
            daemon=True,
            name="g1-calibrator-browser",
        ).start()
    uvicorn.run(
        "src.gamepad_calibrator.app:create_app",
        factory=True,
        host=LOOPBACK_HOST,
        port=config.port,
    )


def main(argv: Sequence[str] | None = None) -> int:
    config = parse_args(argv)
    try:
        run_calibrator(config)
    except CalibratorLaunchError as exc:
        print(exc.message_zh, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
