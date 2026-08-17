"""Compile and run the standalone C++ gamepad profile/mapper contract tests."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).parents[2]


def test_cpp_gamepad_profile_device_selection_and_mapper(tmp_path: Path) -> None:
    """A compiled consumer must observe strict parsing, identity, and mapping behavior."""
    binary = tmp_path / "gamepad_profile_mapper_test"
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(ROOT / "simulate/src"),
            str(ROOT / "tests/simulate/gamepad_profile_mapper_test.cpp"),
            str(ROOT / "simulate/src/gamepad/gamepad_profile.cc"),
            str(ROOT / "simulate/src/gamepad/device_discovery.cc"),
            str(ROOT / "simulate/src/gamepad/logical_mapper.cc"),
            "-lyaml-cpp",
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    run_result = subprocess.run(
        [binary, ROOT / "tests/fixtures/gamepads/beitong_profile.yaml", tmp_path],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
def test_cpp_configured_joystick_and_linux_reader(tmp_path: Path) -> None:
    """The real adapter must drain coherently and preserve Unitree edge semantics."""
    binary = tmp_path / "configured_joystick_test"
    compile_result = subprocess.run(
        [
            "c++",
            "-std=c++17",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            "/usr/local/include",
            "-I",
            str(ROOT / "simulate/src"),
            str(ROOT / "tests/simulate/configured_joystick_test.cpp"),
            str(ROOT / "simulate/src/joystick/joystick.cc"),
            str(ROOT / "simulate/src/gamepad/gamepad_profile.cc"),
            str(ROOT / "simulate/src/gamepad/device_discovery.cc"),
            str(ROOT / "simulate/src/gamepad/logical_mapper.cc"),
            str(ROOT / "simulate/src/gamepad/configured_joystick.cc"),
            "-lyaml-cpp",
            "-o",
            str(binary),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert compile_result.returncode == 0, compile_result.stdout + compile_result.stderr

    run_result = subprocess.run(
        [binary, ROOT / "tests/fixtures/gamepads/beitong_profile.yaml"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert run_result.returncode == 0, run_result.stdout + run_result.stderr
