from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP_TEST = ROOT / "tests/deploy/gamepad_motion_mapping_test.cpp"


def test_gamepad_motion_mapping(tmp_path: Path) -> None:
  binary = tmp_path / "gamepad_motion_mapping_test"
  compile_result = subprocess.run(
    [
      "c++",
      "-std=c++17",
      "-Wall",
      "-Wextra",
      "-Werror",
      "-I",
      str(ROOT / "deploy/include"),
      str(CPP_TEST),
      "-o",
      str(binary),
    ],
    cwd=ROOT,
    check=False,
    capture_output=True,
    text=True,
  )
  assert compile_result.returncode == 0, compile_result.stderr

  run_result = subprocess.run(
    [str(binary)],
    cwd=ROOT,
    check=False,
    capture_output=True,
    text=True,
  )
  assert run_result.returncode == 0, run_result.stdout + run_result.stderr
  assert "all gamepad motion mapping tests passed" in run_result.stdout
