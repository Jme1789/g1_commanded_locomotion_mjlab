from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP_TEST = ROOT / "tests/deploy/custom_joystick_test.cpp"


def test_custom_joystick_mapping_and_disconnect(tmp_path: Path) -> None:
  binary = tmp_path / "custom_joystick_test"
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
      "-lboost_program_options",
      "-lyaml-cpp",
      "-lfmt",
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
  assert "all custom joystick tests passed" in run_result.stdout
