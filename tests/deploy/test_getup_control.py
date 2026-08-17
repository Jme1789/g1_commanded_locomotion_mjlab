from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CPP_TEST = ROOT / "tests/deploy/getup_control_test.cpp"


def test_getup_control(tmp_path: Path) -> None:
  binary = tmp_path / "getup_control_test"
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
  assert "all GetUp control tests passed" in run_result.stdout
